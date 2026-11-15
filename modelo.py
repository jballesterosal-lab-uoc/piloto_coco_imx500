import argparse
import csv
import hashlib
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


INPUT_SIZE = 160
TPC_VERSION = "4.0"


class CocoCrops(Dataset):
    def __init__(self, rows, images, augment=False):
        self.rows = rows
        self.images = images
        operations = [transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
        if augment:
            operations.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose([*operations, transforms.ToTensor()])
        missing = [row["filename"] for row in rows if not (images / row["filename"]).is_file()]
        if missing:
            raise FileNotFoundError(f"Faltan {len(missing)} imágenes en {images}; primera: {missing[0]}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        crop = tuple(int(row[key]) for key in ("left", "top", "right", "bottom"))
        with Image.open(self.images / row["filename"]) as image:
            if image.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"Dimensiones incorrectas: {row['filename']}")
            tensor = self.transform(image.convert("RGB").crop(crop))
        return tensor, int(row["class_index"])


def build_model(num_classes, pretrained=False):
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[-1] = nn.Linear(model.last_channel, num_classes)
    return model


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def score(loader, predict, labels):
    count = len(labels)
    confusion = torch.zeros((count, count), dtype=torch.int64)
    with torch.no_grad():
        for images, expected in loader:
            logits = predict(images).cpu()
            if logits.shape != (len(images), count) or not torch.isfinite(logits).all():
                raise ValueError(f"Salida inesperada: {tuple(logits.shape)} para {count} clases")
            pairs = count * expected + logits.argmax(dim=1)
            confusion += torch.bincount(pairs, minlength=count**2).reshape(count, count)
    correct = confusion.diag()
    support = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    precision = correct / predicted.clamp(min=1)
    recall = correct / support.clamp(min=1)
    f1 = 2 * correct / (predicted + support).clamp(min=1)
    return {
        "accuracy": correct.sum().item() / confusion.sum().item(),
        "f1_macro": f1.to(torch.float64).mean().item(),
        "labels": labels,
        "confusion_expected_by_predicted": confusion.tolist(),
        "per_class": {
            label: {"precision": precision[i].item(), "recall": recall[i].item(),
                    "f1": f1[i].item(), "support": support[i].item()}
            for i, label in enumerate(labels)
        },
    }


def train(args, data, labels, rows, digest):
    seed = data["seed"]
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    training = DataLoader(CocoCrops(rows["train"], args.images, augment=True), batch_size=16, shuffle=True,
                          generator=torch.Generator().manual_seed(seed))
    validation = DataLoader(CocoCrops(rows["validation"], args.images), batch_size=16)
    test = DataLoader(CocoCrops(rows["test"], args.images), batch_size=16)
    model = build_model(len(labels), pretrained=True).to(device)
    late_layers = model.features[-4:]
    optimizer = torch.optim.AdamW(
        [{"params": model.classifier.parameters(), "lr": 1e-3},
         {"params": late_layers.parameters(), "lr": 2e-4}]
    )
    loss_function = nn.CrossEntropyLoss()
    best_f1 = -1.0
    history = []
    args.run.mkdir(parents=True, exist_ok=True)
    if (args.run / "fp32.pt").exists():
        raise FileExistsError(f"Ya existe un modelo en {args.run}; usa otro directorio --run")
    for epoch in range(1, args.epochs + 1):
        model.features.requires_grad_(False)
        late_layers.requires_grad_(epoch > 2)
        model.train()
        # Conserva las estadísticas BatchNorm de ImageNet con este conjunto pequeño.
        model.features.eval()
        total_loss = 0.0
        for images, expected in training:
            optimizer.zero_grad()
            loss = loss_function(model(images.to(device)), expected.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        model.eval()
        val = score(validation, lambda images: model(images.to(device)), labels)
        history.append({"epoch": epoch, "train_loss": total_loss / len(training.dataset),
                        "validation_f1_macro": val["f1_macro"]})
        print(history[-1], flush=True)
        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            best_epoch = epoch
            best_weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    torch.save({"weights": best_weights, "labels": labels, "manifest_sha256": digest}, args.run / "fp32.pt")
    model.load_state_dict(best_weights)
    result = score(test, lambda images: model(images.to(device)), labels)
    write_json(args.run / "fp32.json", result)
    write_json(args.run / "entrenamiento.json", {
        "seed": seed, "best_epoch": best_epoch, "hardware": str(device), "input_size": INPUT_SIZE,
        "pretrained": "IMAGENET1K_V1", "history": history, "manifest_sha256": digest,
        "python": platform.python_version(), "torch": version("torch"), "torchvision": version("torchvision"),
    })
    shutil.copyfile(args.data / "labels.txt", args.run / "labels.txt")
    print(f"Prueba FP32: F1 macro={result['f1_macro']:.4f}; exactitud={result['accuracy']:.4f}")


def quantize(args, labels, rows, digest):
    import mct_quantizers as mctq
    import model_compression_toolkit as mct
    import onnxruntime as ort
    from edgemdt_tpc import get_target_platform_capabilities

    checkpoint = torch.load(args.run / "fp32.pt", map_location="cpu", weights_only=True)
    if checkpoint["manifest_sha256"] != digest or checkpoint["labels"] != labels:
        raise ValueError("Las muestras o el orden de etiquetas difieren del entrenamiento")
    fp32 = json.loads((args.run / "fp32.json").read_text(encoding="utf-8"))
    calibration = DataLoader(CocoCrops(rows["calibration"], args.images), batch_size=1)
    test = DataLoader(CocoCrops(rows["test"], args.images), batch_size=1)
    model = build_model(len(labels)).eval()
    model.load_state_dict(checkpoint["weights"])

    def representative_data():
        for images, _ in calibration:
            yield [images]

    target = get_target_platform_capabilities(tpc_version=TPC_VERSION, device_type="imx500")
    core = mct.core.CoreConfig(quantization_config=mct.core.QuantizationConfig(z_threshold=16))
    quantized, _ = mct.ptq.pytorch_post_training_quantization(
        in_module=model, representative_data_gen=representative_data,
        core_config=core, target_platform_capabilities=target,
    )
    path = args.run / "imx500.onnx"
    partial = args.run / "imx500_temporal.onnx"
    mct.exporter.pytorch_export_model(
        model=quantized, save_model_path=str(partial), repr_dataset=representative_data, output_names=["logits"],
    )
    options = mctq.get_ort_session_options()
    options.intra_op_num_threads = args.threads
    session = ort.InferenceSession(str(partial), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def predict(images):
        return torch.from_numpy(session.run(None, {input_name: images.numpy()})[0])

    result = score(test, predict, labels)
    result["delta_f1_macro_ptq_minus_fp32"] = result["f1_macro"] - fp32["f1_macro"]
    result["calibration_samples"] = len(calibration.dataset)
    result["tpc"] = TPC_VERSION
    result["versions"] = {name: version(name) for name in (
        "model-compression-toolkit", "edge-mdt-tpc", "mct-quantizers", "onnxruntime",
    )}
    partial.replace(path)
    write_json(args.run / "ptq.json", result)
    print(f"Prueba PTQ: F1 macro={result['f1_macro']:.4f}; exactitud={result['accuracy']:.4f}; "
          f"ΔF1={result['delta_f1_macro_ptq_minus_fp32']:+.4f}")


def main():
    parser = argparse.ArgumentParser(description="Entrena y cuantiza MobileNetV2 para IMX500")
    parser.add_argument("stage", choices=("train", "quantize"))
    parser.add_argument("--data", type=Path, default=Path("datos"))
    parser.add_argument("--images", type=Path, default=Path("datos/val2017"))
    parser.add_argument("--run", type=Path, default=Path("resultados/piloto"))
    parser.add_argument("--epochs", type=int, default=10, help="Solo entrenamiento")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Solo entrenamiento")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.threads) < 1:
        parser.error("Épocas e hilos deben ser positivos")
    torch.set_num_threads(args.threads)
    data = json.loads((args.data / "dataset.json").read_text(encoding="utf-8"))
    labels = (args.data / "labels.txt").read_text(encoding="utf-8").splitlines()
    if labels != data["labels"]:
        raise ValueError("El orden de las etiquetas de dataset.json y labels.txt difiere")
    manifest = args.data / "muestras.csv"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = {name: [] for name in ("train", "validation", "calibration", "test")}
    with manifest.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            rows[row["split"]].append(row)
    if args.stage == "train":
        train(args, data, labels, rows, digest)
    else:
        quantize(args, labels, rows, digest)


if __name__ == "__main__":
    main()
