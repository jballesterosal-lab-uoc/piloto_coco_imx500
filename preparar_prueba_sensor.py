"""Prepara las 45 imágenes de test y sus referencias para la inyección en el IMX500."""

import argparse
import csv
import json
from importlib.metadata import version
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZipFile

import mct_quantizers as mctq
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from evaluar_test_sensor import classification_metrics, file_sha256, write_report
from modelo import CocoCrops, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datos"))
    parser.add_argument("--images", type=Path, default=Path("datos/val2017"))
    parser.add_argument("--run", type=Path, default=Path("resultados/piloto"))
    parser.add_argument("--output", type=Path, default=Path("resultados/test_sensor"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    labels = (args.data / "labels.txt").read_text(encoding="utf-8").splitlines()
    manifest = args.data / "muestras.csv"
    checkpoint = torch.load(args.run / "fp32.pt", map_location="cpu", weights_only=True)
    if checkpoint["manifest_sha256"] != file_sha256(manifest) or checkpoint["labels"] != labels:
        raise ValueError("El manifiesto o las etiquetas difieren del checkpoint original")
    with manifest.open(encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["split"] == "test"]
    if len(rows) != 45:
        raise ValueError("Se esperaban las 45 imágenes de test del piloto")
    crops = CocoCrops(rows, args.images)
    model = build_model(len(labels)).eval()
    model.load_state_dict(checkpoint["weights"])
    options = mctq.get_ort_session_options()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.run / "imx500.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    packer = args.run / "conversion" / "packerOut.zip"
    with ZipFile(packer) as archive:
        params = ElementTree.fromstring(archive.read("dnnParams.xml"))
    input_tensor = params.find(".//inputTensor")
    dimensions = [int(item.attrib["size"]) for item in input_tensor.findall("dimensions/dimension")]
    scale, shift = float(input_tensor.attrib["scale"]), int(input_tensor.attrib["shift"])
    if (
        dimensions != [3, 160, 160]
        or input_tensor.attrib["format"] != "unsigned"
        or input_tensor.attrib["bitsPerElement"] != "8"
        or shift != 0
        or scale != 1 / 256
    ):
        raise ValueError("El conversor debe describir la entrada del piloto: RGB uint8, 160 × 160, escala 1/256 y desplazamiento 0")
    args.output.mkdir(parents=True, exist_ok=False)
    test = args.output / "test"
    test.mkdir()
    samples = []
    for row, (tensor, expected) in zip(rows, crops):
        real_input = tensor.unsqueeze(0)
        # Usa la escala del conversor. Una entrada blanca se satura a 255/256.
        quantized = np.clip(np.rint(tensor.numpy() / scale), 0, 255).astype(np.uint8)
        injected_input = quantized.astype(np.float32)[None] * scale
        tensor_path = test / f"{int(row['image_id']):012d}.bin"
        tensor_path.write_bytes(quantized.tobytes())
        png = np.rint(tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(png).save(tensor_path.with_suffix(".png"))
        with torch.no_grad():
            fp32 = model(real_input).numpy()[0]
        onnx = session.run(None, {input_name: real_input.numpy()})[0][0]
        onnx_injected = session.run(None, {input_name: injected_input})[0][0]
        sample = {
            "image_id": int(row["image_id"]), "annotation_id": int(row["annotation_id"]),
            "expected_index": expected, "tensor_file": tensor_path.name,
            "tensor_sha256": file_sha256(tensor_path),
        }
        for backend, logits in (("fp32", fp32), ("onnx", onnx), ("onnx_injected_input", onnx_injected)):
            if logits.shape != (3,) or not np.isfinite(logits).all():
                raise ValueError(f"Salida inválida de {backend} para {row['image_id']}")
            sample[backend] = {"logits": logits.tolist(), "predicted_index": int(logits.argmax())}
        samples.append(sample)
    expected = [sample["expected_index"] for sample in samples]
    references = {
        name: classification_metrics(expected, [sample[name]["predicted_index"] for sample in samples], labels)
        for name in ("fp32", "onnx", "onnx_injected_input")
    }
    for name, result_file in (("fp32", "fp32.json"), ("onnx", "ptq.json")):
        original = json.loads((args.run / result_file).read_text(encoding="utf-8"))
        if references[name]["confusion_expected_by_predicted"] != original["confusion_expected_by_predicted"]:
            raise ValueError(f"La evaluación {name} difiere de la original. Revisa las imágenes y el entorno")
    dataset = {
        "labels": labels, "manifest_sha256": file_sha256(manifest), "input_shape_chw": [3, 160, 160],
        "input_format": "RGB", "input_dtype": "uint8", "input_scale": scale, "input_shift": shift,
        "packer_sha256": file_sha256(packer), "fp32_sha256": file_sha256(args.run / "fp32.pt"),
        "onnx_sha256": file_sha256(args.run / "imx500.onnx"),
        "versions": {name: version(name) for name in ("torch", "torchvision", "Pillow", "onnxruntime", "mct-quantizers")},
        "reference_metrics": references, "samples": samples,
    }
    write_report(test / "test.json", dataset)
    source = Path(__file__).resolve().parent
    with ZipFile(args.output / "prueba_test_imx500.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(test.iterdir()):
            archive.write(path, f"test_sensor/test/{path.name}")
        for filename in ("evaluar_test_sensor.py", "input_format.json", "PRUEBA_SENSOR.md"):
            archive.write(source / filename, f"test_sensor/{filename}")
        archive.write(packer, "test_sensor/packerOut.zip")
    print(json.dumps(references, indent=2))
    print(f"Paquete preparado: {args.output / 'prueba_test_imx500.zip'}")


if __name__ == "__main__":
    main()
