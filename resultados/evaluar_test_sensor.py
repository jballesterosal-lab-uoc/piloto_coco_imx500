"""Evalúa los recortes del piloto mediante inyección en el IMX500."""

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classification_metrics(expected, predicted, labels):
    count = len(labels)
    confusion = np.zeros((count, count), dtype=np.int64)
    np.add.at(confusion, (expected, predicted), 1)
    correct = np.diag(confusion)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    precision = correct / np.maximum(predicted_count, 1)
    recall = correct / np.maximum(support, 1)
    f1 = 2 * correct / np.maximum(predicted_count + support, 1)
    return {
        "sample_count": len(expected),
        "accuracy": float(correct.sum() / len(expected)),
        "f1_macro": float(f1.mean()),
        "labels": labels,
        "confusion_expected_by_predicted": confusion.tolist(),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(labels)
        },
    }


def load_test(folder):
    dataset = json.loads((folder / "test.json").read_text(encoding="utf-8"))
    samples = dataset["samples"]
    labels = dataset["labels"]
    if labels != ["bottle", "banana", "book"] or len(samples) != 45:
        raise ValueError("El paquete debe contener las tres clases y las 45 imágenes de prueba del piloto")
    if (
        dataset["input_shape_chw"] != [3, 160, 160]
        or dataset["input_format"] != "RGB"
        or dataset["input_dtype"] != "uint8"
        or dataset["input_scale"] != 1 / 256
        or dataset["input_shift"] != 0
    ):
        raise ValueError("El formato de los tensores difiere de la entrada del piloto")
    if len({sample["image_id"] for sample in samples}) != len(samples):
        raise ValueError("Hay imágenes de origen repetidas")
    payloads = []
    for sample in samples:
        path = (folder / sample["tensor_file"]).resolve()
        if folder.resolve() not in path.parents or file_sha256(path) != sample["tensor_sha256"]:
            raise ValueError(f"Tensor incorrecto: {sample['tensor_file']}")
        payload = path.read_bytes()
        if len(payload) != 3 * 160 * 160 or sample["expected_index"] not in range(len(labels)):
            raise ValueError(f"Muestra incompatible: {sample['image_id']}")
        payloads.append(payload)
    if np.bincount([sample["expected_index"] for sample in samples], minlength=3).tolist() != [15, 15, 15]:
        raise ValueError("La prueba debe contener 15 imágenes de cada clase")
    return dataset, payloads


# Errores transitorios del driver al fijar el tensor: EAGAIN (11) y EREMOTEIO (121).
# La documentación de inyección de Sony los describe y su biblioteca reintenta la inyección.
TRANSIENT_INJECTION_ERRNOS = {11, 121}


def set_input_with_retry(camera, payload, io_retries, pause=0.05):
    """Fija el tensor de entrada y devuelve el fotograma de comparación y los reintentos de E/S."""
    # Esta API privada es la usada por el ejemplo oficial de Picamera2.
    set_input = camera._IMX500__set_input_tensor
    for io_attempt in range(1, io_retries + 1):
        descriptor = os.memfd_create("coco_test_tensor", os.MFD_CLOEXEC)
        try:
            with os.fdopen(os.dup(descriptor), "wb") as tensor_file:
                tensor_file.write(payload)
            try:
                set_input(descriptor)
            except RuntimeError as error:
                # Picamera2 encadena el OSError original en __cause__.
                cause = error.__cause__
                if isinstance(cause, OSError) and cause.errno in TRANSIENT_INJECTION_ERRNOS and io_attempt < io_retries:
                    time.sleep(pause)
                    continue
                raise
            return camera.get_injection_cmp_frm(), io_attempt - 1
        finally:
            os.close(descriptor)
    raise RuntimeError("No se pudo fijar el tensor de entrada")


def inject_and_wait(camera, picam, payload, class_count, timeout, retries, io_retries):
    # Esta API privada es la usada por el ejemplo oficial de Picamera2.
    parse_info = camera._IMX500__get_output_tensor_info
    started = time.monotonic()
    last_reason = "sin resultado asociado a la inyección"
    io_retry_count = 0
    for attempt in range(1, retries + 1):
        # get_injection_cmp_frm devuelve 0 si falla la lectura (Picamera2 lo indica en consola).
        raw_target, used_io_retries = set_input_with_retry(camera, payload, io_retries)
        io_retry_count += used_io_retries
        if not isinstance(raw_target, int) or raw_target < 0:
            raise RuntimeError(f"Contador de inyección inesperado: {raw_target!r}")
        target = raw_target & 0xFF
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                metadata = picam.capture_metadata(wait=max(0.001, deadline - time.monotonic()))
            except (TimeoutError, FutureTimeoutError):
                last_reason = f"no llegó ningún fotograma en {timeout} s"
                break
            info = metadata.get("CnnOutputTensorInfo")
            if info is None:
                continue
            frame_count = parse_info(info)["frameCount"]
            difference = (frame_count - target + 128) % 256 - 128
            if difference < 0:
                continue
            if difference > 0:
                last_reason = f"se perdió el fotograma {target}, se recibió {frame_count}"
                break
            outputs = camera.get_outputs(metadata)
            if outputs is None:
                continue
            logits = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
            if len(outputs) != 1 or logits.shape != (class_count,) or not np.isfinite(logits).all():
                raise ValueError("El sensor devolvió una salida incompatible con el clasificador")
            kpi = camera.get_kpi_info(metadata)
            return {
                "logits": logits.tolist(), "predicted_index": int(logits.argmax()),
                "injection_frame": target, "output_frame": int(frame_count), "attempts": attempt,
                "io_retries": io_retry_count,
                "dnn_ms": float(kpi[0]) if kpi is not None else None,
                "injection_to_result_ms": 1000 * (time.monotonic() - started),
            }
    raise TimeoutError(f"Inyección sin resultado tras {retries} intentos: {last_reason}")


def write_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate(args, dataset, payloads, report):
    from picamera2 import Picamera2
    from picamera2.devices.imx500 import IMX500

    required = ("_IMX500__set_input_tensor", "_IMX500__get_output_tensor_info", "get_injection_cmp_frm")
    if not all(hasattr(IMX500, name) for name in required):
        raise RuntimeError("Esta versión de Picamera2 carece de las APIs de inyección requeridas. Consulta PRUEBA_SENSOR.md")
    camera = IMX500(str(args.model), tensor_injection=True)
    config = camera.config["input_tensor"]
    if camera.get_input_size() != (160, 160) or config["dtype"] != "unsigned" or config["input_format"] != "RGB":
        raise ValueError("El RPK debe corresponder al piloto RGB de 160 × 160 y entrada uint8")
    report["sensor_input_config"] = config
    try:
        report["picamera2_version"] = version("picamera2")
    except PackageNotFoundError:
        report["picamera2_version"] = None
    with Picamera2(camera.camera_num) as picam:
        configuration = picam.create_preview_configuration(
            main={"size": (640, 480)}, controls={"FrameRate": args.frame_rate, "CnnEnableInputTensor": True},
            buffer_count=30, queue=False,
        )
        camera.show_network_fw_progress_bar()
        picam.start(configuration, show_preview=False)
        try:
            # Espera la primera captura para que el firmware esté preparado.
            picam.capture_metadata(wait=args.startup_timeout)
            for index, (sample, payload) in enumerate(zip(dataset["samples"], payloads), start=1):
                result = inject_and_wait(
                    camera, picam, payload, len(dataset["labels"]), args.timeout, args.retries, args.io_retries
                )
                result.update({"image_id": sample["image_id"], "expected_index": sample["expected_index"]})
                report["samples"].append(result)
                write_report(args.output, report)
                expected = dataset["labels"][sample["expected_index"]]
                predicted = dataset["labels"][result["predicted_index"]]
                print(f"{index:02d}/45 imagen={sample['image_id']} real={expected} predicción={predicted}", flush=True)
        finally:
            picam.stop()


def finish_report(report, dataset):
    records = report["samples"]
    if [record["image_id"] for record in records] != [sample["image_id"] for sample in dataset["samples"]]:
        raise ValueError("La secuencia de resultados no coincide con las 45 muestras de prueba")
    expected = [sample["expected_index"] for sample in records]
    predicted = [sample["predicted_index"] for sample in records]
    report["metrics"] = classification_metrics(expected, predicted, dataset["labels"])
    report["comparison"] = {}
    for backend, metrics in dataset["reference_metrics"].items():
        reference = [sample[backend]["predicted_index"] for sample in dataset["samples"]]
        disagreements = [sample["image_id"] for sample, ref, pred in zip(dataset["samples"], reference, predicted) if ref != pred]
        report["comparison"][backend] = {
            "reference_accuracy": metrics["accuracy"], "reference_f1_macro": metrics["f1_macro"],
            "delta_f1_sensor_minus_reference": report["metrics"]["f1_macro"] - metrics["f1_macro"],
            "prediction_agreement": 1 - len(disagreements) / len(records),
            "disagreement_image_ids": disagreements,
            "logits_mean_absolute_error": float(np.mean([
                np.abs(np.array(record["logits"]) - sample[backend]["logits"]).mean()
                for record, sample in zip(records, dataset["samples"])
            ])),
        }
    dnn = [sample["dnn_ms"] for sample in records if sample["dnn_ms"] is not None]
    report["dnn_ms_median"] = float(np.median(dnn)) if dnn else None
    report["dnn_ms_p95"] = float(np.percentile(dnn, 95)) if dnn else None
    report["status"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="RPK empaquetado a partir del packerOut.zip incluido")
    parser.add_argument("--test", type=Path, default=Path("test"))
    parser.add_argument("--output", type=Path, default=Path("resultado_sensor.json"))
    parser.add_argument("--timeout", type=float, default=15, help="Segundos por intento de inyección")
    parser.add_argument("--startup-timeout", type=float, default=600, help="Con bypass-cache la carga del firmware puede tardar varios minutos")
    parser.add_argument("--retries", type=int, default=10, help="Reinyecciones por fotograma perdido")
    parser.add_argument("--io-retries", type=int, default=50, help="Reintentos por error transitorio al fijar el tensor")
    parser.add_argument("--frame-rate", type=float, default=12, help="Sony indica un mínimo de 5 para la inyección")
    args = parser.parse_args()
    if min(args.timeout, args.startup_timeout) <= 0 or min(args.retries, args.io_retries) < 1:
        parser.error("Los tiempos y el número de intentos deben ser positivos")
    if args.frame_rate < 5:
        parser.error("La inyección requiere --frame-rate de al menos 5")
    if not args.model.is_file():
        parser.error(f"No existe {args.model}")
    dataset, payloads = load_test(args.test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserva un resultado nuevo para conservar las pruebas anteriores.
    with args.output.open("x", encoding="utf-8"):
        pass
    report = {
        "status": "in_progress", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": file_sha256(args.model), "test_sha256": file_sha256(args.test / "test.json"),
        "model": str(args.model), "python": platform.python_version(), "platform": platform.platform(),
        "labels": dataset["labels"], "expected_sample_count": len(payloads), "samples": [],
        "settings": {"frame_rate": args.frame_rate, "timeout_s": args.timeout,
                     "retries": args.retries, "io_retries": args.io_retries},
    }
    write_report(args.output, report)
    try:
        evaluate(args, dataset, payloads, report)
        finish_report(report, dataset)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        write_report(args.output, report)
        raise
    write_report(args.output, report)
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "expected", "sensor", "fp32", "onnx", "onnx_injected_input", "dnn_ms"])
        for sample, result in zip(dataset["samples"], report["samples"]):
            indices = [sample["expected_index"], result["predicted_index"]]
            indices.extend(sample[name]["predicted_index"] for name in ("fp32", "onnx", "onnx_injected_input"))
            writer.writerow([sample["image_id"], *(dataset["labels"][i] for i in indices), result["dnn_ms"]])
    print(json.dumps({key: value for key, value in report.items() if key not in ("samples", "sensor_input_config")}, indent=2))


if __name__ == "__main__":
    main()
