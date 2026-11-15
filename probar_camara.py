import argparse
import json
import time
from collections import Counter
from pathlib import Path


def capture(args, labels):
    import numpy as np
    from picamera2 import Picamera2
    from picamera2.devices.imx500 import IMX500

    camera = IMX500(str(args.model))
    width, height = camera.get_input_size()
    if width != height:
        raise ValueError(f"La prueba espera una entrada cuadrada, recibió {(width, height)}")
    intrinsics = camera.network_intrinsics
    if intrinsics is not None and intrinsics.task not in (None, "classification"):
        raise ValueError(f"El paquete declara la tarea {intrinsics.task!r}")
    records = []
    with Picamera2(camera.camera_num) as picam:
        config = picam.create_preview_configuration(main={"size": (640, 480)})
        camera.show_network_fw_progress_bar()
        picam.start(config, show_preview=args.preview)
        try:
            camera.set_inference_aspect_ratio((width, height))
            start = time.monotonic()
            while len(records) < args.samples and time.monotonic() - start < args.timeout:
                metadata = picam.capture_metadata()
                outputs = camera.get_outputs(metadata)
                if not outputs or time.monotonic() - start < args.warmup:
                    continue
                logits = np.asarray(outputs[0], dtype=np.float64).reshape(-1)
                if len(outputs) != 1 or len(logits) != len(labels) or not np.isfinite(logits).all():
                    raise ValueError(f"Salida incompatible con las {len(labels)} etiquetas")
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                winner = int(probabilities.argmax())
                timestamp = metadata.get("SensorTimestamp")
                kpi = camera.get_kpi_info(metadata)
                records.append({
                    "sensor_timestamp_ns": int(timestamp) if timestamp is not None else None,
                    "host_timestamp_s": time.monotonic() - start,
                    "predicted": labels[winner], "logits": logits.tolist(),
                    "softmax": probabilities.tolist(), "dnn_ms": kpi[0] if kpi else None,
                })
        finally:
            picam.stop()
    if len(records) < args.samples:
        raise RuntimeError(f"Solo se recibieron {len(records)}/{args.samples} salidas en {args.timeout} s")
    timestamps = [record["sensor_timestamp_ns"] for record in records]
    if all(timestamp is not None for timestamp in timestamps) and timestamps[-1] > timestamps[0]:
        elapsed = (timestamps[-1] - timestamps[0]) * 1e-9
        rate_clock = "sensor_timestamp"
    else:
        elapsed = records[-1]["host_timestamp_s"] - records[0]["host_timestamp_s"]
        rate_clock = "host_monotonic"
    dnn_times = [record["dnn_ms"] for record in records if record["dnn_ms"] is not None]
    counts = Counter(record["predicted"] for record in records)
    return {
        "model": str(args.model), "labels": labels, "input_size": [width, height],
        "expected": args.expected, "sample_count": len(records), "predicted_counts": dict(counts),
        "expected_fraction": counts[args.expected] / len(records) if args.expected else None,
        "outputs_per_s": (len(records) - 1) / elapsed if elapsed > 0 else None, "rate_clock": rate_clock,
        "dnn_kpi_samples": len(dnn_times),
        "dnn_ms_median": float(np.median(dnn_times)) if dnn_times else None,
        "dnn_ms_p95": float(np.percentile(dnn_times, 95)) if dnn_times else None,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Comprueba la inferencia de un clasificador en el sensor IMX500")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected", help="Etiqueta del objeto situado en la región central")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--warmup", type=float, default=3)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--preview", action="store_true", help="Muestra la cámara si hay escritorio")
    args = parser.parse_args()
    if args.samples < 2 or not 0 <= args.warmup < args.timeout:
        parser.error("Se necesitan al menos dos muestras y un calentamiento menor que el tiempo límite")
    if not args.model.is_file():
        parser.error(f"No existe el paquete {args.model}")
    labels = args.labels.read_text(encoding="utf-8").splitlines()
    if len(labels) < 2 or len(set(labels)) != len(labels) or not all(labels):
        parser.error("El fichero debe contener una etiqueta distinta por línea")
    if args.expected is not None and args.expected not in labels:
        parser.error(f"La etiqueta esperada debe ser una de {', '.join(labels)}")
    print("Coloca un objeto en el centro de la imagen y mantenlo durante la captura.", flush=True)
    result = capture(args, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value for name, value in result.items() if name != "records"}, indent=2))


if __name__ == "__main__":
    main()
