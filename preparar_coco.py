import argparse
import csv
import hashlib
import io
import json
import math
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


LABELS = ("bottle", "banana", "book")
SPLITS = ("train", "validation", "calibration", "test")
VAL_JSON = "instances_val2017.json"
COCO_IMAGES = "https://s3.amazonaws.com/images.cocodataset.org/val2017"


def read_annotations(path):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            matches = [name for name in archive.namelist() if Path(name).name == VAL_JSON]
            if len(matches) != 1:
                raise ValueError(f"Se esperaba un solo {VAL_JSON} en {path}")
            raw = archive.read(matches[0])
    else:
        raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def box_and_crop(bbox, width, height):
    x, y, w, h = bbox
    left, top = max(0, math.floor(x)), max(0, math.floor(y))
    right, bottom = min(width, math.ceil(x + w)), min(height, math.ceil(y + h))
    w, h = right - left, bottom - top
    if min(w, h) < 24 or max(w, h) > min(width, height):
        return None
    side = min(math.ceil(1.1 * max(w, h)), width, height)
    crop_x = min(max(0, (left + right - side) // 2), width - side)
    crop_y = min(max(0, (top + bottom - side) // 2), height - side)
    return w * h, (crop_x, crop_y, crop_x + side, crop_y + side)


def select_instances(coco):
    images = {row["id"]: row for row in coco["images"]}
    categories = {row["name"]: row["id"] for row in coco["categories"]}
    class_indices = {categories[name]: index for index, name in enumerate(LABELS)}
    selected = {}
    for annotation in coco["annotations"]:
        category = annotation["category_id"]
        if category not in class_indices or annotation["iscrowd"]:
            continue
        image_id = annotation["image_id"]
        image = images[image_id]
        result = box_and_crop(annotation["bbox"], image["width"], image["height"])
        if result is None:
            continue
        area, crop = result
        rank = (area, -annotation["id"])
        if image_id in selected and selected[image_id][0] >= rank:
            continue
        selected[image_id] = (rank, {
            "image_id": image_id,
            "annotation_id": annotation["id"],
            "filename": image["file_name"],
            "class_index": class_indices[category],
            "width": image["width"],
            "height": image["height"],
            "left": crop[0], "top": crop[1], "right": crop[2], "bottom": crop[3],
        })
    grouped = defaultdict(list)
    for _, row in sorted(selected.values(), key=lambda pair: pair[1]["image_id"]):
        grouped[row["class_index"]].append(row)
    return grouped


def split_instances(grouped, seed, counts):
    rng = random.Random(seed)
    rows = []
    for index, label in enumerate(LABELS):
        candidates = grouped[index].copy()
        if len(candidates) < sum(counts):
            raise ValueError(f"{label}: {len(candidates)} imágenes elegibles, se necesitan {sum(counts)}")
        rng.shuffle(candidates)
        start = 0
        for split, count in zip(SPLITS, counts):
            for row in candidates[start:start + count]:
                rows.append({"split": split, **row})
            start += count
    return rows


def write_manifest(output, rows, seed, counts, grouped, annotation_sha):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "muestras.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source": "COCO 2017 val2017 / instances",
        "annotations_sha256": annotation_sha,
        "seed": seed,
        "labels": LABELS,
        "eligible_per_class": {label: len(grouped[index]) for index, label in enumerate(LABELS)},
        "samples_per_class": dict(zip(SPLITS, counts)),
    }
    (output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "labels.txt").write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


def download_one(row, images):
    from PIL import Image

    filename = row["filename"]
    if filename != f'{row["image_id"]:012d}.jpg':
        raise ValueError(f"Nombre inesperado en COCO: {filename}")
    target = images / filename
    if target.is_file():
        try:
            with Image.open(target) as image:
                image.load()
                if image.size == (row["width"], row["height"]):
                    return
        except OSError:
            pass
    for attempt in range(3):
        try:
            with urlopen(f"{COCO_IMAGES}/{filename}", timeout=20) as response:
                data = response.read()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                if image.size != (row["width"], row["height"]):
                    raise ValueError("Las dimensiones no coinciden con las anotaciones")
        except (OSError, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f"No se pudo recuperar {filename}: {exc}") from exc
        else:
            partial = target.with_suffix(".part")
            partial.write_bytes(data)
            partial.replace(target)
            return


def main():
    parser = argparse.ArgumentParser(description="Genera las particiones de clasificación a partir de COCO val2017")
    parser.add_argument("--annotations", type=Path, required=True, help=f"{VAL_JSON} o ZIP que lo contiene")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, nargs=4, default=[45, 10, 10, 15],
                        metavar=("TRAIN", "VALIDATION", "CALIBRATION", "TEST"))
    parser.add_argument("--download-to", type=Path, help="Descarga solo las imágenes seleccionadas a este directorio")
    args = parser.parse_args()
    if min(args.per_class) < 1:
        parser.error("Cada partición necesita al menos una imagen por clase")
    coco, digest = read_annotations(args.annotations)
    grouped = select_instances(coco)
    rows = split_instances(grouped, args.seed, args.per_class)
    write_manifest(args.output, rows, args.seed, args.per_class, grouped, digest)
    if args.download_to is not None:
        args.download_to.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = pool.map(lambda row: download_one(row, args.download_to), rows)
            for index, _ in enumerate(results, start=1):
                if index % 50 == 0 or index == len(rows):
                    print(f"Imágenes disponibles: {index}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
