#!/usr/bin/env python3
"""Write COCO-format annotation files (train.json, val.json) from the CODH page images and box coordinates,
for the Cascade R-CNN detector (mmdetection).

    python convert_to_coco.py --codh CODH_DIR --out detector_data/ [--train-ratio 0.9] [--seed 42] [--per-class]

Boxes are a single category ``character`` unless ``--per-class`` is given, in which case every Unicode label becomes a
category. Image paths in the files are relative to ``--codh``; point ``data_root`` of the config at that folder.
"""
import argparse
import csv
import json
import os
import random

from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codh", required=True, help="folder of CODH book folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-class", action="store_true", help="one category per Unicode label instead of one class")
    a = ap.parse_args()

    rows_by_image = {}
    for book in sorted(os.listdir(a.codh)):
        src = os.path.join(a.codh, book, f"{book}_coordinate.csv")
        if not os.path.exists(src):
            continue
        with open(src, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows_by_image.setdefault((book, r["Image"]), []).append(r)

    images, annotations = [], []
    categories = {"character": 0}
    if a.per_class:
        codes = sorted({r["Unicode"] for rows in rows_by_image.values() for r in rows})
        categories = {c: i for i, c in enumerate(codes)}
    for (book, stem), rows in sorted(rows_by_image.items()):
        path = os.path.join(book, "images", stem + ".jpg")
        full = os.path.join(a.codh, path)
        if not os.path.exists(full):
            continue
        with Image.open(full) as im:
            W, H = im.size
        image_id = len(images)
        images.append({"id": image_id, "file_name": path, "width": W, "height": H})
        for r in rows:
            x, y, w, h = int(r["X"]), int(r["Y"]), int(r["Width"]), int(r["Height"])
            annotations.append({
                "id": len(annotations), "image_id": image_id,
                "category_id": categories[r["Unicode"]] if a.per_class else 0, "iscrowd": 0,
                "bbox": [x, y, w, h], "area": w * h,
                "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
            })

    random.Random(a.seed).shuffle(images)
    n_train = int(len(images) * a.train_ratio)
    parts = {"train": images[:n_train], "val": images[n_train:]}
    cats = [{"id": i, "name": n, "supercategory": "character"} for n, i in sorted(categories.items(), key=lambda kv: kv[1])]
    info = {"description": "CODH Kuzushiji dataset character boxes", "version": "1.0"}
    licenses = [{"id": 0, "name": "Creative Commons Attribution Share Alike 4.0 International"}]
    os.makedirs(a.out, exist_ok=True)
    for part, imgs in parts.items():
        ids = {im["id"] for im in imgs}
        data = {"info": info, "licenses": licenses, "images": imgs,
                "annotations": [an for an in annotations if an["image_id"] in ids], "categories": cats}
        with open(os.path.join(a.out, f"{part}.json"), "w") as f:
            json.dump(data, f)
        print(f"{part}: {len(imgs):,} images, {len(data['annotations']):,} boxes")


if __name__ == "__main__":
    main()
