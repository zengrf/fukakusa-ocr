#!/usr/bin/env python3
"""Build a single-class YOLO detection dataset (character boxes) from the CODH page images and box coordinates.

Every CODH book folder has ``images/<image>.jpg`` and ``<book>_coordinate.csv`` with one row per character box
(``Image, X, Y, Width, Height`` in pixels). Pages are split 90/10 into train and val with a fixed seed. Labels are
written in YOLO format (class 0, normalised centre x, centre y, width, height) and a ``data.yaml`` is produced.

    python make_yolo_dataset.py --codh CODH_DIR --out datasets/kuzu [--val-ratio 0.1] [--seed 42] [--link]

The released detector was trained on 4,864 training pages and 540 validation pages of the 44 CODH books.
"""
import argparse
import csv
import os
import random
import shutil

from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codh", required=True, help="folder of CODH book folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="hard-link page images instead of copying")
    a = ap.parse_args()
    put = os.link if a.link else shutil.copy2

    boxes = {}      # image stem -> (book, [(x, y, w, h), ...])
    for book in sorted(os.listdir(a.codh)):
        src = os.path.join(a.codh, book, f"{book}_coordinate.csv")
        if not os.path.exists(src):
            continue
        with open(src, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                boxes.setdefault(r["Image"], (book, []))[1].append(
                    (int(r["X"]), int(r["Y"]), int(r["Width"]), int(r["Height"])))
    stems = sorted(boxes)
    random.Random(a.seed).shuffle(stems)
    n_val = round(len(stems) * a.val_ratio)
    split = {s: "val" for s in stems[:n_val]}
    split.update({s: "train" for s in stems[n_val:]})

    for part in ("train", "val"):
        os.makedirs(os.path.join(a.out, "images", part), exist_ok=True)
        os.makedirs(os.path.join(a.out, "labels", part), exist_ok=True)
    n_boxes = 0
    for stem in stems:
        book, bb = boxes[stem]
        img_src = os.path.join(a.codh, book, "images", stem + ".jpg")
        if not os.path.exists(img_src):
            continue
        part = split[stem]
        with Image.open(img_src) as im:
            W, H = im.size
        dst = os.path.join(a.out, "images", part, stem + ".jpg")
        if not os.path.exists(dst):
            put(img_src, dst)
        with open(os.path.join(a.out, "labels", part, stem + ".txt"), "w") as f:
            for x, y, w, h in bb:
                f.write(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n")
        n_boxes += len(bb)
    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n- character\n")
    print(f"{len(stems):,} pages ({len(stems) - n_val:,} train, {n_val:,} val), {n_boxes:,} boxes -> {a.out}/data.yaml")


if __name__ == "__main__":
    main()
