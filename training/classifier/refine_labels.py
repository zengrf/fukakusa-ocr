#!/usr/bin/env python3
"""Write the jibo-level label files from a tree of crops sorted by variant.

The jibo labels were produced by sorting the crops of each kana into its hentaigana variants and checking the result by
hand. The sorted crops live in a tree ``<kana code>/<variant code>/<crop>.jpg`` where the crop file keeps its CODH
name ``<code>_<image>_X####_Y####.jpg``. This script reads that tree and rewrites every CODH ``<book>_coordinate.csv``
as ``<book>_coordinate_jibo_v2.csv`` with the ``Unicode`` column set to the variant where one was assigned.

    python refine_labels.py --sorted SORTED_DIR --codh CODH_DIR --out LABELS_DIR

The output of this step for the 44 CODH books is the ``labels-jibo-v2`` release asset.
"""
import argparse
import csv
import os
import re

FIELDS = ["Unicode", "Image", "X", "Y", "Block ID", "Char ID", "Width", "Height"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sorted", required=True, help="tree <kana code>/<variant code>/<crop>.jpg")
    ap.add_argument("--codh", required=True, help="folder of CODH book folders")
    ap.add_argument("--out", required=True, help="output folder; writes <book>/<book>_coordinate_jibo_v2.csv")
    a = ap.parse_args()

    variant_of = {}
    for code in os.listdir(a.sorted):
        for variant in os.listdir(os.path.join(a.sorted, code)):
            for fn in os.listdir(os.path.join(a.sorted, code, variant)):
                if not fn.endswith(".jpg"):
                    continue
                parts = os.path.splitext(fn)[0].split("_")
                image = "_".join(parts[1:-2])
                x = parts[-2][1:]
                if re.match(r"000-\d+", x):      # a few source files carry a malformed X field
                    x = "0000"
                variant_of[(image, int(x), int(parts[-1][1:]))] = variant
    print(f"{len(variant_of):,} crops with a variant label")

    for book in sorted(os.listdir(a.codh)):
        src = os.path.join(a.codh, book, f"{book}_coordinate.csv")
        if not os.path.exists(src):
            continue
        n = 0
        os.makedirs(os.path.join(a.out, book), exist_ok=True)
        dst = os.path.join(a.out, book, f"{book}_coordinate_jibo_v2.csv")
        with open(src, newline="", encoding="utf-8") as f, open(dst, "w", newline="", encoding="utf-8") as g:
            w = csv.DictWriter(g, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for row in csv.DictReader(f):
                key = (row["Image"], int(row["X"]), int(row["Y"]))
                if key in variant_of:
                    row["Unicode"] = variant_of[key]
                    n += 1
                w.writerow(row)
        print(f"{book}: {n:,} boxes relabelled -> {dst}")


if __name__ == "__main__":
    main()
