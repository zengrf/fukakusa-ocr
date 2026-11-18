#!/usr/bin/env python3
"""Build the classifier's training folder from the CODH Kuzushiji dataset and the jibo-level label files.

CODH distributes every book as ``<book>/<book>_coordinate.csv`` (one row per character box, modern-kana labels),
``<book>/images/`` (pages) and ``<book>/characters/<U+XXXX>/<U+XXXX>_<image>_X####_Y####.jpg`` (one crop per box).
Our label files ``<book>_coordinate_jibo_v2.csv`` have the same rows in the same order, with the ``Unicode`` column
replaced by the hentaigana code point (U+1B000–U+1B16F) wherever a kana box was assigned a jibo.

This script copies every CODH crop into ``<out>/<label>/``, taking the label from our file, so that the output is one
folder per class and is ready for ``train_vit.py``.

    python make_crop_folders.py --codh CODH_DIR --labels LABELS_DIR --out crops/ [--link]

``--labels`` is a folder with ``<book>/<book>_coordinate_jibo_v2.csv`` (the ``labels-jibo-v2`` release asset); it may be
the CODH folder itself if the files were placed next to the originals. ``--link`` hard-links instead of copying.
"""
import argparse
import csv
import os
import shutil


def crop_name(row):
    return f"{row['Unicode']}_{row['Image']}_X{row['X'].zfill(4)}_Y{row['Y'].zfill(4)}.jpg"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codh", required=True, help="folder of CODH book folders")
    ap.add_argument("--labels", required=True, help="folder with <book>/<book>_coordinate_jibo_v2.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--link", action="store_true", help="hard-link crops instead of copying")
    a = ap.parse_args()
    put = os.link if a.link else shutil.copy2

    books = sorted(b for b in os.listdir(a.codh) if os.path.isdir(os.path.join(a.codh, b, "characters")))
    n_crops = n_relabelled = n_missing = 0
    for book in books:
        orig = os.path.join(a.codh, book, f"{book}_coordinate.csv")
        ours = os.path.join(a.labels, book, f"{book}_coordinate_jibo_v2.csv")
        if not os.path.exists(ours):
            print(f"{book}: no label file, skipped")
            continue
        relabel = {}
        with open(orig, newline="", encoding="utf-8") as fo, open(ours, newline="", encoding="utf-8") as fl:
            for row, ours_row in zip(csv.DictReader(fo), csv.DictReader(fl)):
                if row["Unicode"] != ours_row["Unicode"]:
                    relabel[crop_name(row)] = ours_row["Unicode"]
        chars = os.path.join(a.codh, book, "characters")
        for code in os.listdir(chars):
            for fn in os.listdir(os.path.join(chars, code)):
                label = relabel.get(fn, code)
                dst_dir = os.path.join(a.out, label)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, fn)
                if not os.path.exists(dst):
                    put(os.path.join(chars, code, fn), dst)
                n_crops += 1
                n_relabelled += fn in relabel
        n_missing += sum(1 for fn in relabel if not os.path.exists(os.path.join(a.out, relabel[fn], fn)))
        print(f"{book}: {len(relabel):,} boxes relabelled")
    print(f"{n_crops:,} crops written to {a.out}; {n_relabelled:,} carry a jibo label; "
          f"{n_missing:,} relabelled rows had no crop file")


if __name__ == "__main__":
    main()
