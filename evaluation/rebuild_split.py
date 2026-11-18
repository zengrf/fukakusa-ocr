"""Rebuild the 90/10 split of the vit-v2 training run (train_vit.py, May 2025).

The run listed its crop folder with os.listdir (directory order, not sorted), built one row per crop file, and called
datasets.Dataset.train_test_split(test_size=0.1, seed=42), i.e. numpy.random.default_rng(42).permutation(n) with the first
ceil(0.1 n) indices as the test part. Folders whose names are not labels of the checkpoint's config (six kana-string folders
added to the crop folder in July 2025) are skipped. The directory order is a property of the filesystem that held the folder,
so the split can be rebuilt only on that filesystem or a byte-identical copy; on the original folder it gives 1,086,326 rows and
108,633 test crops. Elsewhere use test_list.tsv.gz, which names the test crops by their CODH file names.

    python rebuild_split.py <crop_folder> <checkpoint>/config.json <out_dir>
"""
import os, sys, json, math, hashlib
import numpy as np

data_dir, config, out_dir = sys.argv[1:4]
label2id = json.load(open(config))["label2id"]
rows, skipped = [], []
for code in os.listdir(data_dir):
    path = os.path.join(data_dir, code)
    if not os.path.isdir(path):
        continue
    if code not in label2id:
        skipped.append((code, len(os.listdir(path))))
        continue
    for fn in os.listdir(path):
        rows.append((code, os.path.join(path, fn)))
n = len(rows)
n_test = math.ceil(0.1 * n)
test_idx = np.random.default_rng(42).permutation(n)[:n_test]
test = [(os.path.relpath(rows[i][1], data_dir), label2id[rows[i][0]]) for i in test_idx]   # paths relative to the crop folder
digest = hashlib.sha256("\n".join(f"{p}\t{l}" for p, l in test).encode()).hexdigest()
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "test_list.tsv"), "w") as f:
    for p, l in test:
        f.write(f"{p}\t{l}\n")
json.dump({"n_rows": n, "n_test": n_test, "skipped_folders": skipped, "test_list_sha256": digest,
           "classes_in_rows": len({r[0] for r in rows})}, open(os.path.join(out_dir, "split.json"), "w"), indent=1, ensure_ascii=False)
print("rows", n, "test", n_test, "sha256", digest)
