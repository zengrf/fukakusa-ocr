"""Break down hold-out predictions by label category and by training frequency.

Categories from the normalised label: hentaigana (U+1B000–U+1B16F, the jibo-bearing classes), kana (U+3040–U+30FF, sound only),
cjk-unified (U+4E00–U+9FFF), cjk-other, symbol/other. An error on a hentaigana or kana reference is a same-sound error when the
predicted class has the same sound in labels/variants.csv (a jibo confusion), otherwise a different-sound error. Frequency band
is the number of labelled crops of the reference class (labels/class_prior.json, 1,086,287 crops).

    python breakdown.py <checkpoint>/config.json <test_list.tsv(.gz)> <preds.npy> <out.json>
preds.npy holds one integer prediction per test crop (row order of the test list).
"""
import sys, os, json, csv, gzip, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from fukakusa_ocr.labels import norm_label

config, test_list, preds, out_json = sys.argv[1:5]
lab = {int(k): norm_label(v) for k, v in json.load(open(config))["id2label"].items()}
labels_dir = os.path.join(os.path.dirname(HERE), "labels")
sound = {r["code"]: r["sound_romaji"] for r in csv.DictReader(open(os.path.join(labels_dir, "variants.csv")))}
prior = json.load(open(os.path.join(labels_dir, "class_prior.json")))
op = gzip.open if test_list.endswith(".gz") else open
y = np.array([int(l.rstrip("\n").split("\t")[1]) for l in op(test_list, "rt")])
p = np.load(preds)


def cat(l):
    cp = int(l[2:], 16)
    if 0x1B000 <= cp <= 0x1B16F: return "hentaigana"
    if 0x3040 <= cp <= 0x30FF: return "kana"
    if 0x4E00 <= cp <= 0x9FFF: return "cjk-unified"
    if 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x3FFFF or 0xF900 <= cp <= 0xFAFF: return "cjk-other"
    return "symbol/other"


def band(l):
    n = prior.get(l, 0)
    return "<10" if n < 10 else "10-49" if n < 50 else "50-499" if n < 500 else ">=500"


by = collections.defaultdict(lambda: [0, 0, 0, 0]); bb = collections.defaultdict(lambda: [0, 0])
for yi, pi in zip(y, p):
    c = by[cat(lab[yi])]; c[0] += 1
    b = bb[band(lab[yi])]; b[0] += 1
    if yi == pi:
        c[1] += 1; b[1] += 1
    elif lab[yi] in sound and lab[pi] in sound and sound[lab[yi]] == sound[lab[pi]]:
        c[2] += 1
    else:
        c[3] += 1
out = {"n": int(len(y)), "correct": int((p == y).sum()), "accuracy": float((p == y).mean()),
       "by_category": {k: {"n": v[0], "correct": v[1], "accuracy": v[1] / v[0], "same_sound_errors": v[2], "different_sound_errors": v[3]} for k, v in sorted(by.items())},
       "by_training_frequency": {k: {"n": v[0], "correct": v[1], "accuracy": v[1] / v[0]} for k, v in sorted(bb.items())}}
json.dump(out, open(out_json, "w"), indent=1)
print(json.dumps(out, indent=1))
