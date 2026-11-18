"""Box post-processing and reading order: overlap merging,
column segmentation from the projection profile of box centres (right to left, top to bottom), furigana-sized column
removal."""
import itertools
import statistics

import numpy as np


def cluster_preds(preds, width_fraction=0.5):
    """Group boxes into vertical columns by x-centre overlap (returns {(cx, w): [preds]})."""
    preds = sorted(preds, key=lambda p: p["bbox"][2] - p["bbox"][0], reverse=True)
    clusters = {}
    for p in preds:
        x1, y1, x2, y2 = p["bbox"]
        w = x2 - x1; cx = (x1 + x2) / 2
        key = None
        for (CX, W) in clusters:
            if CX - W / 2 <= cx <= CX + W / 2:
                key = (CX, W); break
        if key is None:
            clusters[(cx, w * width_fraction)] = [p]
        else:
            n = len(clusters[key])
            CX = (key[0] * n + cx) / (n + 1); W = (key[1] * n + w * width_fraction) / (n + 1)
            L = clusters[key] + [p]; del clusters[key]; clusters[(CX, W)] = L
    while True:
        merged = False
        for (c1, c2) in itertools.product(list(clusters.keys()), repeat=2):
            if c1 == c2 or c1 not in clusters or c2 not in clusters:
                continue
            cx1, w1 = c1; cx2, w2 = c2
            n1, n2 = len(clusters[c1]), len(clusters[c2])
            if (cx1 - w1 / 2 <= cx2 <= cx1 + w1 / 2) or (cx2 - w2 / 2 <= cx1 <= cx2 + w2 / 2):
                CX = (cx1 * n1 + cx2 * n2) / (n1 + n2); W = (w1 * n1 + w2 * n2) / (n1 + n2)
                merged_list = clusters[c1] + clusters[c2]
                del clusters[c1]; del clusters[c2]; clusters[(CX, W)] = merged_list
                merged = True; break
        if not merged:
            break
    return clusters


def remove_furigana_clusters(clusters, size_ratio=0.6, min_page_chars=12):
    """Drop columns that are furigana-like: their characters are markedly smaller than the page's typical
    character (median box height AND width below size_ratio of the page medians). The old rule (a column narrower
    than half its right-hand neighbour) also deleted real text columns on dense pages."""
    import statistics
    allb = [p["bbox"] for v in clusters.values() for p in v]
    if len(allb) < min_page_chars or len(clusters) < 2:
        return clusters
    ph = statistics.median(b[3] - b[1] for b in allb); pw = statistics.median(b[2] - b[0] for b in allb)
    out = {}
    for k, v in clusters.items():
        h = statistics.median(p["bbox"][3] - p["bbox"][1] for p in v); w = statistics.median(p["bbox"][2] - p["bbox"][0] for p in v)
        if h < size_ratio * ph and w < size_ratio * pw and len(v) < len(allb) / 2:
            for p in v:
                p["furigana"] = True
            continue
        out[k] = v
    return out


def _median_w(v):
    import statistics
    return statistics.median(p["bbox"][2] - p["bbox"][0] for p in v)


def final_merge_clusters(clusters, width_similarity=0.6):
    """Merge clusters that are really one split column: heavy x-overlap AND similar character width.
    Narrow small-script columns (split-column commentary) next to a wide main column stay separate."""
    ret = dict(clusters)
    while True:
        merged = False
        for (c1, c2) in itertools.product(list(ret.keys()), repeat=2):
            if c1 == c2 or c1 not in ret or c2 not in ret:
                continue
            n1, n2 = len(ret[c1]), len(ret[c2])
            w1m, w2m = _median_w(ret[c1]), _median_w(ret[c2])
            if min(w1m, w2m) / max(w1m, w2m, 1e-6) < width_similarity:
                continue
            x1_min, x1_max = min(p["bbox"][0] for p in ret[c1]), max(p["bbox"][2] for p in ret[c1])
            x2_min, x2_max = min(p["bbox"][0] for p in ret[c2]), max(p["bbox"][2] for p in ret[c2])
            w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            if w >= min((x1_max - x1_min) / 2, (x2_max - x2_min) / 2):
                CX = (c1[0] * n1 + c2[0] * n2) / (n1 + n2); W = (c1[1] * n1 + c2[1] * n2) / (n1 + n2)
                merged_list = ret[c1] + ret[c2]; del ret[c1]; del ret[c2]; ret[(CX, W)] = merged_list
                merged = True; break
        if not merged:
            break
    return attach_singletons(ret)


def attach_singletons(clusters):
    """A lone box whose centre falls inside another column's x-span joins that column (closest centre wins)."""
    singles = [k for k, v in clusters.items() if len(v) == 1]
    for k in singles:
        if k not in clusters:
            continue
        p = clusters[k][0]; cx = (p["bbox"][0] + p["bbox"][2]) / 2
        best, bd = None, None
        for k2, v in clusters.items():
            if k2 == k or len(v) < 2:
                continue
            lo, hi = min(q["bbox"][0] for q in v), max(q["bbox"][2] for q in v)
            if lo <= cx <= hi:
                d = abs(k2[0] - cx)
                if bd is None or d < bd:
                    best, bd = k2, d
        if best is not None:
            clusters[best].append(p); del clusters[k]
    return clusters


def merge_overlaps(preds, threshold=0.5):
    """Drop the lower-scoring of two boxes when one mostly contains the other."""
    preds = list(preds)
    i = 0
    while i < len(preds):
        removed = False
        for j in range(len(preds)):
            if i == j:
                continue
            x1, y1, x2, y2 = preds[i]["bbox"]; X1, Y1, X2, Y2 = preds[j]["bbox"]
            ix1, iy1, ix2, iy2 = max(x1, X1), max(y1, Y1), min(x2, X2), min(y2, Y2)
            a = 0 if ix2 < ix1 or iy2 < iy1 else (ix2 - ix1) * (iy2 - iy1)
            ai, aj = max(1e-6, (x2 - x1) * (y2 - y1)), max(1e-6, (X2 - X1) * (Y2 - Y1))
            if max(a / ai, a / aj) >= threshold and preds[i]["score"] < preds[j]["score"]:
                preds.pop(i); removed = True; break
        if not removed:
            i += 1
    return preds


def columns_by_profile(preds, bw_frac=0.25, merge_frac=0.6):
    """Column segmentation from the density of box centres along x (each box adds a Gaussian weighted by its
    width, so wide main-text characters anchor their column). Peaks are columns; boxes join the nearest peak.
    Handles narrow split-column commentary beside wide main columns, which x-overlap clustering fused."""
    if not preds:
        return []
    cx = np.array([(p["bbox"][0] + p["bbox"][2]) / 2 for p in preds]); w = np.array([p["bbox"][2] - p["bbox"][0] for p in preds])
    med_w = float(np.median(w)); bw = max(3.0, bw_frac * med_w)
    grid = np.arange(cx.min() - 3 * bw, cx.max() + 3 * bw, 1.0)
    dens = np.zeros_like(grid)
    for c, ww in zip(cx, w):
        dens += (ww / med_w) * np.exp(-0.5 * ((grid - c) / bw) ** 2)
    peaks = [i for i in range(1, len(grid) - 1) if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1] and dens[i] > 0.05 * dens.max()]
    if not peaks:
        return [sorted(preds, key=lambda p: (p["bbox"][1] + p["bbox"][3]) / 2)]
    merged = []
    for x, hgt in sorted((grid[i], dens[i]) for i in peaks):
        if merged and x - merged[-1][0] < merge_frac * med_w:
            if hgt > merged[-1][1]:
                merged[-1] = (x, hgt)
        else:
            merged.append((x, hgt))
    px = [x for x, _ in merged]
    cols = {x: [] for x in px}
    for p, c in zip(preds, cx):
        cols[min(px, key=lambda x: abs(x - c))].append(p)
    return [sorted(v, key=lambda p: (p["bbox"][1] + p["bbox"][3]) / 2) for x, v in sorted(cols.items(), key=lambda kv: -kv[0]) if v]


def reading_order(preds, drop_furigana=True):
    """-> list of columns (right to left), each a list of preds top to bottom. Furigana-sized columns are dropped."""
    cols = columns_by_profile(preds)
    if drop_furigana and len(cols) > 1:
        clusters = {(i, 0): c for i, c in enumerate(cols)}
        kept = remove_furigana_clusters(clusters)
        cols = [kept[k] for k in sorted(kept.keys())]
    return cols


# ---------------------------------------------------------------- result assembly
