"""Page reading: detector boxes -> reading order -> classifier top-k -> text and jibo frequencies.
The detector is Ultralytics YOLO11m (`yolo11m-kuzu-v6.pt`, single class "character"); the result dictionary has the keys
`characters`, `columns`, `text` and `jibo_frequencies`."""
import time

from PIL import Image

from .layout import merge_overlaps, reading_order


def detect_yolo(image, weights, imgsz=1024, conf=0.25, device=None):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    res = model.predict(image, imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
    preds = []
    for b, s in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
        preds.append({"bbox": tuple(float(v) for v in b), "score": float(s)})
    return preds


def build_result(image, preds, classifier, variants, detector, topk=5):
    t0 = time.time()
    preds = merge_overlaps(preds)
    cols = reading_order(preds)
    ordered = [p for col in cols for p in col]
    crops = [image.crop(tuple(int(round(v)) for v in p["bbox"])) for p in ordered]
    tops = classifier.topk(crops, k=topk) if crops else []
    chars, columns, freq = [], [], {}
    idx = 0
    for ci, col in enumerate(cols):
        ids = []
        for pi, p in enumerate(col):
            x1, y1, x2, y2 = p["bbox"]
            top = [dict(variants.describe(c), p=round(pr, 4)) for c, pr in tops[idx]]
            best = top[0]
            chars.append({"id": idx, "x": round(x1, 1), "y": round(y1, 1), "w": round(x2 - x1, 1), "h": round(y2 - y1, 1),
                          **{k: best[k] for k in ("code", "char", "kind", "sound", "jibo", "name")},
                          "confidence": best["p"], "det_score": round(float(p["score"]), 4), "topk": top, "column": ci, "pos": pi})
            if best["kind"] in ("hiragana", "hentaigana") and best["sound"]:
                freq[(best["sound"], best["code"])] = freq.get((best["sound"], best["code"]), 0) + 1
            ids.append(idx); idx += 1
        columns.append(ids)
    by_sound = {}
    for (s, c), n in freq.items():
        by_sound[s] = by_sound.get(s, 0) + n
    jf = [dict(variants.describe(c), sound=s, count=n, share=round(n / by_sound[s], 4)) for (s, c), n in sorted(freq.items(), key=lambda kv: (kv[0][0], -kv[1]))]
    return {"detector": detector, "image": {"width": image.width, "height": image.height}, "classify_ms": round((time.time() - t0) * 1000),
            "n_characters": len(chars), "characters": chars, "columns": columns,
            "text": ["".join(chars[i]["char"] for i in col) for col in columns], "jibo_frequencies": jf}


def read_page(image_path, classifier, variants, yolo_weights, imgsz=1024, conf=0.25, topk=5, device=None):
    image = Image.open(image_path).convert("RGB")
    preds = detect_yolo(image, yolo_weights, imgsz=imgsz, conf=conf, device=device)
    return build_result(image, preds, classifier, variants, detector="yolo11m-v6", topk=topk)
