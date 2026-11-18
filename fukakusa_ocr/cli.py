"""Command line: `python -m fukakusa_ocr.cli classify --model <dir> crop1.jpg crop2.jpg ...`
                 `python -m fukakusa_ocr.cli page --model <dir> --yolo yolo11m-kuzu-v6.pt page.jpg [-o out.json]`"""
import argparse
import json
import sys

from .classify import Classifier
from .labels import Variants


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fukakusa_ocr")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify"); c.add_argument("--model", required=True); c.add_argument("--variants"); c.add_argument("--device"); c.add_argument("--fp32", action="store_true")
    c.add_argument("--tau", type=float, default=0.0); c.add_argument("-k", type=int, default=5); c.add_argument("images", nargs="+")
    p = sub.add_parser("page"); p.add_argument("--model", required=True); p.add_argument("--yolo", required=True); p.add_argument("--variants"); p.add_argument("--device"); p.add_argument("--fp32", action="store_true")
    p.add_argument("--tau", type=float, default=0.0); p.add_argument("--imgsz", type=int, default=1024); p.add_argument("--conf", type=float, default=0.25); p.add_argument("-k", type=int, default=5); p.add_argument("-o", "--out"); p.add_argument("image")
    a = ap.parse_args(argv)
    variants = Variants(a.variants)
    clf = Classifier(a.model, device=a.device, half=(False if a.fp32 else None), tau=a.tau)
    if a.cmd == "classify":
        res = clf.classify_files(a.images, k=a.k)
        out = [{"image": img, "topk": [dict(variants.describe(code), p=round(pr, 6)) for code, pr in top]} for img, top in zip(a.images, res)]
        json.dump(out, sys.stdout, ensure_ascii=False, indent=1); print()
    else:
        from .page import read_page
        res = read_page(a.image, clf, variants, a.yolo, imgsz=a.imgsz, conf=a.conf, topk=a.k, device=a.device)
        if a.out:
            json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(f"wrote {a.out}: {res['n_characters']} characters in {len(res['columns'])} columns")
        else:
            json.dump(res, sys.stdout, ensure_ascii=False, indent=1); print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
