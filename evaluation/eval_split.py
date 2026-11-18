"""Evaluate classifier checkpoints on the hold-out test list.

Two preprocessing paths:
  --training   the training script's: AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True) on the image as
               decoded, evaluated under float16 autocast as the Trainer did (fp16=True) and in float32 (the released measurements used
               Python 3.8.20, torch 2.4.1, transformers 4.46.3, datasets 3.1.0 on a Tesla T4);
  --package    this package's (fukakusa_ocr.classify): ViTImageProcessor(224x224, bilinear, mean=std=0.5) on image.convert("RGB"), with the
               model cast to float16 on CUDA (the package default) and in float32.

    python eval_split.py --training|--package <crop_folder> <test_list.tsv(.gz)> <out.json> <checkpoint_dir> [<checkpoint_dir> ...]
"""
import sys, os, json, gzip, time
import numpy as np, torch
from PIL import Image

mode, data_dir, test_list, out_json, *ckpts = sys.argv[1:]
op = gzip.open if test_list.endswith(".gz") else open
test = [l.rstrip("\n").split("\t") for l in op(test_list, "rt")]
paths = [os.path.join(data_dir, p) for p, _ in test]
labels = np.array([int(l) for _, l in test])
from transformers import AutoImageProcessor, ViTImageProcessor, ViTForImageClassification
if mode == "--training":
    proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
    load = lambda p: Image.open(p)
else:
    proc = ViTImageProcessor(size={"height": 224, "width": 224}, do_resize=True, do_normalize=True,
                             image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    load = lambda p: Image.open(p).convert("RGB")


class DS(torch.utils.data.Dataset):
    def __len__(self): return len(paths)
    def __getitem__(self, i):
        im = load(paths[i]); im.load()
        return proc([im], return_tensors="pt")["pixel_values"][0]


dl = torch.utils.data.DataLoader(DS(), batch_size=256, num_workers=6, shuffle=False)
res = {"mode": mode, "n": int(len(labels))}
for ck in ckpts:
    m32 = ViTForImageClassification.from_pretrained(ck).cuda().eval()
    m16 = ViTForImageClassification.from_pretrained(ck).cuda().eval().half() if mode == "--package" else None
    a, b = [], []
    t = time.time()
    with torch.no_grad():
        for x in dl:
            x = x.cuda()
            if mode == "--training":
                with torch.autocast("cuda", dtype=torch.float16):
                    a.append(m32(pixel_values=x).logits.float().argmax(-1).cpu().numpy())
            else:
                a.append(m16(pixel_values=x.half()).logits.float().argmax(-1).cpu().numpy())
            b.append(m32(pixel_values=x).logits.argmax(-1).cpu().numpy())
    a = np.concatenate(a); b = np.concatenate(b)
    res[os.path.basename(ck.rstrip("/"))] = {"half_or_autocast_correct": int((a == labels).sum()), "half_or_autocast_accuracy": float((a == labels).mean()),
                                            "fp32_correct": int((b == labels).sum()), "fp32_accuracy": float((b == labels).mean()),
                                            "top1_changes_between_the_two": int((a != b).sum()), "seconds": round(time.time() - t, 1)}
    print(ck, res[os.path.basename(ck.rstrip("/"))], flush=True)
json.dump(res, open(out_json, "w"), indent=1)
