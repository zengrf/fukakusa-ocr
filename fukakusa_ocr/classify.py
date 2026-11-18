"""The ViT-B/16 jibo classifier.

Preprocessing: the crop is converted to RGB, resized to 224×224
(bilinear, no aspect preservation) and normalised with mean 0.5 and std 0.5 per channel (ViTImageProcessor defaults
with size 224). Inference runs in float32 on CPU and, by default, in float16 on CUDA; pass
`half=False` to keep float32 on the GPU. Optional post-hoc logit adjustment (Menon et al. 2021): logits − τ·log(prior),
with the class counts of the labelled data in labels/class_prior.json; τ = 0 leaves the checkpoint's own outputs unchanged."""
import json
import math
from pathlib import Path

import torch
from PIL import Image

from .labels import norm_label


class Classifier:
    def __init__(self, model_dir, device=None, half=None, batch=64, tau=0.0, class_prior=None):
        from transformers import ViTForImageClassification, ViTImageProcessor
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(str(model_dir)).to(self.device).eval()
        self.half = (self.device.type == "cuda") if half is None else bool(half)
        if self.half:
            self.model = self.model.half()
        self.proc = ViTImageProcessor(size={"height": 224, "width": 224}, do_resize=True, do_normalize=True,
                                      image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
        self.id2label = {int(k): norm_label(v) for k, v in self.model.config.id2label.items()}
        self.batch = batch
        self.tau = float(tau)
        self.adjust = None
        if self.tau > 0:
            prior = Path(class_prior) if class_prior else Path(__file__).resolve().parent.parent / "labels" / "class_prior.json"
            counts = json.load(open(prior)); total = max(1, sum(counts.values()))
            adj = torch.tensor([math.log(max(counts.get(self.id2label[i], 1), 1) / total) for i in range(len(self.id2label))], dtype=torch.float32)
            self.adjust = (self.tau * adj).to(self.device)

    @torch.no_grad()
    def topk(self, images, k=5):
        """images: list of PIL images -> list of [(code, probability), ...] of length k."""
        out = []
        for i in range(0, len(images), self.batch):
            px = self.proc([im.convert("RGB") for im in images[i:i + self.batch]], return_tensors="pt")["pixel_values"].to(self.device)
            if self.half:
                px = px.half()
            logits = self.model(pixel_values=px).logits.float()
            if self.adjust is not None:
                logits = logits - self.adjust
            probs = torch.softmax(logits, -1)
            tp, ti = probs.topk(k, -1)
            for r in range(px.shape[0]):
                out.append([(self.id2label[int(ti[r, j])], float(tp[r, j])) for j in range(k)])
        return out

    def classify_files(self, paths, k=5):
        ims = [Image.open(p) for p in paths]
        return self.topk(ims, k=k)
