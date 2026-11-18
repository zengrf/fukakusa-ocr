# Changelog

## 1.0.0 — November 2026

First public release of the Fukakusa models.

- Classifier `vit-v2`: ViT-B/16, 4,429 classes, two checkpoints of one training run. `vit-v2-epoch24` (final epoch,
  hold-out top-1 96.94 %) is the default; `vit-v2-epoch06` (lowest validation loss, 96.52 %) is the checkpoint of the
  project's web analyzer. The earlier `vit-v1` (4,438 classes, 95.92 %) is included as an asset.
- Detectors: `yolo11m-kuzu-v6.pt` (YOLO11m, one class; mAP50 0.992 on 540 validation pages) and
  `rcnn-res101-5-10.pth` with `rcnn-config.py` (Cascade R-CNN R101-FPN, mmdetection 3.3; mAP50 0.892).
- Labels: `labels-jibo-v2.tar.gz`, the jibo-level label files of the 44 CODH books (1,086,287 boxes, 285,599 with a
  hentaigana code point); label maps, `variants.csv` and `class_prior.json` in `labels/`.
- Code: the `fukakusa_ocr` package and the `fukakusa-ocr` command (`classify`, `page`), the training recipes in
  `training/` and the hold-out evaluation in `evaluation/`.
- Examples and fixtures: one CODH page and six crops with the outputs of both `vit-v2` checkpoints; 60 fixture crops with
  float32 outputs on CPU and GPU.

### Verified environments

| check | environment | result |
|---|---|---|
| `fukakusa-ocr classify` on the six example crops | Python 3.12, torch 2.15.0 (CPU), transformers 5.22.1, safetensors 0.8.0, pillow 13.0.0, numpy 2.5.4 | both `vit-v2` checkpoints; outputs in `examples/crops.expected*.json` |
| `fukakusa-ocr page` on `examples/200021086_00026_2.jpg` | the same, plus ultralytics 8.4.202 | 285 characters in 12 columns; `examples/200021086_00026_2.expected*.json` |
| `from_pretrained` of the checkpoint folders and of the Hugging Face layout (`subfolder="epoch06-deployed"`) | the same | loads as `ViTForImageClassification` without custom code |
| GPU path | Tesla T4, Python 3.8.20, torch 2.4.1+cu124, transformers 4.46.3 | 60-crop fixture for every checkpoint, `fixtures/outputs-*-cuda.json` |
| CPU against GPU on the 60 fixture crops, float32 | `fixtures/tolerance.json` | max abs logit difference 1.1e-4 (epoch 24), 3.2e-5 (epoch 6), 2.1e-4 (`vit-v1`); top-1 and top-5 identical |
| hold-out evaluation | Tesla T4, training environment | `evaluation/results.json` |
| training scripts | Python 3.8.20, torch 2.4.1, transformers 4.46.3 (`train_vit.py`, one epoch on a small synthetic crop set); data-preparation scripts on a synthetic CODH-style book | run end to end |

Not exercised: Apple MPS, Windows, ONNX or TorchScript export.
