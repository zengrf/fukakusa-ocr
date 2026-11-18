"""fukakusa-ocr: the Fukakusa jibo classifier (ViT-B/16) and page reader for classical Japanese manuscripts
(Atkins, Chau and Zeng, University of Washington; JINMONKON 2025 and Manuscript Cultures 26, 2026).

Two entry points:
  Classifier(model_dir).classify_files(paths)   -> top-k labels with probabilities for character crops
  page.read_page(image, classifier, ...)        -> detector boxes, reading order, per-character top-k, jibo frequencies
The classifier is a standard Hugging Face ViTForImageClassification checkpoint; the detector is an Ultralytics YOLO11m
model (optional dependency).
"""
__version__ = "1.0.0"
from .labels import label_char, label_kind, Variants  # noqa: F401
from .classify import Classifier  # noqa: F401
