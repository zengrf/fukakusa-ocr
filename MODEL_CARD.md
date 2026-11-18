---
language: ja
license: cc-by-sa-4.0
library_name: transformers
pipeline_tag: image-classification
tags:
  - kuzushiji
  - hentaigana
  - jibo
  - classical-japanese
  - vision-transformer
inference: false
---

# Fukakusa jibo classifier (ViT-B/16)

**Authors.** Paul S. Atkins, Herman Chau, Michael R. Zeng (University of Washington).
**Papers.** Chau, Zeng and Atkins, "Deriving Orthographic Data from Classical Japanese Texts with Machine-Learning
Methods", JINMONKON 2025, https://hdl.handle.net/1773/55355. Atkins and Zeng, "Quantitative Orthographic Analysis of the
Ogura Shikishi", *Manuscript Cultures* 26 (2026), https://doi.org/10.15460/s8e93666.
**Code and weights.** https://github.com/zengrf/fukakusa-ocr · release assets https://github.com/zengrf/fukakusa-ocr/releases/tag/v1.0.0 · Hugging Face https://huggingface.co/zengrf/fukakusa-vit-v2.

## Model description

A ViT-B/16 image classifier fine-tuned from `google/vit-base-patch16-224` (ImageNet-21k pre-training, ImageNet-2012
fine-tuning) on 977,693 character crops from the CODH Kuzushiji dataset. The input is a 224×224 RGB crop of one
character, normalised with mean 0.5 and standard deviation 0.5 per channel. The output is a distribution over 4,429
Unicode code points:

| category | classes | share of the hold-out crops |
|---|---:|---:|
| hentaigana (U+1B000–U+1B16F), each naming a jibo 字母 | 101 | 26.6 % |
| modern kana (sound only) | 160 | 42.9 % |
| CJK unified ideographs | 4,063 | 29.8 % |
| other CJK, symbols, digits | 105 | 0.8 % |

The jibo is the Chinese character from which a cursive kana derives; the same sound は may be written with 者, 波, 盤 or 八.
A hentaigana prediction names that character. A modern-kana prediction means that the sound was recognised and no jibo
was assigned to that box in the training labels. `labels/variants.csv` gives sound, jibo and Unicode name for every kana
and hentaigana class.

## Checkpoints

| checkpoint | classes | origin | hold-out top-1 | SHA-256 of `model.safetensors` |
|---|---:|---|---:|---|
| `vit-v2-epoch24` (default; Hugging Face root) | 4,429 | final epoch of the 24-epoch run | 96.94 % | `97302e5d664fd1a975bffd1f101b7a86d78df182e354302f4ea33b9fea7daceb` |
| `vit-v2-epoch06` (Hugging Face `epoch06-deployed/`) | 4,429 | epoch 6 of the same run, the lowest validation loss; the checkpoint of the project's web analyzer, where it runs with the logit adjustment τ = 0.25 | 96.52 % | `97b616799a1e822d23ea669f2f89c10056d952c553968f862acbf437c240ac28` |
| `vit-v1` | 4,438 | earlier run, 10 epochs, 80/20 split, 110 hentaigana code points | 95.92 % | `13dc625889b8745a562c7f82fadb00b253fbb2e494e75641f598638a0d9ad747` |

Both `vit-v2` checkpoints share one `config.json` (SHA-256 `df1332977e61ee9988d916e7c7c2d3311665e8c48185bc9cae5ad053fe930339`).
The two checkpoints agree on the top-1 class for 278 of the 285 characters of the example page and for all 66 example and
fixture crops; where they differ, the difference is mostly ハ (U+30CF) against the hentaigana U+1B09E.

## Training data

The CODH Kuzushiji dataset (Center for Open Data in the Humanities / National Institute of Japanese Literature, CC BY-SA
4.0): 44 books, 1,086,287 character boxes with modern-kana and kanji labels. We relabelled the kana boxes to the jibo
level: crops of each kana were sorted into hentaigana variants and checked by hand, and 285,599 boxes received one of
101 hentaigana code points. The label files (`<book>_coordinate_jibo_v2.csv`, same rows and columns as the CODH files)
are the release asset `labels-jibo-v2.tar.gz`; `training/classifier/make_crop_folders.py` turns them and the CODH crops
into the training folder of 1,086,326 crops in 4,429 class folders.

## Training procedure

`training/classifier/train_vit.py` with its default settings:

| setting | value |
|---|---|
| starting point | `google/vit-base-patch16-224` (Apache-2.0), classification head re-initialised for 4,429 classes |
| split | 90 % train (977,693), 10 % evaluation (108,633); `datasets.train_test_split(test_size=0.1, seed=42)` |
| epochs | 24, evaluation and checkpoint after every epoch |
| optimiser | AdamW (Trainer default), learning rate 2e-5, linear decay, 5 warm-up steps |
| batch size | 64, mixed precision (fp16) |
| augmentation | none; the processor resizes to 224×224 and normalises |
| hardware and time | one NVIDIA Tesla T4, about 70 hours (14–17 May 2025) |
| software | Python 3.8, torch 2.4.1, transformers 4.46.3, datasets 3.1.0 |

Evaluation accuracy rose from 91.68 % after epoch 1 to 96.52 % at epoch 6 and 96.94 % at epoch 24; the evaluation loss
reached its minimum, 0.171, at epoch 6 and was 0.240 at epoch 24. The full per-epoch record is in `training/README.md`.

## Evaluation

Hold-out set: the 108,633 evaluation crops of the run (`evaluation/test_list.tsv.gz`), never seen in training.
Top-1 accuracy with this package's preprocessing (`ViTImageProcessor`, bilinear resize):

| checkpoint | float32 | float16 on CUDA | with logit adjustment τ = 0.25 (float32) |
|---|---:|---:|---:|
| `vit-v2-epoch24` | 96.94 % (105,305 correct) | 96.94 % (105,304) | 97.00 % (105,369) |
| `vit-v2-epoch06` | 96.52 % (104,855) | 96.52 % (104,853) | 96.59 % (104,932) |

The logit adjustment subtracts τ · log(class prior) from the logits before the argmax, with the class counts of the
labelled data (`labels/class_prior.json`); at τ = 0.5 the figures are 97.00 % and 96.54 %, at τ = 1.0 96.96 % and 96.06 %.
With the training script's own preprocessing (`AutoImageProcessor`, float16 autocast) the accuracies are 96.94 % and
96.52 %, identical to the run's logged evaluation to every digit.

By class category (float16 autocast). A same-sound error confuses two classes that share a sound, i.e. a jibo confusion.

| category | classes | test crops | `vit-v2-epoch24` | `vit-v2-epoch06` | same-sound errors (epoch 24 / epoch 6) |
|---|---:|---:|---:|---:|---:|
| hentaigana | 101 | 28,849 | 97.33 % | 97.21 % | 418 / 390 |
| kana | 160 | 46,585 | 97.97 % | 97.62 % | 204 / 274 |
| CJK unified | 4,063 | 32,361 | 95.29 % | 94.57 % | — |
| other CJK | 75 | 51 | 70.59 % | 68.63 % | — |
| symbols, digits | 30 | 787 | 90.60 % | 87.93 % | — |

By class frequency (number of labelled crops of the class):

| crops of the class | test crops | `vit-v2-epoch24` | `vit-v2-epoch06` |
|---|---:|---:|---:|
| fewer than 10 | 672 | 39.88 % | 26.49 % |
| 10–49 | 2,381 | 87.86 % | 84.84 % |
| 50–499 | 14,654 | 95.07 % | 94.57 % |
| 500 or more | 90,926 | 97.90 % | 97.66 % |

Scripts: `evaluation/eval_split.py` (accuracy), `evaluation/breakdown.py` (the two tables above); measured on a Tesla T4.

## How to use

```python
from transformers import ViTForImageClassification, ViTImageProcessor
model = ViTForImageClassification.from_pretrained("zengrf/fukakusa-vit-v2").eval()                               # epoch 24
# model = ViTForImageClassification.from_pretrained("zengrf/fukakusa-vit-v2", subfolder="epoch06-deployed")      # epoch 6
proc = ViTImageProcessor(size={"height": 224, "width": 224}, image_mean=[0.5] * 3, image_std=[0.5] * 3)
logits = model(**proc(images=[crop.convert("RGB")], return_tensors="pt")).logits
code = model.config.id2label[int(logits.argmax())]          # a Unicode code point such as "U+1B0A6"
```
The command line of the code repository adds top-k output, the logit adjustment and whole-page reading:
`fukakusa-ocr classify --model vit-v2-epoch24 crop.jpg` and
`fukakusa-ocr page --model vit-v2-epoch24 --yolo yolo11m-kuzu-v6.pt page.jpg`.

Preprocessing and decoding: the crop is the detector box without padding, converted to RGB, resized to 224×224 (bilinear,
no aspect preservation), normalised with mean 0.5 and standard deviation 0.5; softmax over all classes; float16 on CUDA and
float32 on CPU by default. On 60 fixture crops the CPU and GPU float32 logits differ by at most 1.1e-4 (epoch 24) and
3.2e-5 (epoch 6) with identical top-5 rankings (`fixtures/`).

## Detectors

The page reader uses one of two detectors trained on the page images and box coordinates of the same 44 books
(`training/detector/`):

| detector | recipe | validation result |
|---|---|---|
| `yolo11m-kuzu-v6.pt` (Ultralytics YOLO11m, one class) | 4,864 training pages, 1024 px, 100 epochs, batch 4, mosaic and copy-paste augmentation, no horizontal flip; 13.8 h on a Tesla T4 | 540 pages: precision 0.990, recall 0.990, mAP50 0.992, mAP50-95 0.883 |
| `rcnn-res101-5-10.pth` (Cascade R-CNN R101-FPN, mmdetection 3.3) | random 512 px crops of pages resized to 1024 px, 38 epochs, batch 1 | mAP50 0.892, mAP50-95 0.325 |

The reader merges overlapping boxes (one box contains at least half of the other), segments columns from the projection
profile of the box centres, reads them right to left and top to bottom, and drops furigana-sized columns.

## Scope

- The class inventory is fixed: 4,429 code points of the CODH labels plus the 101 hentaigana. Kanji variants are predicted at
  the code point the source annotation used.
- Accuracy is highest for frequent classes and lowest for classes with fewer than ten training crops (see the frequency table).
- The reader assumes vertical columns read right to left, the layout of the training manuscripts.
- Modern-kana predictions are sounds; use the hentaigana classes for jibo statistics.

## Licence and citation

Weights, label maps and tables: CC BY-SA 4.0, derived from the CODH Kuzushiji dataset; credit "Center for Open Data in the
Humanities (CODH), Kuzushiji dataset" and "Atkins, Chau and Zeng, Fukakusa (University of Washington)". YOLO detector
weights: AGPL-3.0. See `LICENSE` and `CITATION.cff`.
