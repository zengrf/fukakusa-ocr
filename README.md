# Fukakusa OCR

Models, training scripts and evaluation tools for reading classical Japanese manuscripts at the level of the *jibo* 字母,
the Chinese character from which each cursive kana derives. A detector finds the characters on a page, a ViT-B/16
classifier assigns each one a Unicode code point (4,429 classes, among them 101 hentaigana code points that carry the
jibo), and the page reader orders the columns right to left and returns the text together with a table of jibo
frequencies for orthographic analysis.

The release accompanies two papers by the Fukakusa project at the University of Washington:

- Herman Chau, Michael R. Zeng and Paul S. Atkins, "Deriving Orthographic Data from Classical Japanese Texts with
  Machine-Learning Methods", JINMONKON 2025 (IPSJ SIG Computers and the Humanities). https://hdl.handle.net/1773/55355
- Paul S. Atkins and Michael R. Zeng, "Quantitative Orthographic Analysis of the Ogura Shikishi", *Manuscript Cultures*
  26 (2026). https://doi.org/10.15460/s8e93666

Code: https://github.com/zengrf/fukakusa-ocr · Weights: https://github.com/zengrf/fukakusa-ocr/releases/tag/v1.0.0 · Classifier on Hugging Face: https://huggingface.co/zengrf/fukakusa-vit-v2

## Contents

| folder | what it holds |
|---|---|
| `fukakusa_ocr/` | inference package and the `fukakusa-ocr` command: classifier, box post-processing, reading order, page reader |
| `training/` | the training recipes: label refinement, crop folders and `train_vit.py` for the classifier; dataset builders, `train_yolo.py` and the Cascade R-CNN config for the detectors |
| `evaluation/` | the hold-out test list (108,633 crops) and the scripts that evaluate a checkpoint on it and break the result down by class category and class frequency |
| `labels/` | label maps (index → `U+XXXX`), the hentaigana table `variants.csv` (sound, jibo, name per class) and the class counts `class_prior.json` |
| `examples/` | one CODH page and six crops with the outputs of both classifier checkpoints |
| `fixtures/` | 60 crops with CPU and GPU outputs for checking a new installation |
| `vit-v2-epoch24/`, `vit-v2-epoch06/`, `vit-v1/` | `config.json` of each classifier checkpoint; the weights are release assets |
| `MODEL_CARD.md`, `CHANGELOG.md`, `CITATION.cff`, `LICENSE`, `checksums.txt` | documentation |

## Models

| asset | model | training | result |
|---|---|---|---|
| `vit-v2-epoch24-model.safetensors` (default) | ViT-B/16, 4,429 classes | 977,693 crops, 24 epochs | top-1 96.94 % on the 108,633 held-out crops |
| `vit-v2-epoch06-model.safetensors` | the same run, epoch 6 (lowest validation loss) | | top-1 96.52 %; the checkpoint of the project's web analyzer |
| `yolo11m-kuzu-v6.pt` | YOLO11m character detector, one class | 4,864 pages, 1024 px, 100 epochs | precision 0.990, recall 0.990, mAP50 0.992 on 540 validation pages |
| `rcnn-res101-5-10.pth` + `rcnn-config.py` | Cascade R-CNN R101-FPN detector (mmdetection) | 38 epochs | mAP50 0.892 on the validation pages |
| `labels-jibo-v2.tar.gz` | the jibo-level labels of all 1,086,287 character boxes of the 44 CODH books | | 285,599 boxes carry a hentaigana code point |
| `vit-v1-model.safetensors` | earlier classifier, 4,438 classes, 10 epochs | | top-1 95.92 % on a 20 % split |

SHA-256 of every asset: `checksums.txt`. Place a classifier's `model.safetensors` next to its `config.json`
(for example `vit-v2-epoch24/model.safetensors`). On Hugging Face the epoch-24 checkpoint is at the root of
`zengrf/fukakusa-vit-v2` and the epoch-6 checkpoint in the subfolder `epoch06-deployed/`.

## Install

```
python -m venv .venv && . .venv/bin/activate
pip install torch transformers safetensors pillow numpy      # classifier
pip install ultralytics                                       # page reading with the YOLO detector
pip install -e .                                              # the fukakusa_ocr package and the fukakusa-ocr command
```

## Use

Top-5 labels with probabilities for character crops:
```
fukakusa-ocr classify --model vit-v2-epoch24 crop1.jpg crop2.jpg
```
Read a page (detector → reading order → classifier → text and jibo frequencies):
```
fukakusa-ocr page --model vit-v2-epoch24 --yolo yolo11m-kuzu-v6.pt examples/200021086_00026_2.jpg -o out.json
```
Options: `--tau 0.25` applies the logit adjustment of the web analyzer (it raises hold-out accuracy by 0.06 points
for epoch 24 and 0.07 points for epoch 6, mostly on rare classes), `--imgsz 1280` runs the detector at higher
resolution, `--fp32` keeps float32 on the GPU, `-k` sets the number of candidates per character.

From Python, with the standard Transformers classes:
```python
from transformers import ViTForImageClassification, ViTImageProcessor
model = ViTForImageClassification.from_pretrained("vit-v2-epoch24").eval()          # or "zengrf/fukakusa-vit-v2"
proc = ViTImageProcessor(size={"height": 224, "width": 224}, image_mean=[0.5] * 3, image_std=[0.5] * 3)
logits = model(**proc(images=[crop.convert("RGB")], return_tensors="pt")).logits
code = model.config.id2label[int(logits.argmax())]          # e.g. "U+1B0A6": hentaigana HA written with 者
```
Every class is a Unicode code point. A hentaigana class (U+1B000–U+1B16F) names the jibo; a modern kana class
names the sound only. `labels/variants.csv` gives sound, jibo and Unicode name for every hentaigana class.

## Train

`training/README.md` walks through both recipes from the public CODH Kuzushiji dataset and the label asset:

1. `refine_labels.py` writes the jibo-level label files (the shipped `labels-jibo-v2` asset is its output).
2. `make_crop_folders.py` sorts the CODH crops into one folder per class.
3. `train_vit.py` fine-tunes `google/vit-base-patch16-224` with the settings of the released run.
4. `make_yolo_dataset.py` + `train_yolo.py` train the YOLO11m detector; `convert_to_coco.py` + `cascade_rcnn_r101.py`
   train the Cascade R-CNN detector.

## Evaluate

`evaluation/README.md` describes the hold-out protocol. `evaluation/test_list.tsv.gz` names the 108,633 crops
that were held out from the classifier's training by their CODH file names, so the evaluation can be repeated on
any copy of the crop folder; `eval_split.py` measures top-1 accuracy and `breakdown.py` splits it by class category
(hentaigana, kana, kanji) and by class frequency.

## Data

Training data are the CODH Kuzushiji dataset (Center for Open Data in the Humanities and the National Institute of
Japanese Literature; 44 books, 1,086,287 character boxes; CC BY-SA 4.0), relabelled by the project to the jibo level:
285,599 kana boxes were assigned one of 101 hentaigana code points. The label files are the `labels-jibo-v2` asset;
the images are downloaded from CODH (http://codh.rois.ac.jp/char-shape/).

## Licence and citation

Weights, labels and tables: CC BY-SA 4.0, with attribution to CODH and to the project. The YOLO detector weights
are AGPL-3.0 (Ultralytics). Details in `LICENSE`. Please cite the JINMONKON 2025 paper (`CITATION.cff`) and the
CODH Kuzushiji dataset.
