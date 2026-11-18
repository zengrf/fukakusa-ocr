# Training recipes

Everything here starts from two public inputs: the CODH Kuzushiji dataset (http://codh.rois.ac.jp/char-shape/; one folder per
book with `<book>_coordinate.csv`, `images/` and `characters/<U+XXXX>/`) and the release asset `labels-jibo-v2.tar.gz`
(our jibo-level label files, one per book). Paths below assume the CODH books are in `codh/` and the labels in `labels/`.

## Classifier (ViT-B/16)

```
# 1. (how the labels were made) sort crops of each kana into hentaigana variants, then write the label files
python training/classifier/refine_labels.py --sorted sorted_crops/ --codh codh/ --out labels/

# 2. one folder per class, using the jibo-level labels
python training/classifier/make_crop_folders.py --codh codh/ --labels labels/ --out crops/ --link

# 3. fine-tune; <out>/best = lowest validation loss, <out>/final = last epoch
python training/classifier/train_vit.py --data crops/ --out runs/vit-v2
```

Step 1 is shown for completeness: its output for the 44 books is the shipped label asset, so step 2 can start from it.
Step 2 writes 1,086,326 crops into 4,429 class folders. Step 3 runs with the settings of the released run as defaults
(`google/vit-base-patch16-224`, 90/10 split with seed 42, 24 epochs, learning rate 2e-5, batch 64, fp16); it took about
70 hours on one Tesla T4 with Python 3.8, torch 2.4.1, transformers 4.46.3 and datasets 3.1.0, and runs unchanged on
current versions. Evaluation record of the released run:

| epoch | accuracy | loss | epoch | accuracy | loss | epoch | accuracy | loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 91.68 % | 0.463 | 9 | 96.67 % | 0.188 | 17 | 96.80 % | 0.224 |
| 2 | 94.47 % | 0.281 | 10 | 96.58 % | 0.200 | 18 | 96.77 % | 0.230 |
| 3 | 95.62 % | 0.212 | 11 | 96.65 % | 0.204 | 19 | 96.81 % | 0.232 |
| 4 | 96.07 % | 0.188 | 12 | 96.65 % | 0.208 | 20 | 96.77 % | 0.240 |
| 5 | 96.34 % | 0.176 | 13 | 96.61 % | 0.218 | 21 | 96.85 % | 0.233 |
| **6** | **96.52 %** | **0.171** | 14 | 96.68 % | 0.220 | 22 | 96.87 % | 0.237 |
| 7 | 96.57 % | 0.178 | 15 | 96.69 % | 0.221 | 23 | 96.93 % | 0.239 |
| 8 | 96.47 % | 0.189 | 16 | 96.70 % | 0.226 | **24** | **96.94 %** | 0.240 |

Epoch 6 (lowest loss) is the released `vit-v2-epoch06`, epoch 24 the released `vit-v2-epoch24`.

## Detector: YOLO11m

```
python training/detector/make_yolo_dataset.py --codh codh/ --out datasets/kuzu --link     # 90/10 split of the pages
python training/detector/train_yolo.py --data datasets/kuzu/data.yaml                       # best.pt under runs/detect/
```

Single class, 1024 px, 100 epochs, batch 4, patience 30, seed 42, deterministic; augmentation tuned for small, dense
characters (mosaic 0.5, copy-paste 0.9, scale 0.5, no left-right flip, light photometric changes). The released
`yolo11m-kuzu-v6.pt` was trained this way on 4,864 pages and validated on 540 (precision 0.990, recall 0.990, mAP50 0.992,
mAP50-95 0.883 at epoch 100); 13.8 hours on a Tesla T4 with Ultralytics 8.3. Start from COCO-pretrained `yolo11m.pt`
(downloaded by Ultralytics) or pass `--model yolo11m-kuzu-v6.pt` to continue from the released weights.

## Detector: Cascade R-CNN R101-FPN

```
python training/detector/convert_to_coco.py --codh codh/ --out codh/            # writes codh/train.json, codh/val.json
git clone https://github.com/open-mmlab/mmdetection && pip install -e mmdetection
python mmdetection/tools/train.py training/detector/cascade_rcnn_r101.py --cfg-options data_root=codh/ --work-dir runs/cascade
```

Anchors sized for small characters (base sizes 4–64 px), random 512 px crops of pages resized to 1024 px, photometric
distortion and blur, 38 epochs at batch size 1, learning rate 5e-5 with a step schedule. The released `rcnn-res101-5-10.pth`
reached mAP50 0.892 on its validation pages (mmdetection 3.3.0, mmcv 2.1.0). For inference the release ships `rcnn-config.py`,
the configuration the checkpoint was saved with.
