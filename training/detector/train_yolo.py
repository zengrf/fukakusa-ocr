#!/usr/bin/env python3
"""Train the character detector: Ultralytics YOLO11m, single class, 1024 px.

These are the settings of the released ``yolo11m-kuzu-v6.pt`` (100 epochs, batch 4, patience 30, seed 42,
deterministic; 13.8 hours on one Tesla T4). Augmentation is tuned for small, dense characters on vertical pages:
no horizontal flip, light photometric changes, mosaic and copy-paste, scale 0.5.

    python train_yolo.py --data datasets/kuzu/data.yaml [--model yolo11m.pt] [--imgsz 1024] [--epochs 100] [--batch 4]

The best checkpoint by validation fitness is written to ``<project>/<name>/weights/best.pt``.
"""
import argparse

from ultralytics import YOLO

AUGMENTATION = dict(
    hsv_h=0.015, hsv_s=0.35, hsv_v=0.25,     # photometric
    degrees=2.0, translate=0.1, shear=0.3, perspective=0.0001,
    flipud=0.1, fliplr=0.0,                  # no left-right flip: the script is vertical, right to left
    mosaic=0.5, mixup=0.0, copy_paste=0.9, close_mosaic=20,
    auto_augment="randaugment", erasing=0.4,
)
LOSS = dict(box=7.5, cls=3.0, dfl=4.0)
OPTIMIZER = dict(lr0=0.001, lrf=0.0001, momentum=0.937, weight_decay=0.0005,
                 warmup_epochs=5, warmup_momentum=0.8, warmup_bias_lr=0.1, cos_lr=False)
VALIDATION = dict(conf=0.0001, iou=0.3, max_det=500, agnostic_nms=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data.yaml from make_yolo_dataset.py")
    ap.add_argument("--model", default="yolo11m.pt", help="starting weights (COCO-pretrained YOLO11m, or a previous best.pt)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--scale", type=float, default=0.5, help="scale augmentation")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="kuzu-yolo11m")
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, patience=a.patience, batch=a.batch, imgsz=a.imgsz,
        scale=a.scale, **AUGMENTATION, **LOSS, **OPTIMIZER, **VALIDATION,
        single_cls=True, rect=False, amp=True, seed=a.seed, deterministic=True,
        device=a.device, workers=a.workers, project=a.project, name=a.name,
        save=True, save_period=20, val=True, plots=True, exist_ok=True, verbose=True,
    )


if __name__ == "__main__":
    main()
