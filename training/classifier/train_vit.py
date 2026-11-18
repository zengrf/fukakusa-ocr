#!/usr/bin/env python3
"""Fine-tune ViT-B/16 on per-class crop folders (one folder per Unicode code point, one JPEG per character).

This is the training script of the released ``vit-v2`` classifier, with the settings of that run as defaults:
``google/vit-base-patch16-224`` as the starting point, a 90/10 split with seed 42, 24 epochs, learning rate 2e-5,
batch size 64, mixed precision. The Trainer evaluates and saves a checkpoint after every epoch.

    python train_vit.py --data crops/ --out runs/vit-v2 [--epochs 24] [--wandb-project NAME]

Two models are written: ``<out>/best`` is the checkpoint with the lowest evaluation loss (epoch 6 in the released run,
shipped as ``vit-v2-epoch06``) and ``<out>/final`` is the last epoch (epoch 24, shipped as ``vit-v2-epoch24``).
Per-epoch evaluation accuracy and loss are in ``<out>/checkpoints/<last>/trainer_state.json``.
"""
import argparse
import os
import re
import shutil

import numpy as np
import torch
from datasets import Dataset, Image
from transformers import AutoImageProcessor, Trainer, TrainingArguments, ViTForImageClassification


def list_crops(data_dir):
    """One (label, path) row per crop; folder names are the labels (U+XXXX). Directory order, as in the released run."""
    rows = []
    for code in os.listdir(data_dir):
        path = os.path.join(data_dir, code)
        if os.path.isdir(path):
            rows += [(code, os.path.join(path, f)) for f in os.listdir(path)]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="crop folder: <data>/<U+XXXX>/*.jpg")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--base", default="google/vit-base-patch16-224", help="starting checkpoint")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--test-size", type=float, default=0.1, help="fraction held out for evaluation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-fp16", action="store_true", help="train in float32")
    ap.add_argument("--wandb-project", default=None, help="log to Weights & Biases under this project (default: no logging)")
    a = ap.parse_args()

    rows = list_crops(a.data)
    labels = sorted({code for code, _ in rows})
    if len(labels) < 2:
        raise SystemExit("need at least two class folders")
    label2id = {c: i for i, c in enumerate(labels)}
    id2label = {i: c for c, i in label2id.items()}
    print(f"{len(rows):,} crops in {len(labels):,} classes")

    ds = Dataset.from_list([{"image": p, "label": label2id[c]} for c, p in rows]).cast_column("image", Image())
    split = ds.train_test_split(test_size=a.test_size, seed=a.seed)
    processor = AutoImageProcessor.from_pretrained(a.base, use_fast=True)

    def transform(batch):
        inputs = processor([im for im in batch["image"]], return_tensors="pt")
        inputs["label"] = batch["label"]
        return inputs

    train_ds = split["train"].with_transform(transform)
    eval_ds = split["test"].with_transform(transform)

    model = ViTForImageClassification.from_pretrained(
        a.base, num_labels=len(labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

    if a.wandb_project:
        os.environ["WANDB_PROJECT"] = a.wandb_project
    ckpt_dir = os.path.join(a.out, "checkpoints")
    args = dict(
        output_dir=ckpt_dir,
        report_to="wandb" if a.wandb_project else "none",
        remove_unused_columns=False,
        save_strategy="epoch",
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        learning_rate=a.lr,
        save_total_limit=2,            # keeps the best checkpoint and the most recent one
        warmup_steps=5,
        logging_steps=10,
        load_best_model_at_end=True,   # best = lowest evaluation loss
        fp16=not a.no_fp16,
        seed=a.seed,
    )
    try:
        targs = TrainingArguments(eval_strategy="epoch", **args)
    except TypeError:                   # transformers < 4.41 names the option evaluation_strategy
        targs = TrainingArguments(evaluation_strategy="epoch", **args)

    def compute_metrics(pred):
        logits, refs = pred
        return {"accuracy": float((np.asarray(logits).argmax(-1) == np.asarray(refs)).mean())}

    def collate(examples):
        return {"pixel_values": torch.stack([e["pixel_values"] for e in examples]),
                "labels": torch.tensor([e["label"] for e in examples])}

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      compute_metrics=compute_metrics, data_collator=collate)
    result = trainer.train()
    trainer.log_metrics("train", result.metrics)

    trainer.save_model(os.path.join(a.out, "best"))            # the model in memory is the best checkpoint
    processor.save_pretrained(os.path.join(a.out, "best"))
    last = max((d for d in os.listdir(ckpt_dir) if re.fullmatch(r"checkpoint-\d+", d)),
               key=lambda d: int(d.split("-")[1]), default=None)
    if last:
        final = os.path.join(a.out, "final")
        shutil.copytree(os.path.join(ckpt_dir, last), final, dirs_exist_ok=True)
        processor.save_pretrained(final)
        print(f"best model -> {a.out}/best ; final epoch ({last}) -> {final}")
    if a.wandb_project:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
