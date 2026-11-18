#!/usr/bin/env bash
# Assemble dist/ (the GitHub release assets) from the checkpoint folders, the label files and the detector weights,
# then write dist/SHA256SUMS and checksums.txt. Paths of the sources are given as environment variables.
#   VIT_V2_E24=<dir> VIT_V2_E06=<dir> VIT_V1=<dir> YOLO=<pt> RCNN=<pth> RCNN_CFG=<py> LABELS_DIR=<dir with 44 book folders> \
#   bash scripts/make_release_assets.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p dist
cp "${VIT_V2_E24}/model.safetensors" dist/vit-v2-epoch24-model.safetensors; cp "${VIT_V2_E24}/config.json" dist/vit-v2-epoch24-config.json
cp "${VIT_V2_E06}/model.safetensors" dist/vit-v2-epoch06-model.safetensors; cp "${VIT_V2_E06}/config.json" dist/vit-v2-epoch06-config.json
cp "${VIT_V1}/model.safetensors" dist/vit-v1-model.safetensors; cp "${VIT_V1}/config.json" dist/vit-v1-config.json
cp "$YOLO" dist/yolo11m-kuzu-v6.pt; cp "$RCNN" dist/rcnn-res101-5-10.pth; cp "$RCNN_CFG" dist/rcnn-config.py
cp labels/variants.csv labels/class_prior.json labels/labels-vit-v2.json labels/labels-vit-v1.json dist/
tar -czf dist/labels-jibo-v2.tar.gz --owner=0 --group=0 --sort=name --mtime='2026-11-18 00:00Z' -C "$(dirname "$LABELS_DIR")" "$(basename "$LABELS_DIR")"
( cd dist && sha256sum $(ls | grep -v SHA256SUMS) > SHA256SUMS )
cp dist/SHA256SUMS checksums.txt
cat checksums.txt
