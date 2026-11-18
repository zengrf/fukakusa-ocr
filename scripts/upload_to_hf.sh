#!/usr/bin/env bash
# Upload the Hugging Face model repository from hf/fukakusa-vit-v2 (classifier checkpoints, labels, card).
# Run as the account that owns the namespace, after scripts/set_destination.py has filled in the destinations:
#   hf auth login                                     # or export HF_TOKEN=<write token of the namespace>
#   HF_REPO=<namespace>/fukakusa-vit-v2 bash scripts/upload_to_hf.sh
# Layout: model.safetensors + config.json at the root = vit-v2 epoch 24; epoch06-deployed/ = vit-v2 epoch 6.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="${HF_REPO:?set HF_REPO to namespace/fukakusa-vit-v2}"
python scripts/set_destination.py --check
python scripts/build_hf_card.py
( cd hf/fukakusa-vit-v2 && sha256sum -c <<SUMS )
97302e5d664fd1a975bffd1f101b7a86d78df182e354302f4ea33b9fea7daceb  model.safetensors
97b616799a1e822d23ea669f2f89c10056d952c553968f862acbf437c240ac28  epoch06-deployed/model.safetensors
df1332977e61ee9988d916e7c7c2d3311665e8c48185bc9cae5ad053fe930339  config.json
df1332977e61ee9988d916e7c7c2d3311665e8c48185bc9cae5ad053fe930339  epoch06-deployed/config.json
SUMS
hf repo create "$REPO" --repo-type model 2>/dev/null || true
hf upload "$REPO" hf/fukakusa-vit-v2 . --commit-message "fukakusa-ocr 1.0.0: ViT-B/16 jibo classifier (vit-v2 epoch 24; epoch06-deployed)"
echo "Uploaded. Record the commit hash shown on the repository page and run scripts/verify_public_download.sh with it."
