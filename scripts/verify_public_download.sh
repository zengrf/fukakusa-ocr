#!/usr/bin/env bash
# Fresh-download check after publication: downloads the release assets and the Hugging Face repository anonymously into a new,
# empty directory and cache, verifies the checksums and runs the examples with both copies of the default checkpoint.
#   GITHUB_REPO=OWNER/NAME TAG=v1.0.0 HF_REPO=namespace/fukakusa-vit-v2 HF_REVISION=<commit sha>  bash scripts/verify_public_download.sh
# Writes verify-<UTC time>/receipt.json with URLs, hashes, commands and observed outputs.
set -euo pipefail
: "${GITHUB_REPO:?}" "${TAG:?}" "${HF_REPO:?}" "${HF_REVISION:?}"
W="$(pwd)/verify-$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$W/gh" "$W/hf" "$W/hf-cache" "$W/src"
export HF_HOME="$W/hf-cache" HF_HUB_DISABLE_TELEMETRY=1
unset HF_TOKEN GH_TOKEN GITHUB_TOKEN || true          # anonymous access only
base="https://github.com/$GITHUB_REPO/releases/download/$TAG"
for f in SHA256SUMS vit-v2-epoch24-model.safetensors vit-v2-epoch24-config.json vit-v2-epoch06-model.safetensors vit-v2-epoch06-config.json \
         yolo11m-kuzu-v6.pt class_prior.json variants.csv labels-jibo-v2.tar.gz; do
  curl -fsSL --retry 3 -o "$W/gh/$f" "$base/$f"
done
( cd "$W/gh" && sha256sum -c --ignore-missing SHA256SUMS ) | tee "$W/gh-sha256.txt"
curl -fsSL --retry 3 -o "$W/src.tar.gz" "https://github.com/$GITHUB_REPO/archive/refs/tags/$TAG.tar.gz"
tar -xzf "$W/src.tar.gz" -C "$W/src" --strip-components=1
python -m venv "$W/venv" && "$W/venv/bin/pip" -q install torch transformers safetensors pillow numpy huggingface_hub ultralytics
"$W/venv/bin/pip" -q install -e "$W/src"
"$W/venv/bin/hf" download "$HF_REPO" --revision "$HF_REVISION" --local-dir "$W/hf"
( cd "$W/hf" && sha256sum model.safetensors epoch06-deployed/model.safetensors config.json ) | tee "$W/hf-sha256.txt"
mkdir -p "$W/m24" && cp "$W/gh/vit-v2-epoch24-config.json" "$W/m24/config.json" && cp "$W/gh/vit-v2-epoch24-model.safetensors" "$W/m24/model.safetensors"
cd "$W/src"
"$W/venv/bin/fukakusa-ocr" classify --model "$W/m24" --fp32 examples/crops/*.jpg > "$W/classify-github-e24.json"
"$W/venv/bin/fukakusa-ocr" classify --model "$W/hf" --fp32 examples/crops/*.jpg > "$W/classify-hf-e24.json"
"$W/venv/bin/fukakusa-ocr" page --model "$W/m24" --yolo "$W/gh/yolo11m-kuzu-v6.pt" --fp32 examples/200021086_00026_2.jpg -o "$W/page-e24.json"
"$W/venv/bin/python" - "$W" <<'PY'
import json, sys, hashlib, pathlib, datetime, os
w = pathlib.Path(sys.argv[1])
rec = {"verified_at_utc": datetime.datetime.utcnow().isoformat() + "Z", "github_repo": os.environ["GITHUB_REPO"], "tag": os.environ["TAG"],
       "hf_repo": os.environ["HF_REPO"], "hf_revision": os.environ["HF_REVISION"],
       "github_sha256_check": (w / "gh-sha256.txt").read_text().splitlines(), "hf_sha256": (w / "hf-sha256.txt").read_text().splitlines(),
       "classify_github_equals_hf": json.load(open(w / "classify-github-e24.json")) == json.load(open(w / "classify-hf-e24.json"))}
json.dump(rec, open(w / "receipt.json", "w"), indent=1); print(json.dumps(rec, indent=1))
PY
