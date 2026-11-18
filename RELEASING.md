# Release plan and status

Version 1.0.0, November 2026. Three public places carry the release:

| place | content | status |
|---|---|---|
| GitHub repository https://github.com/zengrf/fukakusa-ocr | code, training recipes, evaluation tools, label tables, documentation | **published** |
| GitHub release https://github.com/zengrf/fukakusa-ocr/releases/tag/v1.0.0 | the weights, `rcnn-config.py`, `labels-jibo-v2.tar.gz`, label maps and tables, `SHA256SUMS` (1.9 GB in 15 files) | **published** |
| Hugging Face https://huggingface.co/zengrf/fukakusa-vit-v2 | the two `vit-v2` checkpoints, label map, `variants.csv`, `class_prior.json`, model card | **to upload** (step 2 below) |

## Steps

1. GitHub: done from the package folder with `gh repo create --public --source . --push` and
   `gh release create v1.0.0 dist/*`. Every asset was hashed before upload (`checksums.txt`).
2. Hugging Face (needs a write token of the chosen namespace; run on a machine that holds the package with `hf/` staged):
   ```
   hf auth login
   HF_REPO=zengrf/fukakusa-vit-v2 bash scripts/upload_to_hf.sh
   ```
   The script checks the staged files against their SHA-256, creates the repository if needed and uploads
   `hf/fukakusa-vit-v2/` (model card built from `MODEL_CARD.md`).
3. Fresh-download check from a clean machine or directory, with the commit hash shown by the Hugging Face upload:
   ```
   GITHUB_REPO=<owner/name> TAG=v1.0.0 HF_REPO=zengrf/fukakusa-vit-v2 HF_REVISION=<commit> bash scripts/verify_public_download.sh
   ```
   It downloads everything anonymously, verifies the checksums, runs the examples from both copies of the default
   checkpoint and writes a receipt.

## Moving the release

To publish under another GitHub owner or Hugging Face namespace, transfer the repository (GitHub keeps redirects) or
create the Hugging Face repository under the new namespace, then run

```
python scripts/set_destination.py --github NEW_OWNER/NAME --hf NEW_NAMESPACE/NAME
python scripts/build_hf_card.py
```

and commit; the script rewrites every link in the documents and records the current destination in `scripts/destination.json`.

## Staging folders (not in git)

`dist/` holds the release assets exactly as uploaded; `hf/fukakusa-vit-v2/` is the Hugging Face repository layout.
`scripts/make_release_assets.sh` rebuilds `dist/` and the checksum files from the checkpoint folders, the detector weights
and the label files.
