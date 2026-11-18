# Fixture tolerance: GPU (Tesla T4, CUDA) against CPU, float32, batch 16

Same 60 crops (`inputs.json`), same preprocessing as `fukakusa_ocr.classify`, same software (Python 3.8.20, torch 2.4.1,
transformers 4.46.3, cuDNN 9.1). Differences are computed over the full logit matrix (60 × number of classes) and over the
softmax probabilities derived from it; `tolerance.json` holds the values, `outputs-<model>-<device>.json` the top-5 outputs.

| model | max abs diff logits | mean abs diff logits | max abs diff probs | top-1 agreement | top-5 set agreement | top-5 order agreement |
|---|---|---|---|---|---|---|
| vit-v2-epoch24 | 1.146e-04 | 2.606e-06 | 8.941e-07 | 60/60 | 60/60 | 60/60 |
| vit-v2-epoch06 (`v2`) | 3.195e-05 | 1.907e-06 | 9.724e-07 | 60/60 | 60/60 | 60/60 |
| vit-v1 | 2.086e-04 | 3.411e-06 | 1.402e-07 | 60/60 | 60/60 | 60/60 |

A new installation reproduces these outputs when its max abs logit difference to the stored matrices is at most 1e-2 and the top-1
class agrees on all 60 crops. Float16 inference on CUDA (the package default) differs from the float32 files at the 1e-3 level in the
probabilities. `outputs-v2-*.json` also carries `top5_tau0.25`, the ranking with the logit adjustment τ = 0.25 computed from the float32 logits.
