# Hold-out evaluation

The classifier's training run held out 10 % of the crops (108,633 of 1,086,326). `test_list.tsv.gz` names them, one per
line, as `<label>/<CODH crop file name>` and the label index of `vit-v2`. The names are the CODH file names, so the list
applies to any crop folder built with `training/classifier/make_crop_folders.py`, and it also marks the crops that are
held out from the released checkpoints' training for anyone evaluating on CODH material.

| file | content |
|---|---|
| `test_list.tsv.gz` | the 108,633 held-out crops (SHA-256 of the uncompressed list `4c1595e0…abf14`) |
| `eval_split.py` | top-1 accuracy of one or more checkpoints on the list, with this package's preprocessing (`--package`) or the training script's (`--training`) |
| `breakdown.py` | accuracy by class category (hentaigana, kana, CJK, other) and by class frequency, and the count of same-sound (jibo) confusions |
| `rebuild_split.py` | how the list was derived from the crop folder: `datasets.train_test_split(test_size=0.1, seed=42)` over the folder in directory order |
| `results.json` | the measurements reported in `MODEL_CARD.md` (Tesla T4, 7 November 2026) |

```
python evaluation/eval_split.py --package crops/ evaluation/test_list.tsv.gz results.json vit-v2-epoch24 vit-v2-epoch06
python evaluation/breakdown.py vit-v2-epoch24/config.json evaluation/test_list.tsv.gz preds.npy breakdown.json
```

`eval_split.py` reports, per checkpoint, the number of correct predictions in float32 and in float16 (the package's CUDA
default) and how many top-1 predictions change between the two. `breakdown.py` takes a `.npy` file with one predicted
label index per test crop, in the order of the list.
