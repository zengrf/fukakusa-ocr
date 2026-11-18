# Labels

| file | content |
|---|---|
| `labels-vit-v2.json` | the 4,429 classes of `vit-v2` in index order (`U+XXXX`); identical to `id2label` in the checkpoints' `config.json` |
| `labels-vit-v1.json` | the 4,438 classes of the earlier `vit-v1` |
| `variants.csv` | one row per kana and hentaigana class (348 rows): `code`, `sound_kana`, `sound_romaji`, `jibo_kanji`, `unicode_name`, `short_name`, `palette_order`, `n_ninjal` (exemplars in the NINJAL hentaigana database), `notes` |
| `class_prior.json` | number of labelled crops per class (1,086,287 in all), used by the optional logit adjustment |

The jibo-level label files of the 44 CODH books are the release asset `labels-jibo-v2.tar.gz`: `labels/<book>/<book>_coordinate_jibo_v2.csv`
with the CODH columns `Unicode, Image, X, Y, Block ID, Char ID, Width, Height`. The rows are those of the CODH `<book>_coordinate.csv`
in the same order; `Unicode` is the hentaigana code point where a kana box was assigned a jibo (285,599 of 1,086,287 boxes) and the CODH
label otherwise.
