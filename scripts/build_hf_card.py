"""Build hf/fukakusa-vit-v2/README.md (the Hugging Face model card) from MODEL_CARD.md and copy CITATION.cff and LICENSE.

The Hugging Face repository holds the classifier files only, so the card gets a section on its layout, inserted before
"## Model description"."""
import json, pathlib, shutil
R = pathlib.Path(__file__).resolve().parent.parent
HF = R / "hf" / "fukakusa-vit-v2"
card = (R / "MODEL_CARD.md").read_text(encoding="utf-8")
section = """## Files in this repository

| path | content |
|---|---|
| `model.safetensors`, `config.json` | `vit-v2-epoch24`, the default checkpoint (hold-out top-1 96.94 %) |
| `epoch06-deployed/model.safetensors`, `epoch06-deployed/config.json` | `vit-v2-epoch06`, the checkpoint of the project's web analyzer (96.52 %) |
| `labels-vit-v2.json` | the 4,429 classes in index order (`U+XXXX`) |
| `variants.csv` | sound, jibo and Unicode name of every kana and hentaigana class |
| `class_prior.json` | number of labelled crops per class, for the optional logit adjustment |
| `LICENSE`, `CITATION.cff` | licence and citation |

The detectors, the page reader, the training scripts and the evaluation tools are in the code repository {{CODE_REPO_URL}};
the detector weights and the label files are its release assets ({{RELEASE_URL}}).

"""
marker = "## Model description"
assert marker in card
out = card.replace(marker, section + marker, 1)
state = R / "scripts" / "destination.json"           # destinations already filled in by set_destination.py
if state.exists():
    for k, v in json.load(open(state))["values"].items():
        out = out.replace("{{%s}}" % k, v)
(HF / "README.md").write_text(out, encoding="utf-8")
for f in ("CITATION.cff", "LICENSE"):
    shutil.copy2(R / f, HF / f)
print("wrote", HF / "README.md")
