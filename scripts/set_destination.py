"""Point the documents at the public destinations of this release, or check that none is missing.

Documents name four destinations: the code repository {{CODE_REPO_URL}}, its release page {{RELEASE_URL}}, the Hugging
Face model repository {{HF_REPO_ID}} and its page {{HF_URL}}. The current values are kept in scripts/destination.json
(placeholders until first filled), so the script can also move the release to a new owner or namespace later.

    python scripts/set_destination.py --github OWNER/NAME --hf NAMESPACE/NAME [--tag v1.0.0]
    python scripts/set_destination.py --check        # exit 1 while any placeholder remains
"""
import argparse, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATE = ROOT / "scripts" / "destination.json"
FILES = ["README.md", "MODEL_CARD.md", "CHANGELOG.md", "CITATION.cff", "RELEASING.md", "training/README.md",
         "evaluation/README.md", "labels/README.md", "hf/fukakusa-vit-v2/README.md", "hf/fukakusa-vit-v2/CITATION.cff"]
KEYS = ["CODE_REPO_URL", "RELEASE_URL", "HF_REPO_ID", "HF_URL"]
PLACEHOLDERS = {k: "{{%s}}" % k for k in KEYS}


def values(github, hf, tag):
    return {"CODE_REPO_URL": f"https://github.com/{github}", "RELEASE_URL": f"https://github.com/{github}/releases/tag/{tag}",
            "HF_REPO_ID": hf, "HF_URL": f"https://huggingface.co/{hf}"}


def current():
    return json.load(open(STATE))["values"] if STATE.exists() else dict(PLACEHOLDERS)


def remaining():
    pat = re.compile(r"\{\{(%s)\}\}" % "|".join(KEYS))
    return [f"{f}:{n}: {line.strip()[:100]}" for f in FILES if (ROOT / f).exists()
            for n, line in enumerate((ROOT / f).read_text(encoding="utf-8").splitlines(), 1) if pat.search(line)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--github"); ap.add_argument("--hf"); ap.add_argument("--tag", default="v1.0.0")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        left = remaining(); print("\n".join(left) if left else "no placeholders left"); sys.exit(1 if left else 0)
    if not (a.github and a.hf and re.fullmatch(r"[\w.-]+/[\w.-]+", a.github) and re.fullmatch(r"[\w.-]+/[\w.-]+", a.hf)):
        sys.exit("give --github OWNER/NAME and --hf NAMESPACE/NAME")
    old, new = current(), values(a.github, a.hf, a.tag)
    # replace longer strings first so that HF_URL (which contains HF_REPO_ID) is substituted before HF_REPO_ID
    order = sorted(KEYS, key=lambda k: -len(old[k]))
    for f in FILES:
        p = ROOT / f
        if not p.exists():
            continue
        s = t = p.read_text(encoding="utf-8")
        for k in order:
            if old[k] != new[k]:
                t = t.replace(old[k], new[k])
        if t != s:
            p.write_text(t, encoding="utf-8"); print("updated", f)
    json.dump({"github": a.github, "hf": a.hf, "tag": a.tag, "values": new}, open(STATE, "w"), indent=1)
    left = remaining()
    sys.exit(("placeholders left:\n" + "\n".join(left)) if left else 0)


if __name__ == "__main__":
    main()
