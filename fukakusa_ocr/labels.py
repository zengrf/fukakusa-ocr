"""Label conventions of the Fukakusa classifiers: every class is a Unicode code point written as U+XXXX.
Hentaigana classes (U+1B000–U+1B16F) carry the jibo distinction; modern kana classes carry the sound only."""
import csv
import unicodedata
from pathlib import Path


def norm_label(raw):
    return raw.strip().split()[0] if raw and raw.strip() else raw


def label_char(code):
    try:
        return chr(int(code[2:], 16))
    except Exception:
        return code


def label_kind(code):
    try:
        cp = int(code[2:], 16)
    except Exception:
        return "other"
    if 0x1B000 <= cp <= 0x1B16F:
        return "hentaigana"
    if 0x3041 <= cp <= 0x309F:
        return "hiragana"
    if 0x30A0 <= cp <= 0x30FF:
        return "katakana"
    if 0x3400 <= cp <= 0x4DBF or 0x4E00 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF or cp >= 0x20000:
        return "kanji"
    return "other"


def unicode_name(code):
    try:
        return unicodedata.name(label_char(code))
    except Exception:
        return ""


class Variants:
    """sound, jibo and short name per hentaigana class, from labels/variants.csv (columns code,sound_kana,sound_romaji,
    jibo_kanji,unicode_name,short_name,...). Falls back to the Unicode name when a code is not listed."""

    def __init__(self, path=None):
        path = Path(path) if path else Path(__file__).resolve().parent.parent / "labels" / "variants.csv"
        self.rows = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    self.rows[r["code"]] = r

    def describe(self, code):
        code = norm_label(code)
        v = self.rows.get(code, {})
        kind = label_kind(code)
        ch = label_char(code)
        sound = v.get("sound_kana") or (ch if kind == "hiragana" else None)
        return {"code": code, "char": ch, "kind": kind, "sound": sound, "jibo": v.get("jibo_kanji") or "",
                "name": v.get("short_name") or ch or unicode_name(code)}
