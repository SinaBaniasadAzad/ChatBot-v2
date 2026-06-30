"""
Lightweight normalization of Persian/Arabic/English text.

This layer is used only for helper logic (cue matching, dedup, clean logs).
The original raw text always reaches the model and the ticket verbatim; we
never strip or translate anything here.
"""
from __future__ import annotations

import re

# Map Arabic characters to Persian + digits to Latin
_CHAR_MAP = {
    "ي": "ی", "ك": "ک", "ى": "ی", "ۀ": "ه", "ة": "ه",
    "ﻻ": "لا", "آ": "ا", "أ": "ا", "إ": "ا", "ؤ": "و", "ئ": "ی",
}
# Persian digits (U+06F0..) and Arabic digits (U+0660..) -> Latin
for i, d in enumerate("۰۱۲۳۴۵۶۷۸۹"):
    _CHAR_MAP[d] = str(i)
for i, d in enumerate("٠١٢٣٤٥٦٧٨٩"):
    _CHAR_MAP[d] = str(i)

_TRANS = str.maketrans(_CHAR_MAP)

# Zero-width non-joiner (ZWNJ) and tatweel (kashida)
_ZWNJ = "‌"
_TATWEEL = "ـ"

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize for *matching* (not display). ZWNJ/tatweel become a plain space."""
    if not text:
        return ""
    text = text.translate(_TRANS)
    text = text.replace(_TATWEEL, "")
    text = text.replace(_ZWNJ, " ")
    text = text.lower()  # for English keywords
    text = _WS.sub(" ", text)
    return text.strip()


def contains_cue(text: str, cue: str) -> bool:
    """Is the cue (after normalization) present in the text? "تایم‌شیت" and "تایم شیت" become equal."""
    return normalize(cue) in normalize(text)


def find_cues(text: str, cues: list[str]) -> list[str]:
    """The list of cues (in their original form) found in the text."""
    return [cue for cue in cues if contains_cue(text, cue)]
