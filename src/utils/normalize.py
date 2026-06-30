"""
نرمال‌سازی سبکِ متن فارسی/عربی/انگلیسی.

این لایه فقط برای منطق کمکی (تطبیق کلیدواژه، dedup، لاگ تمیز) استفاده می‌شود.
متنِ خامِ اصلی همیشه عیناً به مدل و به تیکت می‌رود؛ اینجا چیزی را حذف یا ترجمه نمی‌کنیم.
"""
from __future__ import annotations

import re

# نگاشت کاراکترهای عربی به فارسی + ارقام به لاتین
_CHAR_MAP = {
    "ي": "ی", "ك": "ک", "ى": "ی", "ۀ": "ه", "ة": "ه",
    "ﻻ": "لا", "آ": "ا", "أ": "ا", "إ": "ا", "ؤ": "و", "ئ": "ی",
}
# ارقام فارسی (۰۶F0..) و عربی (۰۶60..) -> لاتین
for i, d in enumerate("۰۱۲۳۴۵۶۷۸۹"):
    _CHAR_MAP[d] = str(i)
for i, d in enumerate("٠١٢٣٤٥٦٧٨٩"):
    _CHAR_MAP[d] = str(i)

_TRANS = str.maketrans(_CHAR_MAP)

# نیم‌فاصله (ZWNJ) و کشیده (tatweel)
_ZWNJ = "‌"
_TATWEEL = "ـ"

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """نرمال‌سازی برای *تطبیق* (نه برای نمایش). نیم‌فاصله/کشیده به فاصلهٔ ساده تبدیل می‌شوند."""
    if not text:
        return ""
    text = text.translate(_TRANS)
    text = text.replace(_TATWEEL, "")
    text = text.replace(_ZWNJ, " ")
    text = text.lower()  # برای کلیدواژه‌های انگلیسی
    text = _WS.sub(" ", text)
    return text.strip()


def contains_cue(text: str, cue: str) -> bool:
    """آیا کلیدواژه (پس از نرمال‌سازی) در متن وجود دارد؟ «تایم‌شیت» و «تایم شیت» یکی می‌شوند."""
    return normalize(cue) in normalize(text)


def find_cues(text: str, cues: list[str]) -> list[str]:
    """فهرست کلیدواژه‌هایی (به شکل اصلی) که در متن یافت شدند."""
    return [cue for cue in cues if contains_cue(text, cue)]
