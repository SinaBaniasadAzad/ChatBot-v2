"""بارگذاری و جستجوی FAQ (قالب‌های آمادهٔ تیکت).

منبع داده `data/faq.json` است تا HR/IT بدون تغییر کد بتوانند سوال‌ها را ویرایش کنند.
جستجو روی متنِ نرمال‌شده انجام می‌شود (فارسی: ي→ی، ك→ک؛ ارقام عربی/فارسی → لاتین)
تا کاربر با هر صفحه‌کلیدی همان نتیجه را بگیرد. SPA همین نرمال‌سازی را سمتِ کلاینت
تکرار می‌کند؛ این ماژول برای Gradio و تست‌هاست.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import PROJECT_ROOT

FAQ_PATH = PROJECT_ROOT / "data" / "faq.json"

_CHAR_MAP = str.maketrans(
    {
        "ي": "ی",
        "ك": "ک",
        "ة": "ه",
        "أ": "ا",
        "إ": "ا",
        "ؤ": "و",
        "‌": " ",  # نیم‌فاصله
        **{chr(0x06F0 + i): str(i) for i in range(10)},  # ۰-۹
        **{chr(0x0660 + i): str(i) for i in range(10)},  # ٠-٩
    }
)


def normalize(text: str) -> str:
    return (text or "").translate(_CHAR_MAP).lower().strip()


@dataclass
class FaqItem:
    id: str
    category: str
    question: str
    summary: str
    description: str
    keywords: list[str] = field(default_factory=list)

    @property
    def search_blob(self) -> str:
        return normalize(" ".join([self.question, self.category, *self.keywords]))


def load_faq(path: Path | None = None) -> tuple[list[str], list[FaqItem]]:
    """برمی‌گرداند: (categories, items). فایلِ غایب/خراب = لیستِ خالی (UI باید کار کند)."""
    p = path or FAQ_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], []
    items = [
        FaqItem(
            id=raw.get("id", f"faq-{i}"),
            category=raw.get("category", ""),
            question=raw.get("question", ""),
            summary=raw.get("summary", ""),
            description=raw.get("description", ""),
            keywords=list(raw.get("keywords", [])),
        )
        for i, raw in enumerate(data.get("items", []))
    ]
    categories = list(data.get("categories", []))
    return categories, items


def search_faq(items: list[FaqItem], query: str, category: str | None = None) -> list[FaqItem]:
    """فیلترِ ساده و قابلِ‌پیش‌بینی: همهٔ واژه‌های query باید در متنِ نرمال‌شده باشند."""
    pool = [it for it in items if not category or it.category == category]
    terms = normalize(query).split()
    if not terms:
        return pool
    return [it for it in pool if all(t in it.search_blob for t in terms)]
