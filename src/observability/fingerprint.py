"""
اثرانگشتِ پیکربندی (config fingerprint) — ستونِ فقراتِ «قابلِ‌بازتولید بودن».

هر trace و هر runِ آزمایشی با این اثرانگشت مهر می‌خورد تا بتوان دقت/هزینه/رفتار را
بینِ نسخه‌های مختلفِ پیکربندی (taxonomy، مثال‌های few-shot، prompt، آستانه‌ها، مدل)
مقایسه کرد. اگر دو run اثرانگشتِ یکسان داشته باشند، با پیکربندیِ یکسانی اجرا شده‌اند.

طراحی: system promptِ «ساخته‌شده» hash می‌شود (نه فقط فایل‌ها) چون همهٔ ورودی‌های
مؤثر — تعریفِ لایه‌ها، cueها، signal map، و مثال‌های few-shotِ انتخاب‌شده — در آن
جمع شده‌اند؛ فایل‌ها جداگانه هم hash می‌شوند تا منشأ تغییر قابلِ‌تشخیص باشد.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from config.settings import settings


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return _sha(Path(path).read_bytes())[:12]
    except OSError:
        return "missing"


def compute_fingerprint(system_prompt: str = "") -> dict:
    """اثرانگشتِ کامل: {"fingerprint": ..., "components": {...}}.

    components برای تشخیصِ «چه چیزی عوض شده» است؛ fingerprint برای join/فیلتر.
    """
    behavior = {
        "model": settings.model,
        "temperature": settings.temperature,
        "max_questions": settings.max_questions,
        "enable_self_consistency": settings.enable_self_consistency,
        "retrieval_enabled": settings.retrieval_enabled,
        "retrieval_k_demos": settings.retrieval_k_demos,
        "retrieval_purity_k": settings.retrieval_purity_k,
        "retrieval_sim_floor": settings.retrieval_sim_floor,
        "knn_disagree_purity": settings.knn_disagree_purity,
        "evidence_verification": settings.evidence_verification,
    }
    components = {
        "taxonomy": _file_sha(settings.taxonomy_path),
        "examples": _file_sha(settings.examples_path),
        "system_prompt": _sha(system_prompt.encode("utf-8"))[:12] if system_prompt else "none",
        "behavior": _sha(json.dumps(behavior, sort_keys=True).encode("utf-8"))[:12],
    }
    fingerprint = _sha(json.dumps(components, sort_keys=True).encode("utf-8"))[:12]
    return {"fingerprint": fingerprint, "components": components, "behavior": behavior}
