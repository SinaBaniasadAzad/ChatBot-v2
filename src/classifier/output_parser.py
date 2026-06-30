"""
Output Parser — تبدیل خروجی خامِ LLM به یک ساختار معتبر و قابل‌اعتماد.

چرا لازم است؟ JSON mode فقط «نحوِ» JSON را تضمین می‌کند، نه اینکه خروجی با
schema و taxonomy ما بخواند. مدل ممکن است:
  - برچسبی بدهد که اصلاً وجود ندارد (hallucination)،
  - یک لایه را جا بیندازد یا کلید اضافه بیاورد،
  - نوع اشتباه بدهد (مثلاً evidence رشته به‌جای لیست).
این ماژول دو لایه دفاع دارد:
  ۱) Pre-normalization دفاعی روی dict خام (coerce انواع/ساختار).
  ۲) اعتبارسنجی معنایی در برابر taxonomy (حذف برچسب جعلی، پُرکردن لایهٔ گم‌شده).
قاعدهٔ طلایی: این تابع هیچ‌وقت crash نمی‌کند؛ در بدترین حالت لایه را «مبهم» اعلام
می‌کند تا مکالمه بتواند سوال بپرسد، نه اینکه سرویس بیفتد.
"""
from __future__ import annotations

from src.classifier.schema import Candidate, ClassificationOutput, LayerOutput
from src.taxonomy import Taxonomy
from src.utils.logging import get_logger

log = get_logger("output_parser")


def _coerce_raw(raw: dict) -> dict:
    """coerce دفاعی انواع/ساختار رایجِ منحرف، پیش از اعتبارسنجی Pydantic."""
    if not isinstance(raw, dict):
        return {"layers": {}}

    layers_in = raw.get("layers")
    if not isinstance(layers_in, dict):
        layers_in = {}

    norm_layers: dict = {}
    for lid, lo in layers_in.items():
        if not isinstance(lo, dict):
            continue
        cands = lo.get("candidates")
        if isinstance(cands, dict):       # مدل یک کاندیدا را به‌جای لیست داده
            cands = [cands]
        if not isinstance(cands, list):
            cands = []

        norm_cands = []
        for c in cands:
            if isinstance(c, str):        # فقط رشتهٔ برچسب داده
                c = {"label": c, "evidence": []}
            if not isinstance(c, dict):
                continue
            ev = c.get("evidence", [])
            if isinstance(ev, str):       # evidence رشته به‌جای لیست
                ev = [ev]
            if not isinstance(ev, list):
                ev = []
            norm_cands.append(
                {"label": str(c.get("label", "")).strip(), "evidence": [str(x) for x in ev]}
            )

        norm_layers[lid] = {
            "candidates": norm_cands,
            "needs_clarification": bool(lo.get("needs_clarification", False)),
        }

    return {
        "layers": norm_layers,
        "clarifying_question": raw.get("clarifying_question") or None,
        "suggested_summary": str(raw.get("suggested_summary") or ""),
        "reasoning": str(raw.get("reasoning") or ""),
    }


def parse_and_validate(raw: dict, tax: Taxonomy) -> ClassificationOutput:
    """
    خروجی خام مدل را به ClassificationOutputِ معتبر تبدیل می‌کند:
    - لایه‌های ناشناخته نادیده گرفته می‌شوند.
    - برچسب خارج از مجموعهٔ مجاز (hallucination) حذف می‌شود.
    - هر لایهٔ تعریف‌شده در taxonomy تضمیناً حداقل یک کاندیدای معتبر دارد؛
      اگر هیچ کاندیدای معتبری نبود، آن لایه «مبهم» علامت می‌خورد.
    """
    try:
        out = ClassificationOutput.model_validate(_coerce_raw(raw))
    except Exception as e:  # هرگز crash نکن
        log.warning("اعتبارسنجی خروجی مدل ناموفق بود؛ همهٔ لایه‌ها مبهم فرض شد: %s", e)
        out = ClassificationOutput()

    cleaned: dict[str, LayerOutput] = {}
    for layer in tax.layers:
        allowed = set(layer.label_ids)
        lo = out.layers.get(layer.id) or LayerOutput()
        valid = [c for c in lo.candidates if c.label in allowed]
        needs = lo.needs_clarification

        if not valid:  # هیچ برچسب معتبری -> ابهام به‌جای crash
            valid = [Candidate(label=lid, evidence=[]) for lid in layer.label_ids]
            needs = True

        cleaned[layer.id] = LayerOutput(candidates=valid, needs_clarification=needs)

    out.layers = cleaned
    return out


def label_key(raw: dict, tax: Taxonomy) -> tuple:
    """کلید مقایسه برای self-consistency: تاپلِ برچسب برترِ هر لایه."""
    try:
        parsed = parse_and_validate(raw, tax)
        return tuple(
            (lid, lo.top.label if lo.top else None) for lid, lo in parsed.layers.items()
        )
    except Exception:
        return ("invalid",)
