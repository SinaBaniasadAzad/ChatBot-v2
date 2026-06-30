"""
انتخاب و ساخت مثال‌های few-shot.

استراتژی فعلی: «انتخاب ثابتِ متوازن» — برای هر ترکیب از برچسب‌های همهٔ لایه‌ها
حداکثر K مثال برمی‌داریم تا توزیع طبیعی نامتوازن (۸۰۰ Incident vs ۲۵۰۰ SR) به
مدل بایاس تزریق نکند.

این ماژول یک «درز» (seam) است: امضای build_demonstrations ثابت می‌ماند، اما بعداً
می‌توان پیاده‌سازی را به انتخاب پویا مبتنی بر embedding عوض کرد، بدون تغییر بقیهٔ کد.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from config.settings import settings
from src.taxonomy import Taxonomy
from src.utils.normalize import find_cues


def load_examples(path: Path | None = None) -> list[dict]:
    path = path or settings.examples_path
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _combo_key(example: dict, tax: Taxonomy) -> tuple:
    return tuple(example.get(layer.id) for layer in tax.layers)


def _build_output(example: dict, tax: Taxonomy) -> dict:
    """خروجی ایده‌آلِ نمایشی را از برچسب طلایی می‌سازد (evidence = cueهای حاضر در متن)."""
    text = f"{example.get('summary', '')} {example.get('description', '')}"
    layers_obj = {}
    for layer in tax.layers:
        gold_id = example.get(layer.id)
        gold = layer.get_label(gold_id)
        evidence = find_cues(text, gold.cues) if gold else []
        candidates = [{"label": gold_id, "evidence": evidence[:4]}]
        # یک runner-up با evidence خالی (اولین برچسب دیگر)
        for other in layer.labels:
            if other.id != gold_id:
                candidates.append({"label": other.id, "evidence": []})
                break
        layers_obj[layer.id] = {"candidates": candidates, "needs_clarification": False}
    return {
        "reasoning": "Domain and type are grounded in explicit evidence; no clarification needed.",
        "layers": layers_obj,
        "clarifying_question": None,
        "suggested_summary": example.get("summary", ""),
    }


def build_demonstrations(
    tax: Taxonomy,
    examples: list[dict] | None = None,
    per_combo: int = 3,
) -> list[dict]:
    """فهرستی از {input, output} متوازن بر اساس ترکیب برچسب‌ها."""
    examples = examples if examples is not None else load_examples()
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for ex in examples:
        buckets[_combo_key(ex, tax)].append(ex)

    demos: list[dict] = []
    for combo, items in buckets.items():
        for ex in items[:per_combo]:
            demos.append(
                {
                    "input": {
                        "summary": ex.get("summary", ""),
                        "description": ex.get("description", ""),
                    },
                    "output": _build_output(ex, tax),
                }
            )
    return demos
