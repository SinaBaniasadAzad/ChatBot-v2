"""
سنجش دقت روی یک Gold Set برچسب‌خوردهٔ تأییدشده.

ورودی: یک فایل JSONL با همان ساختار data/examples.jsonl (summary, description,
و برچسب طلایی هر لایه: layer1, layer2, ...).

اجرا:
    python -m scripts.evaluate data/gold.jsonl
    python -m scripts.evaluate data/gold.jsonl --limit 200

خروجی: دقت هر لایه + accuracy کلیِ تطبیق کامل + confusion matrix هر لایه +
نرخ موارد مبهم (single-shot؛ بدون شبیه‌سازی سوال تکمیلی).
نکته: روی برچسب‌های خام نویزی (پیشوند Key یا فیلد Application) اجرا نکنید — فقط Gold Set.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# اجازهٔ اجرا از ریشهٔ پروژه
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.classifier import Classifier  # noqa: E402
from src.classifier.decision import _is_ambiguous  # noqa: E402


def load_gold(path: Path, limit: int | None) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_path", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    clf = Classifier()
    tax = clf.taxonomy
    gold = load_gold(args.gold_path, args.limit)

    per_layer_correct: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict] = {l.id: defaultdict(lambda: defaultdict(int)) for l in tax.layers}
    ambiguous_count = 0
    full_match = 0
    total = len(gold)

    for i, ex in enumerate(gold, 1):
        out, _meta = clf.classify(ex.get("summary", ""), ex.get("description", ""))
        all_ok = True
        any_amb = False
        for layer in tax.layers:
            lo = out.layers.get(layer.id)
            pred = lo.top.label if (lo and lo.top) else None
            true = ex.get(layer.id)
            confusion[layer.id][true][pred] += 1
            if pred == true:
                per_layer_correct[layer.id] += 1
            else:
                all_ok = False
            if lo and _is_ambiguous(lo):
                any_amb = True
        full_match += int(all_ok)
        ambiguous_count += int(any_amb)
        if i % 25 == 0:
            print(f"... {i}/{total}", file=sys.stderr)

    print("\n=========== نتایج ارزیابی ===========")
    print(f"تعداد نمونه‌ها: {total}\n")
    for layer in tax.layers:
        acc = per_layer_correct[layer.id] / total if total else 0
        print(f"دقت لایهٔ «{layer.id}»: {acc:.1%}")
    print(f"\nدقت تطبیق کامل (همهٔ لایه‌ها): {full_match / total:.1%}" if total else "")
    print(f"نرخ موارد مبهم (single-shot): {ambiguous_count / total:.1%}" if total else "")

    for layer in tax.layers:
        print(f"\n--- Confusion matrix لایهٔ «{layer.id}» (سطر=واقعی، ستون=پیش‌بینی) ---")
        labels = layer.label_ids
        header = "true\\pred".ljust(18) + "".join(l.ljust(18) for l in labels)
        print(header)
        for true in labels:
            row = true.ljust(18) + "".join(
                str(confusion[layer.id][true].get(p, 0)).ljust(18) for p in labels
            )
            print(row)


if __name__ == "__main__":
    main()
