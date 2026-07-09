"""
واردکردنِ تیکت‌های غلط‌دسته‌بندی‌شدهٔ ارزیابی به صفِ بازبینیِ انسانی.

ورودی: خروجیِ --errors از eval_incdb (هر خط: Key, wrong, Summary, Description,
layer→{true,pred}). هر ردیف یک آیتمِ pending با source="eval" می‌شود؛ برچسبِ
طلاییِ فعلیِ دیتاست در notes می‌آید تا بازبین بتواند خودِ برچسبِ دیتاست را هم
به چالش بکشد (برچسب‌ها نویز دارند — CLAUDE.md §۴).

اجرا:
    python -m scripts.eval_incdb tests/Ticketing_DB.jsonl --frac 0.2 --seed 42 --errors errors.jsonl
    python -m scripts.import_review_items errors.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.review.store import maybe_build_review_store  # noqa: E402
from src.taxonomy import load_taxonomy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("errors_path", type=Path, help="خروجیِ --errors از eval_incdb")
    ap.add_argument("--source", default="eval", help="برچسبِ منبع در صف (پیش‌فرض: eval)")
    args = ap.parse_args()

    store = maybe_build_review_store()
    if store is None:
        raise SystemExit("صفِ بازبینی غیرفعال است (REVIEW_QUEUE_ENABLED=true لازم است).")

    layer_ids = [layer.id for layer in load_taxonomy().layers]
    total_before = sum(store.stats()["by_status"].values())
    n_rows = 0
    for line in args.errors_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        predicted, gold = {}, {}
        for lid in layer_ids:
            cell = row.get(lid) or {}
            if cell.get("pred"):
                predicted[lid] = cell["pred"]
            if cell.get("true"):
                gold[lid] = cell["true"]
        store.enqueue(
            source=args.source,
            ticket_key=str(row.get("Key") or ""),
            summary=row.get("Summary", ""),
            description=row.get("Description", ""),
            predicted_labels=predicted,
            ambiguity_reasons={"eval": [f"wrong: {', '.join(row.get('wrong') or [])}"]},
            notes=f"برچسبِ دیتاست: {json.dumps(gold, ensure_ascii=False)}",
        )
        n_rows += 1

    n_new = sum(store.stats()["by_status"].values()) - total_before
    print(f"خوانده‌شده: {n_rows} ردیف — افزوده‌شده: {n_new} آیتمِ جدید "
          f"(بقیه تکراری). بازبینی: /review در رابطِ وب.")


if __name__ == "__main__":
    main()
