"""پاک‌سازیِ دیتاست برای بنچمارک embedding — یک‌بار اجرا، خروجی در مخزن می‌ماند.

اجرا (از ریشهٔ پروژه):
    python -m scripts.prepare_retrieval_dataset
    python -m scripts.prepare_retrieval_dataset --force        # تولیدِ مجدد
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.clean import clean_dataset, load_raw, save_clean  # noqa: E402

DEFAULT_IN = Path("tests/Ticketing_DB.jsonl")
DEFAULT_OUT = Path("data/retrieval/tickets_clean.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-chars", type=int, default=1200)
    ap.add_argument("--force", action="store_true", help="بازتولید حتی اگر خروجی موجود است")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        print(f"[skip] {args.out} از قبل موجود است (برای بازتولید: --force)")
        return

    rows = load_raw(args.input)
    clean_rows, report = clean_dataset(rows, max_chars=args.max_chars)
    save_clean(clean_rows, report, args.out)

    r = dict(report)
    r.pop("duplicates", None)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    print(f"\n[ok] {report['n_kept']} ردیفِ پاک → {args.out}")
    print(f"[ok] گزارشِ کامل → {args.out.parent / 'cleaning_report.json'}")


if __name__ == "__main__":
    main()
