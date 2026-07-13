"""خروجیِ JSONL از جدولِ interactions — سازگار با ابزارهای تحلیلِ موجود.

هر خط دقیقاً همان رکوردی است که قبلاً در logs/interactions.jsonl نوشته می‌شد؛ پس:
    python -m scripts.export_interactions --out interactions.jsonl
    python -m scripts.cost_report --from-log interactions.jsonl --out cost_report.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.database import get_database  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="interactions.jsonl", help="مسیر فایل خروجی")
    p.add_argument("--since", default=None, help="فقط رکوردهای ts >= این مقدار (ISO-8601)")
    args = p.parse_args()

    db = get_database()
    sql, params = "SELECT payload FROM interactions", []
    if args.since:
        sql += " WHERE ts >= ?"
        params.append(args.since)
    sql += " ORDER BY id"

    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for (payload,) in db.conn.execute(sql, params):
            f.write(payload + "\n")
            n += 1
    print(f"{n} رکورد → {args.out}")


if __name__ == "__main__":
    main()
