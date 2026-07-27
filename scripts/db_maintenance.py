"""اجرای دستیِ نگه‌داریِ DB: سیاستِ retention/ناشناس‌سازیِ PII (+ فشرده‌سازی اختیاری).

همین منطق روزانه داخلِ خودِ اپ هم اجرا می‌شود (src/api/app.py)؛ این CLI برای
اجرای فوری/زمان‌بندی‌شدهٔ بیرونی است:
    python -m scripts.db_maintenance                 # طبق env (RETENTION_* در .env)
    python -m scripts.db_maintenance --vacuum        # + فشرده‌سازی فایل
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.database import get_database  # noqa: E402
from src.db.maintenance import apply_retention_from_settings  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vacuum", action="store_true", help="پس از retention، فایل DB فشرده شود")
    args = p.parse_args()

    db = get_database()
    result = apply_retention_from_settings(db)
    if args.vacuum:
        db.conn.execute("VACUUM")
        result["vacuumed"] = True
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
