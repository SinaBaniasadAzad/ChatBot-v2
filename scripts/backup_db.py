"""بکاپِ منطقیِ SQLite با VACUUM INTO — یک فایلِ فشرده و سازگار، امن حینِ کارِ سرویس.

اجرا (داخل کانتینر یا venv):
    python -m scripts.backup_db                         # → backups/app-YYYYmmdd-HHMMSS.db
    python -m scripts.backup_db --out-dir /data/backups --keep 14

--keep N: بکاپ‌های قدیمی‌تر از N روز در همان پوشه حذف می‌شوند (پیش‌فرض ۱۴).
بازیابی (restore): سرویس را متوقف کنید، فایلِ بکاپ را جای APP_DB_PATH بگذارید،
فایل‌های -wal/-shm کنارش را حذف کنید، سرویس را بالا بیاورید. (جزئیات: deploy.md)
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402


def backup(db_path: Path, out_dir: Path, keep_days: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    target = out_dir / f"app-{stamp}.db"
    # اتصالِ فقط‌خواندنی: بکاپ هرگز DB را تغییر نمی‌دهد؛ VACUUM INTO فایلِ مقصد را می‌سازد
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=30.0)
    try:
        conn.execute("VACUUM INTO ?", (target.as_posix(),))
    finally:
        conn.close()

    if keep_days > 0:
        horizon = time.time() - keep_days * 86400
        for old in out_dir.glob("app-*.db"):
            if old != target and old.stat().st_mtime < horizon:
                old.unlink()
    return target


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(settings.db_path), help="مسیر DB (پیش‌فرض: APP_DB_PATH)")
    p.add_argument("--out-dir", default=None, help="پوشهٔ بکاپ‌ها (پیش‌فرض: <پوشهٔ DB>/backups)")
    p.add_argument("--keep", type=int, default=14, help="حذف بکاپ‌های قدیمی‌تر از N روز (0=هرگز)")
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB یافت نشد: {db_path}", file=sys.stderr)
        raise SystemExit(1)
    out_dir = Path(args.out_dir) if args.out_dir else db_path.parent / "backups"
    target = backup(db_path, out_dir, args.keep)
    print(target)


if __name__ == "__main__":
    main()
