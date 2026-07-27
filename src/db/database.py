"""لایهٔ دادهٔ SQLite — یک فایل، WAL، UTF-8 ذاتی؛ مناسبِ سرورِ تکی با بارِ کم.

چرا SQLite؟ بارِ طراحی (~۱۰ نوشتن/دقیقه در اوج) دو مرتبه‌بزرگی زیرِ توانِ WAL است؛
سرویسِ DB جداگانه (Postgres) فقط بارِ عملیاتی به تیمِ زیرساخت اضافه می‌کرد.
Backup = یک فایل (scripts/backup_db.py با VACUUM INTO).

الگوی اتصال: هر thread اتصالِ خودش را دارد (endpointهای sync در threadpool اجرا
می‌شوند). autocommit خاموش است (isolation_level=None) و تراکنش‌ها صریح‌اند
(BEGIN IMMEDIATE برای نوشتن‌های چندمرحله‌ای مثل شمارندهٔ تیکت).

Migration: فایل‌های شماره‌دارِ migrations/NNNN_*.sql به‌ترتیب اعمال و در
schema_migrations ثبت می‌شوند. دستورها باید idempotent باشند (IF NOT EXISTS)
تا شکستِ نیمه‌کاره قابلِ اجرای مجدد بماند.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from src.utils.logging import get_logger

log = get_logger("db")

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"

_default_db = None
_default_db_lock = threading.Lock()


def get_database() -> "Database":
    """نمونهٔ مشترکِ DB طبقِ APP_DB_PATH (در اولین استفاده ساخته و migrate می‌شود)."""
    global _default_db
    if _default_db is None:
        with _default_db_lock:
            if _default_db is None:
                from config.settings import settings

                db = Database(settings.db_path)
                db.migrate()
                _default_db = db
    return _default_db


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conns: set[sqlite3.Connection] = set()
        self._conns_lock = threading.Lock()

    @property
    def conn(self) -> sqlite3.Connection:
        """اتصالِ threadِ جاری (در اولین استفاده ساخته می‌شود)."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            # check_same_thread=False فقط برای اجازهٔ close از threadِ shutdown؛
            # هر اتصال عملاً فقط در threadِ سازنده‌اش *استفاده* می‌شود.
            conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
            conn.isolation_level = None  # تراکنشِ صریح
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
            with self._conns_lock:
                self._conns.add(conn)
        return conn

    # ---- migrations ----
    def migrate(self, migrations_dir: Path | None = None) -> list[str]:
        """اعمالِ migrationهای اعمال‌نشده به‌ترتیبِ شماره. خروجی: نامِ فایل‌های اجراشده."""
        mdir = migrations_dir or MIGRATIONS_DIR
        conn = self.conn
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version INTEGER PRIMARY KEY, name TEXT NOT NULL,"
            " applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')))"
        )
        applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        ran: list[str] = []
        for f in sorted(mdir.glob("[0-9]*.sql")):
            version = int(f.name.split("_", 1)[0])
            if version in applied:
                continue
            conn.executescript(f.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO schema_migrations(version, name) VALUES (?, ?)", (version, f.name)
            )
            ran.append(f.name)
            log.info("migration اعمال شد: %s", f.name)
        return ran

    def schema_version(self) -> int:
        row = self.conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        return int(row[0] or 0)

    # ---- health ----
    def check(self) -> None:
        """بررسیِ سبکِ سلامتِ DB (برای readiness). خطا = ناسالم."""
        self.conn.execute("SELECT 1 FROM schema_migrations LIMIT 1").fetchone()

    # ---- shutdown ----
    def close_all(self) -> None:
        """checkpoint و بستنِ همهٔ اتصال‌ها (خاموشیِ تمیز؛ فایلِ WAL جمع می‌شود)."""
        with self._conns_lock:
            conns, self._conns = list(self._conns), set()
        for i, conn in enumerate(conns):
            try:
                if i == 0:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.close()
            except sqlite3.Error as e:
                log.warning("بستنِ اتصالِ DB ناموفق بود: %s", e)
