"""ثبتِ ماندگارِ تیکت‌ها روی SQLite + شمارهٔ پیگیریِ تراکنشی + جستجوی تمام‌متنی.

جایگزینِ JSONLِ قبلی: submit همان رابط و همان شکلِ رکورد را برمی‌گرداند، اما
شمارنده (TKT-YYYY-NNNNN) به‌جای اسکنِ فایل، در جدولِ ticket_counters و داخلِ یک
تراکنشِ BEGIN IMMEDIATE جلو می‌رود — بینِ threadها و پروسه‌ها امن.

جستجو: متنِ نرمال‌شدهٔ فارسی (src/utils/normalize) در tickets_fts (FTS5) ایندکس
می‌شود؛ عبارتِ کاربر با همان نرمال‌سازی و به‌صورت عبارت‌های نقل‌قولی (AND) جستجو
می‌شود تا نحوِ FTS تزریق نشود. آمادهٔ اتصال به ITSM واقعی بدونِ تغییرِ رابط.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src.db.database import Database
from src.utils.normalize import normalize


def _row_to_record(row) -> dict:
    return {
        "reference": row["reference"],
        "submitted_at": row["submitted_at"],
        "employee_id": row["employee_id"],
        "first_name": row["first_name"],
        "last_name": row["last_name"],
        "summary": row["summary"],
        "description": row["description"],
        "labels": json.loads(row["labels"] or "{}"),
        "needs_review": bool(row["needs_review"]),
        "session_id": row["session_id"],
    }


def _fts_query(user_query: str) -> str:
    """هر واژه به عبارتِ نقل‌قولی تبدیل می‌شود (AND ضمنی) — بدونِ نحوِ خامِ FTS."""
    terms = normalize(user_query).split()
    return " ".join('"' + t.replace('"', '""') + '"' for t in terms)


class TicketStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def submit(
        self,
        *,
        employee_id: str,
        first_name: str,
        last_name: str,
        summary: str,
        description: str,
        labels: dict,
        needs_review: bool = False,
        session_id: str | None = None,
    ) -> dict:
        now = datetime.now(timezone.utc)
        conn = self.db.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT last_seq FROM ticket_counters WHERE year = ?", (now.year,)
            ).fetchone()
            seq = (row[0] if row else 0) + 1
            conn.execute(
                "INSERT INTO ticket_counters(year, last_seq) VALUES (?, ?) "
                "ON CONFLICT(year) DO UPDATE SET last_seq = excluded.last_seq",
                (now.year, seq),
            )
            record = {
                "reference": f"TKT-{now.year}-{seq:05d}",
                "submitted_at": now.isoformat(),
                "employee_id": employee_id,
                "first_name": first_name,
                "last_name": last_name,
                "summary": summary,
                "description": description,
                "labels": labels,
                "needs_review": needs_review,
                "session_id": session_id,
            }
            cur = conn.execute(
                "INSERT INTO tickets(reference, submitted_at, employee_id, first_name,"
                " last_name, summary, description, labels, needs_review, session_id)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    record["reference"], record["submitted_at"], employee_id, first_name,
                    last_name, summary, description,
                    json.dumps(labels, ensure_ascii=False), int(needs_review), session_id,
                ),
            )
            conn.execute(
                "INSERT INTO tickets_fts(rowid, content_norm) VALUES (?, ?)",
                (cur.lastrowid, normalize(f"{summary} {description}")),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return record

    def get(self, reference: str) -> dict | None:
        row = self.db.conn.execute(
            "SELECT * FROM tickets WHERE reference = ?", (reference,)
        ).fetchone()
        return _row_to_record(row) if row else None

    def search(
        self,
        query: str = "",
        *,
        needs_review: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """جستجوی تمام‌متنی (یا فهرستِ آخرین تیکت‌ها اگر query خالی باشد)."""
        limit = max(1, min(int(limit), 200))
        where, params = [], []
        if query.strip():
            fts = _fts_query(query)
            if not fts:
                return []
            where.append("t.id IN (SELECT rowid FROM tickets_fts WHERE tickets_fts MATCH ?)")
            params.append(fts)
        if needs_review is not None:
            where.append("t.needs_review = ?")
            params.append(int(needs_review))
        sql = "SELECT t.* FROM tickets t"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY t.id DESC LIMIT ? OFFSET ?"
        params += [limit, max(0, int(offset))]
        return [_row_to_record(r) for r in self.db.conn.execute(sql, params)]

    def count(self) -> int:
        return int(self.db.conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0])
