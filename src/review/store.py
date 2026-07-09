"""
صفِ بازبینیِ انسانی (Review Queue) — مخزنِ SQLite برای حاشیه‌نویسی و اصلاحِ برچسب.

چه چیزی وارد صف می‌شود؟
  • production: جلساتی که needs_review شدند (fallback پس از اتمامِ بودجهٔ سوال).
  • eval: تیکت‌های غلط‌دسته‌بندی‌شدهٔ ارزیابی (import از errors.jsonl).
  • manual/retrieval_audit: هر موردی که تحلیل‌گر لازم بداند.

چرخهٔ عمر: pending → resolved (برچسبِ طلاییِ انسانی) یا dismissed.
خروجی: برچسب‌های تاییدشدهٔ انسانی = سرمایهٔ Gold Set و مثال‌های few-shotِ آینده
(export با گاردِ نشت به‌عهدهٔ اسکریپتِ مصرف‌کننده است).

نکتهٔ امنیتی: ردیف‌ها حاوی PII هستند؛ فایلِ پیش‌فرض داخلِ logs/ (gitignore) است.
طراحی: sqlite3ِ استاندارد (بدونِ وابستگیِ جدید)، WAL، و thread-safe برای uvicorn.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from src.utils.logging import get_logger

log = get_logger("review.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS review_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    dedupe_key TEXT UNIQUE,
    session_id TEXT,
    trace_id TEXT,
    ticket_key TEXT,
    summary TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    clarifications TEXT NOT NULL DEFAULT '[]',
    predicted_labels TEXT NOT NULL DEFAULT '{}',
    ambiguity_reasons TEXT NOT NULL DEFAULT '{}',
    evidence TEXT NOT NULL DEFAULT '{}',
    model_reasoning TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    reviewer TEXT,
    resolved_at TEXT,
    gold_labels TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_items(status);
CREATE INDEX IF NOT EXISTS idx_review_source ON review_items(source);
"""

_JSON_FIELDS = ("clarifications", "predicted_labels", "ambiguity_reasons", "evidence", "gold_labels")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReviewStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ write
    def enqueue(
        self,
        *,
        source: str,
        summary: str,
        description: str,
        session_id: str | None = None,
        trace_id: str | None = None,
        ticket_key: str | None = None,
        clarifications: list | None = None,
        predicted_labels: dict | None = None,
        ambiguity_reasons: dict | None = None,
        evidence: dict | None = None,
        model_reasoning: str = "",
        notes: str | None = None,
        dedupe_key: str | None = None,
    ) -> int | None:
        """افزودن به صف. خروجی: id (یا idِ موجود اگر تکراری بود). هرگز استثنا نمی‌دهد."""
        if dedupe_key is None and session_id:
            dedupe_key = f"{source}:{session_id}"
        elif dedupe_key is None and ticket_key:
            dedupe_key = f"{source}:{ticket_key}"
        try:
            with self._lock:
                cur = self._conn.execute(
                    """INSERT OR IGNORE INTO review_items
                       (created_at, source, dedupe_key, session_id, trace_id, ticket_key,
                        summary, description, clarifications, predicted_labels,
                        ambiguity_reasons, evidence, model_reasoning, notes)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        _now(), source, dedupe_key, session_id, trace_id, ticket_key,
                        summary or "", description or "",
                        json.dumps(clarifications or [], ensure_ascii=False),
                        json.dumps(predicted_labels or {}, ensure_ascii=False),
                        json.dumps(ambiguity_reasons or {}, ensure_ascii=False),
                        json.dumps(evidence or {}, ensure_ascii=False),
                        model_reasoning or "", notes,
                    ),
                )
                self._conn.commit()
                if cur.lastrowid and cur.rowcount:
                    return int(cur.lastrowid)
                if dedupe_key:  # آیتمِ موجود
                    row = self._conn.execute(
                        "SELECT id FROM review_items WHERE dedupe_key=?", (dedupe_key,)
                    ).fetchone()
                    return int(row["id"]) if row else None
                return None
        except sqlite3.Error as e:  # صفِ بازبینی نباید جریانِ اصلی را بشکند
            log.warning("ثبت در صفِ بازبینی ناموفق بود: %s", e)
            return None

    def resolve(
        self, item_id: int, gold_labels: dict, reviewer: str, notes: str | None = None
    ) -> dict | None:
        """ثبتِ برچسبِ طلاییِ انسانی. خروجی: آیتمِ به‌روز یا None اگر نبود/بسته بود."""
        with self._lock:
            cur = self._conn.execute(
                """UPDATE review_items
                   SET status='resolved', gold_labels=?, reviewer=?, notes=?, resolved_at=?
                   WHERE id=? AND status='pending'""",
                (json.dumps(gold_labels, ensure_ascii=False), reviewer, notes, _now(), item_id),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get(item_id)

    def dismiss(self, item_id: int, reviewer: str, notes: str | None = None) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                """UPDATE review_items
                   SET status='dismissed', reviewer=?, notes=?, resolved_at=?
                   WHERE id=? AND status='pending'""",
                (reviewer, notes, _now(), item_id),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get(item_id)

    # ------------------------------------------------------------------- read
    @staticmethod
    def _to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        for f in _JSON_FIELDS:
            if d.get(f) is not None:
                try:
                    d[f] = json.loads(d[f])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    def get(self, item_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM review_items WHERE id=?", (item_id,)
            ).fetchone()
        return self._to_dict(row) if row else None

    def list_items(
        self,
        status: str | None = "pending",
        source: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        q = "SELECT * FROM review_items"
        cond, args = [], []
        if status:
            cond.append("status=?")
            args.append(status)
        if source:
            cond.append("source=?")
            args.append(source)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        args += [int(limit), int(offset)]
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [self._to_dict(r) for r in rows]

    def stats(self) -> dict:
        """آمارِ صف + نرخِ توافقِ انسان-مدل روی آیتم‌های resolved (سیگنالِ کیفیتِ آنلاین)."""
        with self._lock:
            by_status = dict(
                self._conn.execute(
                    "SELECT status, COUNT(*) FROM review_items GROUP BY status"
                ).fetchall()
            )
            by_source = dict(
                self._conn.execute(
                    "SELECT source, COUNT(*) FROM review_items GROUP BY source"
                ).fetchall()
            )
            resolved = self._conn.execute(
                "SELECT predicted_labels, gold_labels FROM review_items WHERE status='resolved'"
            ).fetchall()
        agree = total = 0
        for row in resolved:
            try:
                pred = json.loads(row["predicted_labels"] or "{}")
                gold = json.loads(row["gold_labels"] or "{}")
            except json.JSONDecodeError:
                continue
            for lid, gold_label in gold.items():
                total += 1
                agree += int(pred.get(lid) == gold_label)
        return {
            "by_status": by_status,
            "by_source": by_source,
            "resolved_label_count": total,
            "model_human_agreement": round(agree / total, 4) if total else None,
        }

    def export_gold(self) -> list[dict]:
        """برچسب‌های تاییدشدهٔ انسانی به‌شکلِ ردیف‌های Gold Set (بدونِ PIIِ هویتی)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM review_items WHERE status='resolved' ORDER BY id"
            ).fetchall()
        out = []
        for r in rows:
            d = self._to_dict(r)
            rec = {
                "key": d.get("ticket_key") or f"REVIEW-{d['id']}",
                "summary": d["summary"],
                "description": d["description"],
                "source": "human_review",
                "session_id": d.get("session_id"),
                "reviewed_at": d.get("resolved_at"),
                **(d.get("gold_labels") or {}),
            }
            out.append(rec)
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def maybe_build_review_store() -> ReviewStore | None:
    """ساختِ امن: هر خطا فقط صفِ بازبینی را غیرفعال می‌کند، نه سرویس را."""
    if not settings.review_queue_enabled:
        return None
    try:
        return ReviewStore(settings.review_db_path)
    except Exception as e:
        log.warning("صفِ بازبینی غیرفعال شد: %s", e)
        return None
