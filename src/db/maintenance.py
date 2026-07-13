"""سیاستِ نگه‌داریِ داده (retention) و ناشناس‌سازیِ PII — در سطحِ اپلیکیشن.

سیاست (قابلِ تنظیم با env — بخشِ retention در .env.example):
  - interactions: حذفِ کامل پس از INTERACTION_RETENTION_DAYS روز (متنِ تیکت = PII).
  - tickets: پس از TICKET_ANONYMIZE_DAYS روز، ستون‌های هویتی (کد پرسنلی، نام) NULL
    می‌شوند؛ متن و برچسب‌ها برای آمار/آموزش می‌مانند (توجه: متنِ آزاد ممکن است PII
    اتفاقی داشته باشد — اگر ممنوع است TICKET_DELETE_DAYS را تنظیم کنید).
  - tickets: حذفِ کامل (به‌همراه ردیفِ FTS) پس از TICKET_DELETE_DAYS روز (0 = هرگز).

اجرا: روزانه داخلِ اپ (src/api/app.py) + دستی: python -m scripts.db_maintenance
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.db.database import Database
from src.utils.logging import get_logger

log = get_logger("db.maintenance")


def _cutoff(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def apply_retention(
    db: Database,
    *,
    interaction_days: int,
    ticket_anonymize_days: int,
    ticket_delete_days: int,
) -> dict[str, int]:
    """اعمالِ سیاستِ نگه‌داری. مقدارِ 0 برای هر بخش یعنی «غیرفعال». خروجی: شمارش‌ها."""
    conn = db.conn
    result = {"interactions_deleted": 0, "tickets_anonymized": 0, "tickets_deleted": 0}

    if interaction_days > 0:
        cur = conn.execute("DELETE FROM interactions WHERE ts < ?", (_cutoff(interaction_days),))
        result["interactions_deleted"] = cur.rowcount

    if ticket_anonymize_days > 0:
        cur = conn.execute(
            "UPDATE tickets SET employee_id = NULL, first_name = NULL, last_name = NULL,"
            " session_id = NULL, anonymized_at = ?"
            " WHERE submitted_at < ? AND anonymized_at IS NULL",
            (datetime.now(timezone.utc).isoformat(), _cutoff(ticket_anonymize_days)),
        )
        result["tickets_anonymized"] = cur.rowcount

    if ticket_delete_days > 0:
        cutoff = _cutoff(ticket_delete_days)
        conn.execute(
            "DELETE FROM tickets_fts WHERE rowid IN (SELECT id FROM tickets WHERE submitted_at < ?)",
            (cutoff,),
        )
        cur = conn.execute("DELETE FROM tickets WHERE submitted_at < ?", (cutoff,))
        result["tickets_deleted"] = cur.rowcount

    if any(result.values()):
        log.info(
            "retention اعمال شد: interactions=%d حذف، tickets=%d ناشناس، %d حذف",
            result["interactions_deleted"], result["tickets_anonymized"], result["tickets_deleted"],
        )
    return result


def apply_retention_from_settings(db: Database) -> dict[str, int]:
    from config.settings import settings

    if not settings.retention_enabled:
        return {"interactions_deleted": 0, "tickets_anonymized": 0, "tickets_deleted": 0}
    return apply_retention(
        db,
        interaction_days=settings.interaction_retention_days,
        ticket_anonymize_days=settings.ticket_anonymize_days,
        ticket_delete_days=settings.ticket_delete_days,
    )
