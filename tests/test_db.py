"""تست‌های آفلاینِ لایهٔ داده: migration، TicketStore روی SQLite، FTSِ فارسی، retention."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.db.database import Database
from src.db.maintenance import apply_retention
from src.tickets.store import TicketStore
from src.utils.interaction_log import InteractionLogger


@pytest.fixture()
def db(tmp_path) -> Database:
    d = Database(tmp_path / "app.db")
    d.migrate()
    return d


def _submit(store: TicketStore, **overrides) -> dict:
    base = dict(
        employee_id="263669",
        first_name="Sara",
        last_name="Ahmadi",
        summary="Punch not recorded",
        description="My exit punch on 2026-07-01 was not recorded.",
        labels={"layer1": "incident", "layer2": "erp"},
        needs_review=False,
        session_id="abc",
    )
    base.update(overrides)
    return store.submit(**base)


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------
def test_migrate_is_idempotent(tmp_path):
    d = Database(tmp_path / "app.db")
    first = d.migrate()
    assert first, "initial migration must run"
    assert d.migrate() == []  # اجرای دوباره → هیچ
    assert d.schema_version() >= 1


# ---------------------------------------------------------------------------
# TicketStore
# ---------------------------------------------------------------------------
def test_sequential_references_and_shape(db):
    store = TicketStore(db)
    year = datetime.now(timezone.utc).year
    r1, r2 = _submit(store), _submit(store)
    assert r1["reference"] == f"TKT-{year}-00001"
    assert r2["reference"] == f"TKT-{year}-00002"
    assert r1["labels"] == {"layer1": "incident", "layer2": "erp"}
    assert r1["needs_review"] is False


def test_counter_resumes_after_restart(tmp_path):
    path = tmp_path / "app.db"
    d1 = Database(path)
    d1.migrate()
    _submit(TicketStore(d1))
    d1.close_all()
    d2 = Database(path)  # «پروسهٔ» جدید
    d2.migrate()
    r = _submit(TicketStore(d2))
    assert r["reference"].endswith("00002")


def test_persian_text_utf8_roundtrip(db):
    store = TicketStore(db)
    summary = "عدم ثبت پانچ ورود و خروج"
    description = "پانچ ورود من در تاریخ ۱۹ مرداد در سامانهٔ ERP ثبت نشده است."
    r = _submit(store, summary=summary, description=description, first_name="سارا", last_name="احمدی")
    fetched = store.get(r["reference"])
    assert fetched["summary"] == summary
    assert fetched["description"] == description
    assert fetched["first_name"] == "سارا"


def test_fts_search_persian_normalization(db):
    store = TicketStore(db)
    # نیم‌فاصله در متن؛ جستجو با فاصلهٔ ساده باید پیدا کند
    r1 = _submit(store, summary="مشکل تایم‌شیت", description="تایم‌شیت من تایید نمی‌شود")
    # یِ عربی در جستجو باید با یِ فارسیِ متن یکی شود
    _submit(store, summary="درخواست وام", description="فعال‌سازی ماژول وام")

    hits = store.search("تایم شیت")
    assert [h["reference"] for h in hits] == [r1["reference"]]

    assert store.search("تاييد")  # ي عربی → متنِ «تایید» را پیدا کند
    assert store.search("no-such-word-xyz") == []


def test_search_filters_and_order(db):
    store = TicketStore(db)
    _submit(store, summary="a")
    r2 = _submit(store, summary="b", needs_review=True)
    flagged = store.search(needs_review=True)
    assert [t["reference"] for t in flagged] == [r2["reference"]]
    assert store.search()[0]["reference"] == r2["reference"]  # جدیدترین اول
    assert store.get("TKT-1999-00001") is None


def test_fts_query_injection_is_safe(db):
    store = TicketStore(db)
    _submit(store, summary="hello world")
    # نحوِ FTS (پرانتز/ستاره/NOT) نباید خطای syntax بدهد
    assert store.search('hello AND (world OR "x') is not None


# ---------------------------------------------------------------------------
# InteractionLogger + retention
# ---------------------------------------------------------------------------
class _FakeSession:
    session_id = "s1"
    questions_asked = 0
    summary = "s"
    description = "d"
    clarifications: list = []
    status = None
    result = None


def test_interaction_logger_writes_payload(db):
    logger = InteractionLogger(True, db)
    logger.log_final(_FakeSession())
    rows = list(db.conn.execute("SELECT event, session_id, payload FROM interactions"))
    assert len(rows) == 1
    assert rows[0]["event"] == "session_final"
    assert rows[0]["session_id"] == "s1"
    assert '"summary": "s"' in rows[0]["payload"]


def test_disabled_logger_writes_nothing(db):
    InteractionLogger(False, db).log_final(_FakeSession())
    assert db.conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0] == 0


def _age_rows(db, days: int) -> None:
    """رکوردها را در گذشته جا می‌زند تا retention رویشان اعمال شود."""
    old = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    db.conn.execute("UPDATE interactions SET ts = ?", (old,))
    db.conn.execute("UPDATE tickets SET submitted_at = ?", (old,))


def test_retention_purges_and_anonymizes(db):
    store = TicketStore(db)
    r = _submit(store, summary="پانچ ثبت نشد")
    InteractionLogger(True, db).log_final(_FakeSession())
    _age_rows(db, days=400)

    result = apply_retention(db, interaction_days=90, ticket_anonymize_days=365, ticket_delete_days=0)
    assert result == {"interactions_deleted": 1, "tickets_anonymized": 1, "tickets_deleted": 0}

    t = store.get(r["reference"])
    assert t["employee_id"] is None and t["first_name"] is None and t["last_name"] is None
    assert t["summary"] == "پانچ ثبت نشد"  # متن می‌ماند (سیاست پیش‌فرض)
    # دوباره: چیزی برای انجام نیست
    again = apply_retention(db, interaction_days=90, ticket_anonymize_days=365, ticket_delete_days=0)
    assert again["tickets_anonymized"] == 0


def test_retention_full_delete_removes_fts(db):
    store = TicketStore(db)
    _submit(store, summary="متنِ قدیمی برای حذف")
    _age_rows(db, days=800)
    result = apply_retention(db, interaction_days=0, ticket_anonymize_days=0, ticket_delete_days=730)
    assert result["tickets_deleted"] == 1
    assert store.search("قدیمی") == []
    assert db.conn.execute("SELECT COUNT(*) FROM tickets_fts").fetchone()[0] == 0


def test_retention_zero_means_disabled(db):
    store = TicketStore(db)
    _submit(store)
    _age_rows(db, days=4000)
    result = apply_retention(db, interaction_days=0, ticket_anonymize_days=0, ticket_delete_days=0)
    assert result == {"interactions_deleted": 0, "tickets_anonymized": 0, "tickets_deleted": 0}
    assert store.count() == 1
