"""تست‌های آفلاینِ لایهٔ وب (FAQ، اندپوینت‌های API، سلامت، degradation) — بدون کلید API."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api.app as api_app
import src.db.database as db_module
from src.db.database import Database
from src.faq import FaqItem, load_faq, normalize, search_faq
from src.llm.client import LLMUnavailableError
from src.tickets.store import TicketStore


# ---------------------------------------------------------------------------
# FAQ
# ---------------------------------------------------------------------------
def test_faq_file_loads_and_is_wellformed():
    categories, items = load_faq()
    assert len(items) >= 12, "expected a substantial FAQ set"
    assert categories, "categories must not be empty"
    for it in items:
        assert it.id and it.question and it.summary and it.description
        assert it.category in categories
        assert it.keywords


def test_faq_ids_are_unique():
    _, items = load_faq()
    ids = [it.id for it in items]
    assert len(ids) == len(set(ids))


def test_normalize_maps_arabic_and_digits():
    assert normalize("كيفيت") == normalize("کیفیت")
    assert normalize("۱۲۳") == "123"
    assert normalize("ABC") == "abc"


def test_search_matches_english_and_persian_keywords():
    _, items = load_faq()
    assert any("loan" in it.question.lower() for it in search_faq(items, "loan"))
    assert search_faq(items, "وام")  # کلیدواژهٔ فارسی
    assert search_faq(items, "punch")
    assert search_faq(items, "no-such-keyword-xyz") == []


def test_search_requires_all_terms_and_respects_category():
    items = [
        FaqItem(id="a", category="X", question="loan guarantor error", summary="s", description="d"),
        FaqItem(id="b", category="Y", question="loan activation", summary="s", description="d"),
    ]
    assert [it.id for it in search_faq(items, "loan guarantor")] == ["a"]
    assert [it.id for it in search_faq(items, "loan", category="Y")] == ["b"]
    assert len(search_faq(items, "")) == 2


# ---------------------------------------------------------------------------
# API (بدون دست‌زدن به LLM واقعی)
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch):
    db = Database(tmp_path / "app.db")
    db.migrate()
    # DB و store پیش‌فرض به فایلِ موقت اشاره کنند (تستِ هرمتیک)
    monkeypatch.setattr(db_module, "_default_db", db)
    monkeypatch.setattr(api_app, "_store", TicketStore(db))
    return TestClient(api_app.app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "version" in body


def test_ready_reports_components(client):
    res = client.get("/ready")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ready"
    assert body["checks"]["db"].startswith("ok")
    assert "llm" in body["checks"] and "retrieval" in body["checks"]


def test_metrics_endpoint(client):
    client.get("/health")
    body = client.get("/metrics").json()
    assert "uptime_seconds" in body
    assert any(k.startswith("http_GET_/health") for k in body["counters"])


def test_faq_endpoint(client):
    data = client.get("/api/faq").json()
    assert len(data["items"]) >= 12
    assert data["categories"]


def test_index_serves_spa(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Ticket Assistant" in res.text


def test_submit_ticket_endpoint(client):
    res = client.post("/api/tickets", json={
        "employee_id": "263669",
        "first_name": "Sara",
        "last_name": "Ahmadi",
        "summary": "Punch not recorded",
        "description": "details…",
        "labels": {"layer1": "incident", "layer2": "erp"},
        "needs_review": True,
        "session_id": "xyz",
    })
    assert res.status_code == 200
    body = res.json()
    assert body["reference"].startswith("TKT-")
    assert body["needs_review"] is True


def test_submit_ticket_requires_identity_and_content(client):
    assert client.post("/api/tickets", json={
        "employee_id": "", "first_name": "a", "last_name": "b", "summary": "s",
    }).status_code == 422
    assert client.post("/api/tickets", json={
        "employee_id": "1", "first_name": "a", "last_name": "b",
        "summary": "  ", "description": "",
    }).status_code == 422


def test_submit_ticket_rejects_oversized_text(client):
    assert client.post("/api/tickets", json={
        "employee_id": "1", "first_name": "a", "last_name": "b",
        "summary": "s", "description": "x" * 4001,
    }).status_code == 422


# ---------------------------------------------------------------------------
# Degradation: قطعیِ DeepSeek → 503 ساخت‌یافته؛ ثبتِ تیکت باید همچنان کار کند
# ---------------------------------------------------------------------------
class _DownManager:
    def start(self, summary, description):
        raise LLMUnavailableError("circuit open")

    def answer(self, session_id, answer):
        raise LLMUnavailableError("circuit open")


def test_classify_returns_structured_503_when_llm_down(client, monkeypatch):
    monkeypatch.setattr(api_app, "_manager", _DownManager())
    res = client.post("/classify/start", json={"summary": "s", "description": "d"})
    assert res.status_code == 503
    assert res.json()["detail"]["code"] == "llm_unavailable"
    assert "Retry-After" in res.headers

    res = client.post("/classify/answer", json={"session_id": "s1", "answer": "a"})
    assert res.status_code == 503

    # ثبتِ دستی مستقل از LLM است
    res = client.post("/api/tickets", json={
        "employee_id": "1", "first_name": "a", "last_name": "b",
        "summary": "خطا در پانچ", "needs_review": True,
    })
    assert res.status_code == 200


# ---------------------------------------------------------------------------
# اندپوینت‌های ادمین (توکن‌دار)
# ---------------------------------------------------------------------------
def test_admin_endpoints_disabled_without_token(client):
    assert client.get("/api/tickets", params={"q": "x"}).status_code == 404


def test_admin_search_with_token(client, monkeypatch):
    from config.settings import settings

    monkeypatch.setattr(settings, "admin_api_token", "t0ken")
    client.post("/api/tickets", json={
        "employee_id": "1", "first_name": "a", "last_name": "b",
        "summary": "مشکل تایم‌شیت", "description": "تایید نمی‌شود",
    })
    assert client.get("/api/tickets", headers={"X-Admin-Token": "wrong"}).status_code == 401
    res = client.get(
        "/api/tickets", params={"q": "تایم شیت"}, headers={"X-Admin-Token": "t0ken"}
    )
    assert res.status_code == 200
    assert res.json()["count"] == 1
    ref = res.json()["items"][0]["reference"]
    got = client.get(f"/api/tickets/{ref}", headers={"X-Admin-Token": "t0ken"})
    assert got.status_code == 200
    assert got.json()["summary"] == "مشکل تایم‌شیت"
