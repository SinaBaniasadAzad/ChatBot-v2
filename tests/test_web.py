"""تست‌های آفلاینِ لایهٔ وب (FAQ، TicketStore، اندپوینت‌های API) — بدون نیاز به کلید API."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import src.api.app as api_app
from src.faq import FaqItem, load_faq, normalize, search_faq
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
# TicketStore
# ---------------------------------------------------------------------------
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


def test_ticket_store_sequential_references_and_persistence(tmp_path):
    path = tmp_path / "tickets.jsonl"
    store = TicketStore(path)
    r1 = _submit(store)
    r2 = _submit(store)
    assert r1["reference"].startswith("TKT-")
    assert r1["reference"] != r2["reference"]
    assert int(r2["reference"].rsplit("-", 1)[1]) == int(r1["reference"].rsplit("-", 1)[1]) + 1

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["employee_id"] == "263669"
    assert rec["labels"] == {"layer1": "incident", "layer2": "erp"}


def test_ticket_store_resumes_counter_after_restart(tmp_path):
    path = tmp_path / "tickets.jsonl"
    _submit(TicketStore(path))
    r = _submit(TicketStore(path))  # پروسهٔ جدید → شمارنده باید از فایل بازیابی شود
    assert r["reference"].endswith("00002")


# ---------------------------------------------------------------------------
# API (بدون دست‌زدن به مسیرهای classify که کلید می‌خواهند)
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_app, "_store", TicketStore(tmp_path / "tickets.jsonl"))
    return TestClient(api_app.app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


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
