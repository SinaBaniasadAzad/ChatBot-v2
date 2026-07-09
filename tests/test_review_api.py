"""تست‌های آفلاینِ اندپوینت‌های صفِ بازبینی (بدونِ کلیدِ API و بدونِ Langfuse)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.api.app as api_app
from config.settings import settings
from src.observability import tracing
from src.review.store import ReviewStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "observability_enabled", False)
    tracing._reset_for_tests()
    store = ReviewStore(tmp_path / "review.db")
    monkeypatch.setattr(api_app, "_review_store", store)
    monkeypatch.setattr(api_app, "_review_store_init", True)
    c = TestClient(api_app.app)
    c.review_store = store  # دسترسیِ تست‌ها به store
    yield c
    tracing._reset_for_tests()


def _seed(store: ReviewStore, **over) -> int:
    base = dict(
        source="production",
        session_id="sess-api-1",
        trace_id="tr-9",
        summary="خطا در تایید ماموریت",
        description="ماموریت من تایید نمی‌شود",
        predicted_labels={"layer1": "incident", "layer2": "erp"},
    )
    base.update(over)
    return store.enqueue(**base)


def test_queue_lists_items_with_stats_and_trace_url(client):
    _seed(client.review_store)
    data = client.get("/api/review/queue").json()
    assert data["stats"]["by_status"]["pending"] == 1
    item = data["items"][0]
    assert item["summary"] == "خطا در تایید ماموریت"
    assert item["trace_url"].endswith("/trace/tr-9")


def test_get_single_item_and_404(client):
    item_id = _seed(client.review_store)
    assert client.get(f"/api/review/items/{item_id}").status_code == 200
    assert client.get("/api/review/items/99999").status_code == 404


def test_resolve_validates_labels_against_taxonomy(client):
    item_id = _seed(client.review_store)
    bad = client.post(f"/api/review/items/{item_id}/resolve", json={
        "labels": {"layer1": "not_a_label"}, "reviewer": "sina",
    })
    assert bad.status_code == 422
    bad2 = client.post(f"/api/review/items/{item_id}/resolve", json={
        "labels": {"nope": "incident"}, "reviewer": "sina",
    })
    assert bad2.status_code == 422


def test_resolve_then_double_resolve_conflicts(client):
    item_id = _seed(client.review_store)
    ok = client.post(f"/api/review/items/{item_id}/resolve", json={
        "labels": {"layer1": "service_request", "layer2": "erp"},
        "reviewer": "sina", "notes": "درخواستِ اداریِ تمیز",
    })
    assert ok.status_code == 200
    body = ok.json()
    assert body["status"] == "resolved"
    assert body["gold_labels"] == {"layer1": "service_request", "layer2": "erp"}
    again = client.post(f"/api/review/items/{item_id}/resolve", json={
        "labels": {"layer1": "incident", "layer2": "erp"}, "reviewer": "x",
    })
    assert again.status_code == 409


def test_dismiss_endpoint(client):
    item_id = _seed(client.review_store)
    res = client.post(f"/api/review/items/{item_id}/dismiss",
                      json={"reviewer": "sina", "notes": "duplicate"})
    assert res.status_code == 200
    assert res.json()["status"] == "dismissed"


def test_export_returns_resolved_gold_rows(client):
    item_id = _seed(client.review_store, ticket_key="INC-7")
    client.post(f"/api/review/items/{item_id}/resolve", json={
        "labels": {"layer1": "incident", "layer2": "staff"}, "reviewer": "sina",
    })
    data = client.get("/api/review/export").json()
    assert data["count"] == 1
    assert data["rows"][0]["key"] == "INC-7"
    assert data["rows"][0]["layer2"] == "staff"


def test_taxonomy_endpoint_is_dynamic(client):
    data = client.get("/api/review/taxonomy").json()
    ids = {L["id"] for L in data["layers"]}
    assert {"layer1", "layer2"} <= ids
    layer1 = next(L for L in data["layers"] if L["id"] == "layer1")
    assert {"incident", "service_request"} <= {lbl["id"] for lbl in layer1["labels"]}


def test_review_page_served(client):
    res = client.get("/review")
    assert res.status_code == 200
    assert "Review Queue" in res.text
