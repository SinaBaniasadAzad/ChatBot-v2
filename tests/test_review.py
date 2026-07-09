"""تست‌های آفلاینِ صفِ بازبینی: store + ورودِ خودکارِ جلساتِ needs_review از manager."""
from __future__ import annotations

import pytest

from config.settings import settings
from src.classifier.output_parser import parse_and_validate
from src.conversation.manager import ConversationManager
from src.observability import tracing
from src.review.store import ReviewStore
from src.taxonomy import load_taxonomy
from src.utils.interaction_log import InteractionLogger

TAX = load_taxonomy()


@pytest.fixture(autouse=True)
def _no_langfuse(monkeypatch):
    monkeypatch.setattr(settings, "observability_enabled", False)
    tracing._reset_for_tests()
    yield
    tracing._reset_for_tests()


@pytest.fixture()
def store(tmp_path) -> ReviewStore:
    return ReviewStore(tmp_path / "review.db")


def _enqueue(store, **over):
    base = dict(
        source="production",
        session_id="sess-1",
        trace_id="tr-1",
        summary="پانچ ثبت نشد",
        description="ورود و خروج من ثبت نمی‌شود",
        predicted_labels={"layer1": "incident", "layer2": "erp"},
        ambiguity_reasons={"layer1": ["no_evidence"]},
    )
    base.update(over)
    return store.enqueue(**base)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
def test_enqueue_list_get_roundtrip(store):
    item_id = _enqueue(store)
    assert item_id is not None
    items = store.list_items(status="pending")
    assert len(items) == 1
    it = store.get(item_id)
    assert it["summary"] == "پانچ ثبت نشد"
    assert it["predicted_labels"] == {"layer1": "incident", "layer2": "erp"}
    assert it["ambiguity_reasons"] == {"layer1": ["no_evidence"]}
    assert it["status"] == "pending"


def test_enqueue_dedupes_same_session(store):
    a = _enqueue(store)
    b = _enqueue(store)  # همان session → نباید آیتمِ دوم بسازد
    assert a == b
    assert len(store.list_items(status="pending", limit=10)) == 1


def test_resolve_sets_gold_and_blocks_double_close(store):
    item_id = _enqueue(store)
    it = store.resolve(item_id, {"layer1": "service_request", "layer2": "erp"}, "sina", "توضیح")
    assert it["status"] == "resolved"
    assert it["gold_labels"] == {"layer1": "service_request", "layer2": "erp"}
    assert it["reviewer"] == "sina" and it["resolved_at"]
    # بستنِ دوباره ممنوع
    assert store.resolve(item_id, {"layer1": "incident"}, "x") is None
    assert store.dismiss(item_id, "x") is None


def test_dismiss(store):
    item_id = _enqueue(store)
    it = store.dismiss(item_id, "sina", "برچسبِ دیتاست خودش غلط است")
    assert it["status"] == "dismissed"
    assert store.list_items(status="pending") == []


def test_stats_agreement(store):
    a = _enqueue(store, session_id="s-a")
    b = _enqueue(store, session_id="s-b")
    store.resolve(a, {"layer1": "incident", "layer2": "erp"}, "r")        # ۲/۲ موافق
    store.resolve(b, {"layer1": "service_request", "layer2": "erp"}, "r")  # ۱/۲ موافق
    s = store.stats()
    assert s["by_status"]["resolved"] == 2
    assert s["resolved_label_count"] == 4
    assert s["model_human_agreement"] == pytest.approx(0.75)


def test_export_gold_only_resolved(store):
    a = _enqueue(store, session_id="s-a", ticket_key="INC-1")
    _enqueue(store, session_id="s-b")  # pending می‌ماند
    store.resolve(a, {"layer1": "incident", "layer2": "staff"}, "r")
    rows = store.export_gold()
    assert len(rows) == 1
    assert rows[0]["key"] == "INC-1"
    assert rows[0]["layer1"] == "incident" and rows[0]["layer2"] == "staff"
    assert rows[0]["source"] == "human_review"


# ---------------------------------------------------------------------------
# manager → ورودِ خودکار به صف پس از fallback
# ---------------------------------------------------------------------------
class _AmbiguousClassifier:
    """کلاسیفایرِ ساختگی: همیشه بدونِ شاهد → مبهم → ASK تا اتمامِ بودجه → FALLBACK."""

    taxonomy = TAX

    def classify(self, summary, description, clarifications=None, exclude_keys=None):
        raw = {
            "layers": {
                "layer1": {"candidates": [{"label": "incident", "evidence": []}],
                           "needs_clarification": True},
                "layer2": {"candidates": [{"label": "erp", "evidence": []}],
                           "needs_clarification": True},
            },
            "clarifying_question": "کدام سامانه؟",
            "suggested_summary": "s",
            "reasoning": "r",
        }
        return parse_and_validate(raw, TAX), {"model": "fake", "latency_ms": 1.0, "usage": {}}


def test_needs_review_session_lands_in_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "max_questions", 2)
    store = ReviewStore(tmp_path / "review.db")
    mgr = ConversationManager(
        classifier=_AmbiguousClassifier(),
        interaction_logger=InteractionLogger(False, tmp_path / "log.jsonl"),
        review_store=store,
    )
    r1 = mgr.start("وام", "وام من جلو نمی‌رود")
    assert r1["status"] == "need_info" and r1["question"]
    r2 = mgr.answer(r1["session_id"], "پاسخ اول")
    assert r2["status"] == "need_info"
    r3 = mgr.answer(r1["session_id"], "پاسخ دوم")  # بودجه تمام → fallback → صف
    assert r3["status"] == "completed_low_confidence"
    assert r3["result"]["needs_review"] is True

    items = store.list_items(status="pending")
    assert len(items) == 1
    it = items[0]
    assert it["session_id"] == r1["session_id"]
    assert it["source"] == "production"
    assert it["predicted_labels"] == {"layer1": "incident", "layer2": "erp"}
    assert len(it["clarifications"]) == 2


def test_confident_session_not_enqueued(tmp_path):
    class _Confident(_AmbiguousClassifier):
        def classify(self, summary, description, clarifications=None, exclude_keys=None):
            raw = {
                "layers": {
                    "layer1": {"candidates": [{"label": "incident", "evidence": ["ثبت نمی‌شود"]}],
                               "needs_clarification": False},
                    "layer2": {"candidates": [{"label": "erp", "evidence": ["پانچ"]}],
                               "needs_clarification": False},
                },
                "clarifying_question": None,
                "suggested_summary": "s",
                "reasoning": "r",
            }
            return parse_and_validate(raw, TAX), {"model": "fake", "latency_ms": 1.0, "usage": {}}

    store = ReviewStore(tmp_path / "review.db")
    mgr = ConversationManager(
        classifier=_Confident(),
        interaction_logger=InteractionLogger(False, tmp_path / "log.jsonl"),
        review_store=store,
    )
    r = mgr.start("پانچ", "پانچ من ثبت نمی‌شود")
    assert r["status"] == "completed"
    assert store.list_items(status="pending") == []
