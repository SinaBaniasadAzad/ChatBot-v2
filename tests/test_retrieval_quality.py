"""تست‌های صحتِ انتخابِ همسایه‌ها و شفافیتِ retrieval (آفلاین، با انکودرِ تزریقی).

مکملِ test_retrieval_integration: این‌جا تمرکز روی «قراردادِ observability» است —
explain (دلیلِ کناره‌گیری/پارامترها) و payloadِ traceِ همسایه‌ها که باید دقیقاً
همان چیزی باشد که به prompt تزریق می‌شود.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.classifier.classifier import Classifier
from src.llm.prompts import _PRECEDENT_DESC_CHARS
from src.retrieval.retriever import RetrievalResult, TicketRetriever


@pytest.fixture()
def retriever(tmp_path):
    """۷ سند در دو خوشهٔ متعامد؛ کوئری همیشه به خوشهٔ A (incident/erp) می‌چسبد."""
    dim = 4
    rows = [
        {"key": f"T-{i}", "layer1": lbl1, "layer2": lbl2,
         "summary": f"s{i}", "description": ("d" * 400) + f"-{i}",
         "embed_text": f"s{i}"}
        for i, (lbl1, lbl2) in enumerate(
            [("incident", "erp")] * 4 + [("service_request", "staff")] * 3
        )
    ]
    emb = np.zeros((len(rows), dim), dtype=np.float32)
    emb[:4, 0] = 1.0
    emb[4:, 1] = 1.0
    pool = tmp_path / "pool.jsonl"
    pool.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    np.savez(tmp_path / "index.npz", emb=emb.astype(np.float16),
             keys=np.array([r["key"] for r in rows]), model=np.array("fake"))

    def encode_fn(text):
        v = np.zeros(dim, dtype=np.float32)
        v[0] = 1.0
        return v

    return TicketRetriever(tmp_path / "index.npz", pool, encode_fn=encode_fn)


# ---------------------------------------------------------------------------
# explain — شفافیتِ کناره‌گیری
# ---------------------------------------------------------------------------
def test_explain_records_params_on_success(retriever):
    explain: dict = {}
    res = retriever.retrieve("پانچ", "ثبت نشد", k_demos=3, purity_k=5, sim_floor=0.3,
                             explain=explain)
    assert res is not None
    assert explain["k_demos"] == 3 and explain["purity_k"] == 5
    assert explain["sim_floor"] == 0.3
    assert explain["pool_size"] == 7
    assert explain["query_text"]
    assert "abstain_reason" not in explain


def test_explain_below_sim_floor(retriever):
    explain: dict = {}
    res = retriever.retrieve("x", "y", sim_floor=1.01, explain=explain)
    assert res is None
    assert explain["abstain_reason"] == "below_sim_floor"
    assert explain["top_similarity"] == pytest.approx(1.0, abs=1e-3)


def test_explain_empty_query(retriever):
    explain: dict = {}
    # boilerplate خالص → متنِ تمیزِ خالی → کناره‌گیری با دلیلِ empty_query
    res = retriever.retrieve("", "", explain=explain)
    assert res is None
    assert explain["abstain_reason"] == "empty_query"


def test_explain_is_optional_backcompat(retriever):
    # فراخوانیِ قدیمی بدونِ explain باید مثلِ قبل کار کند
    assert retriever.retrieve("x", "y", sim_floor=0.3) is not None


# ---------------------------------------------------------------------------
# payloadِ trace — همان چیزی که واقعاً به prompt تزریق شد
# ---------------------------------------------------------------------------
def test_retrieval_payload_mirrors_injected_precedents(retriever):
    res = retriever.retrieve("پانچ", "ثبت نشد", k_demos=3, purity_k=5, sim_floor=0.3)
    payload = Classifier._retrieval_payload(res, {})
    assert payload["abstained"] is False
    assert [n["key"] for n in payload["neighbors"]] == [d["key"] for d in res.demos]
    for n, s in zip(payload["neighbors"], res.demo_sims):
        assert n["similarity"] == pytest.approx(round(s, 4))
        assert n["labels"] == {"layer1": "incident", "layer2": "erp"}
        # توضیحات دقیقاً با همان سقفِ prompt کوتاه می‌شود (بدونِ اختلافِ trace/واقعیت)
        assert len(n["description"]) <= _PRECEDENT_DESC_CHARS
    assert payload["knn_votes"]["layer1"]["label"] == "incident"
    assert payload["top_similarity"] == pytest.approx(round(res.top_similarity, 4))


def test_retrieval_payload_on_abstain():
    payload = Classifier._retrieval_payload(
        None, {"abstain_reason": "below_sim_floor", "top_similarity": 0.31}
    )
    assert payload == {
        "abstained": True, "abstain_reason": "below_sim_floor", "top_similarity": 0.31,
    }


def test_retrieval_payload_on_error():
    payload = Classifier._retrieval_payload(None, {"error": "boom"})
    assert payload["abstained"] is True
    assert payload["abstain_reason"] == "boom"


# ---------------------------------------------------------------------------
# صحتِ انتخاب: همسایه‌ها به‌ترتیبِ شباهت و از خوشهٔ درست
# ---------------------------------------------------------------------------
def test_neighbors_come_from_correct_cluster_sorted(retriever):
    res = retriever.retrieve("پانچ", "ثبت نشد", k_demos=4, purity_k=7, sim_floor=0.3)
    keys = [d["key"] for d in res.demos]
    assert keys == ["T-0", "T-1", "T-2", "T-3"]  # خوشهٔ A، مرتب بر اساسِ شباهت
    assert res.demo_sims == sorted(res.demo_sims, reverse=True)
    assert res.votes["layer2"].label == "erp"
    assert res.votes["layer2"].purity == pytest.approx(1.0)


def test_exclude_keys_removes_leakage(retriever):
    res = retriever.retrieve("x", "y", k_demos=4, purity_k=7, sim_floor=0.3,
                             exclude_keys=frozenset({"T-0", "T-1"}))
    keys = [d["key"] for d in res.demos]
    assert "T-0" not in keys and "T-1" not in keys
