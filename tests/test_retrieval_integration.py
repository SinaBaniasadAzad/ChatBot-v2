"""تست‌های آفلاینِ ادغامِ retrieval در خطِ دسته‌بندی (بدونِ مدل/API).

پوشش: گِیتِ اطمینان (شواهدِ توهمی، مخالفتِ kNN)، بازیاب با انکودرِ تزریقی،
تزریقِ سابقه به پیامِ کاربر، و گاردهای نشت/کفِ شباهت.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.classifier.decision import Action, decide
from src.classifier.output_parser import parse_and_validate
from src.llm.prompts import build_user_prompt
from src.retrieval.retriever import TicketRetriever
from src.taxonomy import load_taxonomy

TAX = load_taxonomy()


def _output(l1, l1_ev, l2, l2_ev, q=None):
    raw = {
        "layers": {
            "layer1": {"candidates": [{"label": l1, "evidence": l1_ev}], "needs_clarification": False},
            "layer2": {"candidates": [{"label": l2, "evidence": l2_ev}], "needs_clarification": False},
        },
        "clarifying_question": q,
        "suggested_summary": "s",
        "reasoning": "r",
    }
    return parse_and_validate(raw, TAX)


TICKET = "پانچ ورود و خروج من در سامانه ثبت نشده است"


# ---------------------------------------------------------------------------
# راستی‌آزماییِ شواهد (درمانِ «شاهدِ توهمی = اطمینانِ کاذب»)
# ---------------------------------------------------------------------------
def test_hallucinated_evidence_triggers_clarification():
    out = _output("incident", ["خطای پرداخت"], "erp", ["پانچ"])  # شاهدِ لایه۱ در متن نیست
    d = decide(out, TAX, 0, 2, ticket_text=TICKET)
    assert d.action == Action.ASK
    assert "hallucinated_evidence" in d.layer_decisions["layer1"].reasons
    assert d.layer_decisions["layer2"].ambiguous is False


def test_verified_evidence_passes_and_halfspace_tolerated():
    out = _output("incident", ["ثبت نشده"], "erp", ["ورود و خروج"])
    d = decide(out, TAX, 0, 2, ticket_text=TICKET)
    assert d.action == Action.DONE
    assert d.ambiguity_reasons == {}


def test_verification_skipped_without_ticket_text_backcompat():
    out = _output("incident", ["هرچیزی"], "erp", ["پانچ"])
    assert decide(out, TAX, 0, 2).action == Action.DONE  # فراخوانیِ قدیمی


# ---------------------------------------------------------------------------
# گِیتِ kNN (سابقهٔ خالصِ مخالف → سوال؛ موافق/ناخالص → عبور)
# ---------------------------------------------------------------------------
def _votes(label, purity):
    return {"layer1": {"label": label, "purity": purity, "n": 15}}


def test_pure_knn_disagreement_forces_question():
    out = _output("incident", ["ثبت نشده"], "erp", ["پانچ"])
    d = decide(out, TAX, 0, 2, ticket_text=TICKET,
               knn_votes=_votes("service_request", 0.92), knn_disagree_purity=0.8)
    assert d.action == Action.ASK
    assert any("knn_disagreement" in r for r in d.layer_decisions["layer1"].reasons)


def test_low_purity_disagreement_does_not_block():
    out = _output("incident", ["ثبت نشده"], "erp", ["پانچ"])
    d = decide(out, TAX, 0, 2, ticket_text=TICKET,
               knn_votes=_votes("service_request", 0.55), knn_disagree_purity=0.8)
    assert d.action == Action.DONE


def test_knn_agreement_does_not_block():
    out = _output("incident", ["ثبت نشده"], "erp", ["پانچ"])
    d = decide(out, TAX, 0, 2, ticket_text=TICKET,
               knn_votes=_votes("incident", 0.95), knn_disagree_purity=0.8)
    assert d.action == Action.DONE


def test_budget_exhausted_flags_for_review():
    out = _output("incident", ["ثبت نشده"], "erp", ["پانچ"])
    d = decide(out, TAX, 2, 2, ticket_text=TICKET,
               knn_votes=_votes("service_request", 0.92), knn_disagree_purity=0.8)
    assert d.action == Action.FALLBACK and d.needs_review


# ---------------------------------------------------------------------------
# بازیاب (با انکودرِ تزریقی — بدونِ دانلود)
# ---------------------------------------------------------------------------
@pytest.fixture()
def retriever(tmp_path):
    dim = 4
    rows = [
        {"key": f"T-{i}", "layer1": lbl1, "layer2": lbl2,
         "summary": f"s{i}", "description": f"d{i}", "embed_text": f"s{i}. d{i}"}
        for i, (lbl1, lbl2) in enumerate(
            [("incident", "erp")] * 4 + [("service_request", "staff")] * 3
        )
    ]
    emb = np.zeros((len(rows), dim), dtype=np.float32)
    emb[:4, 0] = 1.0   # خوشهٔ A: incident/erp
    emb[4:, 1] = 1.0   # خوشهٔ B: sr/staff
    pool = tmp_path / "pool.jsonl"
    pool.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    np.savez(tmp_path / "index.npz", emb=emb.astype(np.float16),
             keys=np.array([r["key"] for r in rows]), model=np.array("fake"))

    def encode_fn(text):
        v = np.zeros(dim, dtype=np.float32)
        v[0] = 1.0  # کوئری همیشه نزدیکِ خوشهٔ A
        return v

    return TicketRetriever(tmp_path / "index.npz", pool, encode_fn=encode_fn)


def test_retriever_votes_and_demos(retriever):
    res = retriever.retrieve("پانچ", "ثبت نشد", k_demos=3, purity_k=7, sim_floor=0.3)
    assert res is not None
    assert [d["key"] for d in res.demos] == ["T-0", "T-1", "T-2"]
    assert res.votes["layer1"].label == "incident"
    assert res.votes["layer1"].purity == 1.0  # وزنِ خوشهٔ B صفر است (cos=0)
    assert res.top_similarity == pytest.approx(1.0, abs=1e-3)


def test_retriever_excludes_keys_and_near_self(retriever):
    res = retriever.retrieve("x", "y", k_demos=2, purity_k=4, sim_floor=0.3,
                             exclude_keys={"T-0"}, drop_self_sim=1.5)
    assert "T-0" not in [d["key"] for d in res.demos]

    # همهٔ خوشهٔ A شبه‌خودی حساب می‌شود؛ فقط خوشهٔ B (شباهت ۰) می‌ماند:
    # با کفِ صفر، near-selfها رد و Bها برگردانده می‌شوند...
    res2 = retriever.retrieve("x", "y", k_demos=2, purity_k=4, sim_floor=0.0,
                              drop_self_sim=0.99)
    assert all(d["key"].startswith("T-") and int(d["key"][2:]) >= 4 for d in res2.demos)
    # ...ولی با کفِ واقعی (۰.۳) سابقهٔ بی‌ربط تزریق نمی‌شود → کناره‌گیری (None).
    assert retriever.retrieve("x", "y", k_demos=2, purity_k=4, sim_floor=0.3,
                              drop_self_sim=0.99) is None


def test_retriever_abstains_below_sim_floor(retriever):
    assert retriever.retrieve("x", "y", sim_floor=1.01) is None


# ---------------------------------------------------------------------------
# تزریقِ سابقه به پیامِ کاربر (system prompt دست‌نخورده)
# ---------------------------------------------------------------------------
def test_user_prompt_includes_precedents_block():
    precedents = [{"summary": "دسترسی تایم شیت", "description": "ایجاد شود",
                   "layer1": "service_request", "layer2": "erp"}]
    p = build_user_prompt("s", "d", precedents=precedents)
    assert "PRECEDENTS" in p
    assert '"service_request"' in p
    assert p.index("PRECEDENTS") < p.index("Ticket:")


def test_user_prompt_unchanged_without_precedents():
    p = build_user_prompt("s", "d", [("q1", "a1")])
    assert p.splitlines()[0] == "Ticket:"
    assert "PRECEDENTS" not in p and "Q: q1" in p
