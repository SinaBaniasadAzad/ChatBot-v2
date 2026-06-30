"""
Unit tests that do not connect to the DeepSeek API (offline).
They check normalize logic, taxonomy loading, schema validation, and the
Ambiguity-Driven decision.

Run:  python -m pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.decision import Action, decide  # noqa: E402
from src.classifier.output_parser import parse_and_validate  # noqa: E402
from src.taxonomy import load_taxonomy  # noqa: E402
from src.utils.normalize import contains_cue, normalize  # noqa: E402

TAX = load_taxonomy()


# ---------- normalize ----------
def test_normalize_unifies_arabic_and_zwnj():
    assert normalize("تایم‌شیت") == normalize("تایم شیت")
    assert normalize("كارمند") == normalize("کارمند")  # Arabic kaf -> Persian
    assert normalize("۱۲۳") == "123"


def test_contains_cue_half_space():
    assert contains_cue("درخواست تایم‌شیت اپرور", "تایم شیت")


# ---------- schema validation ----------
def test_parse_coerces_messy_shapes():
    # evidence as a string, and candidates as a single dict (not a list)
    raw = {
        "layers": {
            "layer1": {"candidates": {"label": "incident", "evidence": "خطا"}, "needs_clarification": False},
            "layer2": {"candidates": [{"label": "erp", "evidence": ["پانچ"]}]},
        },
    }
    out = parse_and_validate(raw, TAX)
    assert out.layers["layer1"].top.label == "incident"
    assert out.layers["layer1"].top.evidence == ["خطا"]
    assert out.layers["layer2"].top.label == "erp"


def test_parse_never_crashes_on_garbage():
    out = parse_and_validate({"unexpected": 123}, TAX)
    # every layer must be valid and ambiguous (not crash)
    for layer in TAX.layers:
        assert out.layers[layer.id].needs_clarification is True


def test_parse_drops_hallucinated_label():
    raw = {
        "layers": {
            "layer1": {"candidates": [{"label": "made_up", "evidence": ["x"]}], "needs_clarification": False},
            "layer2": {"candidates": [{"label": "erp", "evidence": ["تایم شیت"]}], "needs_clarification": False},
        },
        "clarifying_question": None,
        "suggested_summary": "s",
        "reasoning": "r",
    }
    out = parse_and_validate(raw, TAX)
    # the fake label is dropped and the layer becomes ambiguous (fallback to all labels)
    assert out.layers["layer1"].needs_clarification is True
    assert all(c.label in TAX.get_layer("layer1").label_ids for c in out.layers["layer1"].candidates)


# ---------- decision (ambiguity-driven) ----------
def _output(l1_label, l1_ev, l2_label, l2_ev, q=None):
    raw = {
        "layers": {
            "layer1": {"candidates": [{"label": l1_label, "evidence": l1_ev}], "needs_clarification": False},
            "layer2": {"candidates": [{"label": l2_label, "evidence": l2_ev}], "needs_clarification": False},
        },
        "clarifying_question": q,
        "suggested_summary": "s",
        "reasoning": "r",
    }
    return parse_and_validate(raw, TAX)


def test_decide_done_when_evidence_present():
    out = _output("incident", ["خطا"], "erp", ["تایم شیت"])
    d = decide(out, TAX, questions_asked=0, max_questions=2)
    assert d.action == Action.DONE
    assert d.labels == {"layer1": "incident", "layer2": "erp"}


def test_decide_asks_when_no_evidence_and_budget_left():
    out = _output("incident", ["خطا"], "erp", [], q="مربوط به وام است یا حضور و غیاب؟")
    d = decide(out, TAX, questions_asked=0, max_questions=2)
    assert d.action == Action.ASK
    assert d.question


def test_decide_fallback_when_budget_exhausted():
    out = _output("incident", ["خطا"], "erp", [])  # layer 2 is ambiguous
    d = decide(out, TAX, questions_asked=2, max_questions=2)
    assert d.action == Action.FALLBACK
    assert d.needs_review is True
