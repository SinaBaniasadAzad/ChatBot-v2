"""تست‌های آفلاینِ لایهٔ observability — بدونِ سرورِ Langfuse و بدونِ کلید.

اصلِ تحتِ آزمون: خاموش‌بودن/نبودنِ Langfuse هرگز نباید مسیرِ اصلی را بشکند (no-op کامل).
"""
from __future__ import annotations

import pytest

from config.settings import settings
from src.observability import tracing
from src.observability.cost import cost_details, split_usage, usage_details
from src.observability.fingerprint import compute_fingerprint
from src.reporting.cost import Pricing


@pytest.fixture(autouse=True)
def _no_langfuse(monkeypatch):
    """ردیابی در تست‌ها همیشه خاموش (حتی اگر .env کلید داشته باشد)."""
    monkeypatch.setattr(settings, "observability_enabled", False)
    tracing._reset_for_tests()
    yield
    tracing._reset_for_tests()


# ---------------------------------------------------------------------------
# facade — رفتارِ no-op
# ---------------------------------------------------------------------------
def test_disabled_client_is_none_and_enabled_false():
    assert tracing.get_client() is None
    assert tracing.enabled() is False


def test_no_keys_means_disabled(monkeypatch):
    monkeypatch.setattr(settings, "observability_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")
    tracing._reset_for_tests()
    assert tracing.get_client() is None


def test_span_and_generation_are_noop_and_absorb_everything():
    with tracing.span("x", input={"a": 1}, as_type="retriever") as s:
        s.update(output={"b": 2})           # نباید استثنا بدهد
        s.set_trace_io(input={"a": 1}, output={"b": 2})
        assert s.trace_id is None
    with tracing.generation("g", model="m", input=[{"role": "user", "content": "hi"}]) as g:
        g.update(output={"ok": True}, usage_details={"input": 1}, cost_details={"total": 0.1})


def test_trace_context_and_helpers_are_noop():
    with tracing.trace_context(session_id="s", trace_name="t", metadata={"k": "v"}):
        tracing.score_current_trace("needs_review", 1)
        assert tracing.current_trace_id() is None
    tracing.score_trace("some-trace", "human_correct", 0.5)
    tracing.flush()


def test_trace_url():
    assert tracing.trace_url(None) is None
    url = tracing.trace_url("abc123")
    assert url.endswith("/trace/abc123")
    assert url.startswith(settings.langfuse_host.rstrip("/"))


# ---------------------------------------------------------------------------
# سازگاری با SDK واقعی (بدونِ سرور): اگر kwargsِ facade با API v4 نخواند،
# span به NOOP سقوط می‌کند و trace_id تهی می‌ماند → این تست آن را لو می‌دهد.
# ---------------------------------------------------------------------------
def test_facade_matches_real_sdk_api(monkeypatch):
    pytest.importorskip("langfuse")
    monkeypatch.setattr(settings, "observability_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")
    monkeypatch.setattr(settings, "langfuse_host", "http://127.0.0.1:1")  # عمداً بسته
    tracing._reset_for_tests()
    try:
        assert tracing.enabled() is True
        with tracing.trace_context(session_id="sess-1", trace_name="t",
                                   metadata={"config_fingerprint": "abc"}):
            with tracing.span("round", input={"a": 1}) as root:
                assert root.trace_id, "spanِ واقعی باید trace_id بدهد (kwargs با v4 نمی‌خواند؟)"
                with tracing.span("retrieval", as_type="retriever") as r:
                    r.update(output={"abstained": True})
                with tracing.generation("gen", model="m",
                                        input=[{"role": "user", "content": "hi"}],
                                        model_parameters={"temperature": 0.0}) as g:
                    g.update(output={"ok": 1},
                             usage_details={"input": 1, "output": 2, "total": 3},
                             cost_details={"input": 0.1, "output": 0.2, "total": 0.3})
                    assert g.trace_id == root.trace_id
                root.update(output={"done": True},
                            metadata={"model": "m"})
                tracing.score_current_trace("needs_review", 0)
                assert tracing.current_trace_id() == root.trace_id
        tracing.score_trace(root.trace_id, "human_correct", 1.0)
    finally:
        tracing._reset_for_tests()  # shutdown؛ ارسال به پورتِ بسته سریع شکست می‌خورد


# ---------------------------------------------------------------------------
# config fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_is_deterministic_and_sensitive_to_prompt():
    a = compute_fingerprint("SYSTEM PROMPT v1")
    b = compute_fingerprint("SYSTEM PROMPT v1")
    c = compute_fingerprint("SYSTEM PROMPT v2")
    assert a["fingerprint"] == b["fingerprint"]
    assert a["fingerprint"] != c["fingerprint"]
    assert set(a["components"]) == {"taxonomy", "examples", "system_prompt", "behavior"}


def test_fingerprint_sensitive_to_behavior_settings(monkeypatch):
    a = compute_fingerprint("p")
    monkeypatch.setattr(settings, "knn_disagree_purity", 0.99)
    b = compute_fingerprint("p")
    assert a["fingerprint"] != b["fingerprint"]
    assert a["components"]["taxonomy"] == b["components"]["taxonomy"]  # فایل‌ها ثابت‌اند


# ---------------------------------------------------------------------------
# cost — هم‌خوانی با موتورِ گزارشِ موجود
# ---------------------------------------------------------------------------
USAGE = {
    "prompt_tokens": 1000,
    "completion_tokens": 200,
    "prompt_cache_hit_tokens": 700,
    "prompt_cache_miss_tokens": 300,
}


def test_split_usage_with_and_without_cache_breakdown():
    assert split_usage(USAGE) == (700, 300, 200)
    # بدونِ تفکیک → محافظه‌کارانه همه cache-miss
    assert split_usage({"prompt_tokens": 50, "completion_tokens": 5}) == (0, 50, 5)
    assert split_usage(None) == (0, 0, 0)


def test_usage_details_shape():
    d = usage_details(USAGE)
    assert d == {"input": 300, "input_cache_read": 700, "output": 200, "total": 1200}


def test_cost_details_matches_reporting_engine():
    p = Pricing(input_per_m=0.14, cache_hit_per_m=0.0028, output_per_m=0.28)
    c = cost_details(USAGE, pricing=p)
    assert c["input"] == pytest.approx(300 * 0.14 / 1e6)
    assert c["input_cache_read"] == pytest.approx(700 * 0.0028 / 1e6)
    assert c["output"] == pytest.approx(200 * 0.28 / 1e6)
    assert c["total"] == pytest.approx(c["input"] + c["input_cache_read"] + c["output"])
