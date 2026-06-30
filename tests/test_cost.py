"""
Offline tests for the cost/token engine (no API).
They check cost math, cache savings, unit economics, and the log loader.

Run:  python -m pytest -q tests/test_cost.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reporting.cost import (  # noqa: E402
    Pricing,
    aggregate_log,
    breakdown_from_eval,
    compute_breakdown,
)

PRICING = Pricing(input_per_m=0.14, cache_hit_per_m=0.0028, output_per_m=0.28)


# ---------- Cost math ----------
def test_cost_math_matches_three_tier_formula():
    b = compute_breakdown(
        {"cache_hit": 900_000, "cache_miss": 150_000, "completion": 36_000},
        n_tickets=300, n_calls=300, pricing=PRICING,
    )
    assert b.prompt_tokens == 1_050_000
    assert b.total_tokens == 1_086_000
    expected = (900_000 * 0.0028 + 150_000 * 0.14 + 36_000 * 0.28) / 1_000_000
    assert abs(b.cost_total - expected) < 1e-12
    assert abs(b.cost_per_ticket - expected / 300) < 1e-15
    assert b.cost_per_call == b.cost_per_ticket  # one call per ticket


def test_cache_savings_vs_no_cache_baseline():
    b = compute_breakdown(
        {"cache_hit": 900_000, "cache_miss": 150_000, "completion": 36_000},
        n_tickets=300, pricing=PRICING,
    )
    no_cache = (1_050_000 * 0.14 + 36_000 * 0.28) / 1_000_000
    assert abs(b.cost_without_cache - no_cache) < 1e-12
    assert abs(b.cache_savings - (no_cache - b.cost_total)) < 1e-12
    assert 0.78 < b.cache_savings_pct < 0.79  # caching saves ~79% here
    assert abs(b.cache_hit_rate - 900_000 / 1_050_000) < 1e-12


def test_missing_cache_split_falls_back_to_all_miss():
    # When there is no hit/miss split, the whole prompt is conservatively assumed to be miss.
    b = compute_breakdown({"prompt": 1_000, "completion": 50}, n_tickets=1, pricing=PRICING)
    assert b.cache_hit_tokens == 0
    assert b.cache_miss_tokens == 1_000
    assert b.cache_hit_rate == 0.0
    assert b.cache_savings == 0.0  # without caching, savings are zero


def test_zero_tickets_does_not_divide_by_zero():
    b = compute_breakdown({"cache_hit": 0, "cache_miss": 0, "completion": 0}, n_tickets=0)
    assert b.cost_per_ticket == 0.0
    assert b.tokens_per_ticket == 0.0
    assert b.cache_hit_rate == 0.0


def test_pricing_from_tuple_preserves_order():
    p = Pricing.from_tuple((0.14, 0.0028, 0.28))
    assert (p.input_per_m, p.cache_hit_per_m, p.output_per_m) == (0.14, 0.0028, 0.28)


def test_projection_is_linear_in_volume():
    b = compute_breakdown(
        {"cache_hit": 900_000, "cache_miss": 150_000, "completion": 36_000},
        n_tickets=300, pricing=PRICING,
    )
    assert abs(b.project(10_000) - b.cost_per_ticket * 10_000) < 1e-12
    assert abs(b.project(20_000) - 2 * b.project(10_000)) < 1e-9


# ---------- Evaluation output ----------
def test_breakdown_from_eval_uses_n_as_calls():
    res = {"n": 50, "model": "deepseek-test", "latency_ms_avg": 700.0,
           "tokens": {"cache_hit": 100, "cache_miss": 50, "completion": 20}}
    b = breakdown_from_eval(res, pricing=PRICING)
    assert b.n_tickets == 50 and b.n_calls == 50
    assert b.model == "deepseek-test"


# ---------- Log aggregation ----------
def test_aggregate_log_counts_calls_and_tickets(tmp_path):
    log = tmp_path / "interactions.jsonl"
    records = [
        {"event": "round", "session_id": "s1",
         "llm": {"model": "m", "latency_ms": 700,
                 "usage": {"prompt_tokens": 3500, "completion_tokens": 110,
                           "prompt_cache_hit_tokens": 3000, "prompt_cache_miss_tokens": 500}}},
        {"event": "round", "session_id": "s1",
         "llm": {"model": "m", "latency_ms": 800,
                 "usage": {"prompt_tokens": 3600, "completion_tokens": 90,
                           "prompt_cache_hit_tokens": 3400, "prompt_cache_miss_tokens": 200}}},
        {"event": "round", "session_id": "s2",
         "llm": {"model": "m", "latency_ms": 600,
                 "usage": {"prompt_tokens": 3400, "completion_tokens": 120,
                           "prompt_cache_hit_tokens": 3000, "prompt_cache_miss_tokens": 400}}},
        {"event": "session_final", "session_id": "s1"},
        {"event": "session_final", "session_id": "s2"},
    ]
    log.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    b = aggregate_log(log, pricing=PRICING)
    assert b.n_tickets == 2          # two session_final
    assert b.n_calls == 3            # three rounds
    assert abs(b.calls_per_ticket - 1.5) < 1e-12
    assert b.cache_hit_tokens == 3000 + 3400 + 3000
    assert b.cache_miss_tokens == 500 + 200 + 400
    assert b.completion_tokens == 110 + 90 + 120
    assert b.model == "m"
    assert abs(b.latency_ms_avg - (700 + 800 + 600) / 3) < 1e-9


def test_aggregate_log_falls_back_to_unique_sessions_without_final(tmp_path):
    log = tmp_path / "interactions.jsonl"
    records = [
        {"event": "round", "session_id": "a", "llm": {"usage": {"prompt_tokens": 10, "completion_tokens": 2}}},
        {"event": "round", "session_id": "b", "llm": {"usage": {"prompt_tokens": 10, "completion_tokens": 2}}},
    ]
    log.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    b = aggregate_log(log)
    assert b.n_tickets == 2          # no session_final → count unique ids
    assert b.n_calls == 2
    # cache split not reported → all miss
    assert b.cache_hit_tokens == 0 and b.cache_miss_tokens == 20


def test_aggregate_log_skips_corrupt_lines(tmp_path):
    log = tmp_path / "interactions.jsonl"
    log.write_text(
        '{"event":"round","session_id":"s","llm":{"usage":{"prompt_tokens":10,"completion_tokens":2}}}\n'
        "this is a corrupt line\n"
        '{"event":"session_final","session_id":"s"}\n',
        encoding="utf-8",
    )
    b = aggregate_log(log)  # must not crash
    assert b.n_tickets == 1 and b.n_calls == 1
