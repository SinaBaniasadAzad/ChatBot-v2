"""
Shared cost & token engine — the **single source of truth** for the numbers.

Why separate? Both the HTML report and the visual dashboard must show *exactly*
the same number. This module is fed by **real** data:
  • the output of `scripts.eval_incdb.run_evaluation` (a real model run over the dataset), or
  • the production interaction log: `logs/interactions.jsonl` (each `round` is one API call).

DeepSeek pricing has three tiers (per 1M tokens):
  • cache-miss input   (price_in)
  • cache-hit input    (price_cache)  ← much cheaper; the main savings lever
  • output/completion  (price_out)

No rate here is "absolute truth"; the defaults are merely "assumptions" and
must be confirmed against current DeepSeek pricing (they are labeled
transparently in the output too).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Pricing (assumptions — confirm against current rates). Unit: USD per 1M tokens.
# Defaults match the common path of scripts/report.evaluate_and_report.
# ---------------------------------------------------------------------------
DEFAULT_PRICE_IN = 0.14      # cache-miss input
DEFAULT_PRICE_CACHE = 0.0028  # cache-hit input
DEFAULT_PRICE_OUT = 0.28     # output (completion)

_MILLION = 1_000_000


@dataclass(frozen=True)
class Pricing:
    """DeepSeek's three-tier rate per 1M tokens (USD)."""

    input_per_m: float = DEFAULT_PRICE_IN
    cache_hit_per_m: float = DEFAULT_PRICE_CACHE
    output_per_m: float = DEFAULT_PRICE_OUT

    @classmethod
    def from_tuple(cls, prices: tuple[float, float, float]) -> "Pricing":
        """Backwards compatible with the legacy (price_in, price_cache, price_out) order."""
        price_in, price_cache, price_out = prices
        return cls(input_per_m=price_in, cache_hit_per_m=price_cache, output_per_m=price_out)


@dataclass(frozen=True)
class CostBreakdown:
    """A full cost/token breakdown — everything the report needs."""

    # Token counts
    cache_hit_tokens: int
    cache_miss_tokens: int
    completion_tokens: int
    # Units of work
    n_tickets: int
    n_calls: int
    # Metadata
    model: str | None
    latency_ms_avg: float
    pricing: Pricing

    # ---- Derived token counts ----
    @property
    def prompt_tokens(self) -> int:
        return self.cache_hit_tokens + self.cache_miss_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Share of cached input tokens (0..1)."""
        return self.cache_hit_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    # ---- Derived cost (USD) ----
    @property
    def cost_cache_hit(self) -> float:
        return self.cache_hit_tokens * self.pricing.cache_hit_per_m / _MILLION

    @property
    def cost_cache_miss(self) -> float:
        return self.cache_miss_tokens * self.pricing.input_per_m / _MILLION

    @property
    def cost_input(self) -> float:
        return self.cost_cache_hit + self.cost_cache_miss

    @property
    def cost_output(self) -> float:
        return self.completion_tokens * self.pricing.output_per_m / _MILLION

    @property
    def cost_total(self) -> float:
        return self.cost_input + self.cost_output

    @property
    def cost_without_cache(self) -> float:
        """Hypothetical cost with no cache (all input at the cache-miss rate)."""
        return (
            self.prompt_tokens * self.pricing.input_per_m + self.completion_tokens * self.pricing.output_per_m
        ) / _MILLION

    @property
    def cache_savings(self) -> float:
        """Dollars saved thanks to prompt caching."""
        return self.cost_without_cache - self.cost_total

    @property
    def cache_savings_pct(self) -> float:
        base = self.cost_without_cache
        return (self.cache_savings / base) if base else 0.0

    # ---- Unit economics ----
    @property
    def cost_per_ticket(self) -> float:
        return self.cost_total / self.n_tickets if self.n_tickets else 0.0

    @property
    def cost_per_call(self) -> float:
        return self.cost_total / self.n_calls if self.n_calls else 0.0

    @property
    def tokens_per_ticket(self) -> float:
        return self.total_tokens / self.n_tickets if self.n_tickets else 0.0

    @property
    def calls_per_ticket(self) -> float:
        return self.n_calls / self.n_tickets if self.n_tickets else 0.0

    def project(self, monthly_tickets: int) -> float:
        """Project monthly cost — simply an extrapolation of the *measured* cost per ticket."""
        return self.cost_per_ticket * monthly_tickets

    def as_dict(self) -> dict:
        """A flat view for logging/testing/serialization."""
        return {
            "model": self.model,
            "n_tickets": self.n_tickets,
            "n_calls": self.n_calls,
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_rate": self.cache_hit_rate,
            "cost_input": self.cost_input,
            "cost_output": self.cost_output,
            "cost_total": self.cost_total,
            "cost_without_cache": self.cost_without_cache,
            "cache_savings": self.cache_savings,
            "cost_per_ticket": self.cost_per_ticket,
            "cost_per_call": self.cost_per_call,
            "tokens_per_ticket": self.tokens_per_ticket,
            "latency_ms_avg": self.latency_ms_avg,
        }


def _split_cache(tokens: dict) -> tuple[int, int, int]:
    """
    Extract (cache_hit, cache_miss, completion) from a token dict.

    If the hit/miss split is absent from the data (some calls don't report it),
    the entire available prompt is conservatively assumed to be cache-miss (the
    most expensive case) so cost is never underestimated.
    """
    hit = int(tokens.get("cache_hit", 0) or 0)
    miss = int(tokens.get("cache_miss", 0) or 0)
    completion = int(tokens.get("completion", 0) or 0)
    if hit == 0 and miss == 0:
        prompt = int(tokens.get("prompt", 0) or 0)
        miss = prompt
    return hit, miss, completion


def compute_breakdown(
    tokens: dict,
    *,
    n_tickets: int,
    n_calls: int | None = None,
    model: str | None = None,
    latency_ms_avg: float = 0.0,
    pricing: Pricing | None = None,
) -> CostBreakdown:
    """Build a `CostBreakdown` from a token dict (keys: prompt/cache_hit/cache_miss/completion)."""
    hit, miss, completion = _split_cache(tokens)
    return CostBreakdown(
        cache_hit_tokens=hit,
        cache_miss_tokens=miss,
        completion_tokens=completion,
        n_tickets=int(n_tickets or 0),
        n_calls=int(n_calls if n_calls is not None else n_tickets or 0),
        model=model,
        latency_ms_avg=float(latency_ms_avg or 0.0),
        pricing=pricing or Pricing(),
    )


def breakdown_from_eval(res: dict, pricing: Pricing | None = None) -> CostBreakdown:
    """A `CostBreakdown` from the output of `scripts.eval_incdb.run_evaluation`.

    In single-shot evaluation each ticket has exactly one call, so n_calls == n.
    """
    return compute_breakdown(
        res.get("tokens", {}),
        n_tickets=res.get("n", 0),
        n_calls=res.get("n", 0),
        model=res.get("model"),
        latency_ms_avg=res.get("latency_ms_avg", 0.0),
        pricing=pricing,
    )


def aggregate_log(path: str | Path, pricing: Pricing | None = None) -> CostBreakdown:
    """
    Aggregate token usage from the production log (`logs/interactions.jsonl`).

    Contract (from `src/utils/interaction_log.py`):
      • each `round` event        = one API call; tokens are in `llm.usage`.
      • each `session_final` event = one completed ticket (the cost-economics unit).
    """
    tokens = {"cache_hit": 0, "cache_miss": 0, "completion": 0}
    n_calls = 0
    n_finals = 0
    session_ids: set[str] = set()
    latency_sum = 0.0
    latency_n = 0
    model: str | None = None

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue  # a corrupt line must not break the whole report

        event = rec.get("event")
        if event == "round":
            n_calls += 1
            if rec.get("session_id"):
                session_ids.add(rec["session_id"])
            llm = rec.get("llm") or {}
            if llm.get("model"):
                model = llm["model"]
            lat = llm.get("latency_ms")
            if lat:
                latency_sum += float(lat)
                latency_n += 1
            usage = llm.get("usage") or {}
            pt = int(usage.get("prompt_tokens", 0) or 0)
            ct = int(usage.get("completion_tokens", 0) or 0)
            hit = usage.get("prompt_cache_hit_tokens")
            miss = usage.get("prompt_cache_miss_tokens")
            if hit is None or miss is None:
                hit, miss = 0, pt  # split not reported → conservatively all miss
            tokens["cache_hit"] += int(hit or 0)
            tokens["cache_miss"] += int(miss or 0)
            tokens["completion"] += ct
        elif event == "session_final":
            n_finals += 1

    # n_tickets: prefer the count of finalized sessions; otherwise unique session ids.
    n_tickets = n_finals or len(session_ids)
    latency_ms_avg = latency_sum / latency_n if latency_n else 0.0
    return compute_breakdown(
        tokens,
        n_tickets=n_tickets,
        n_calls=n_calls,
        model=model,
        latency_ms_avg=latency_ms_avg,
        pricing=pricing,
    )
