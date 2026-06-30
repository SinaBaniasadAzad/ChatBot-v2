"""Cost and token reporting (presentation-oriented).

This package holds the shared cost/token computation logic so that both
outputs — the HTML report (`scripts/cost_report.py`) and the visual dashboard
(`scripts/report.py`) — draw their numbers from a single source and never
disagree.
"""
from src.reporting.cost import (
    CostBreakdown,
    Pricing,
    aggregate_log,
    breakdown_from_eval,
    compute_breakdown,
)

__all__ = [
    "CostBreakdown",
    "Pricing",
    "aggregate_log",
    "breakdown_from_eval",
    "compute_breakdown",
]
