"""گزارش‌گیریِ هزینه و توکن (ارائه‌محور).

این پکیج منطقِ مشترکِ محاسبهٔ هزینه/توکن را نگه می‌دارد تا هر دو خروجی —
گزارشِ HTML (`scripts/cost_report.py`) و داشبوردِ تصویری (`scripts/report.py`) —
از یک منبعِ واحد عدد بگیرند و هیچ‌وقت با هم اختلاف نداشته باشند.
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
