"""
موتورِ متریک‌های دقت — استخراجِ Precision/Recall/F1 و آمادگیِ عملیاتی از خروجیِ eval.

ورودی: dictِ `res` که `scripts.eval_incdb.run_evaluation` می‌سازد (دقت، per-class recall،
ماتریسِ درهم‌ریختگی، و سطل‌های confidence). این ماژول از همان داده، متریک‌های مشتق را
می‌سازد تا گزارشِ HTML فقط «نمایش» بدهد و هیچ منطقی در لایهٔ نمایش نباشد.

تعریف‌ها (از روی ماتریسِ درهم‌ریختگیِ هر لایه):
  • Recall(c)    = درست‌های c / کلِ واقعی‌های c            (سطر)
  • Precision(c) = درست‌های c / کلِ پیش‌بینی‌شده‌های c       (ستون)
  • F1(c)        = میانگینِ هارمونیکِ Precision و Recall
آمادگیِ عملیاتی از سطل‌های confidence (auto در برابر needs-review) می‌آید.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ClassMetric:
    id: str
    name: str
    support: int          # تعداد واقعیِ این کلاس (مخرجِ recall)
    predicted: int        # تعداد دفعاتی که مدل این کلاس را پیش‌بینی کرد (مخرجِ precision)
    tp: int               # درست‌ها (قطرِ ماتریس)

    @property
    def recall(self) -> float:
        return self.tp / self.support if self.support else 0.0

    @property
    def precision(self) -> float:
        return self.tp / self.predicted if self.predicted else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass(frozen=True)
class LayerMetric:
    id: str
    name: str
    accuracy: float
    correct: int
    total: int
    label_ids: list[str]
    names: dict           # {label_id: display_name}
    confusion: dict       # {true_id: {pred_id|None: count}}
    classes: list[ClassMetric] = field(default_factory=list)

    @property
    def macro_f1(self) -> float:
        return sum(c.f1 for c in self.classes) / len(self.classes) if self.classes else 0.0

    @property
    def macro_precision(self) -> float:
        return sum(c.precision for c in self.classes) / len(self.classes) if self.classes else 0.0

    @property
    def macro_recall(self) -> float:
        return sum(c.recall for c in self.classes) / len(self.classes) if self.classes else 0.0


@dataclass(frozen=True)
class OperationalReadiness:
    """auto در برابر needs-review — ترجمهٔ دقت به ارزشِ عملیاتی."""

    auto_total: int
    auto_correct: int
    review_total: int
    review_correct: int

    @property
    def total(self) -> int:
        return self.auto_total + self.review_total

    @property
    def auto_coverage(self) -> float:
        return self.auto_total / self.total if self.total else 0.0

    @property
    def review_share(self) -> float:
        return self.review_total / self.total if self.total else 0.0

    @property
    def auto_accuracy(self) -> float:
        return self.auto_correct / self.auto_total if self.auto_total else 0.0

    @property
    def review_accuracy(self) -> float:
        return self.review_correct / self.review_total if self.review_total else 0.0

    @property
    def has_data(self) -> bool:
        return self.total > 0


@dataclass(frozen=True)
class PerfMetrics:
    n: int
    model: str | None
    errors: int
    overall_accuracy: float
    overall_correct: int
    overall_total: int
    layers: list[LayerMetric]
    readiness: OperationalReadiness
    latency_ms_avg: float
    target: float

    @property
    def passed(self) -> bool:
        return self.overall_accuracy >= self.target


def _predicted_counts(label_ids: list[str], confusion: dict) -> dict:
    """مخرجِ precision: چند بار هر برچسب پیش‌بینی شد (مجموعِ ستون)."""
    counts: dict[str, int] = {c: 0 for c in label_ids}
    for true_id in confusion:
        for pred_id, n in confusion[true_id].items():
            if pred_id in counts:
                counts[pred_id] += n
    return counts


def from_eval(res: dict, *, target: float = 0.90) -> PerfMetrics:
    """ساختِ `PerfMetrics` از خروجیِ `run_evaluation`."""
    layers: list[LayerMetric] = []
    for L in res.get("layers", []):
        names = {c["id"]: c["name"] for c in L["classes"]}
        confusion = L.get("confusion", {})
        pred_counts = _predicted_counts(L["label_ids"], confusion)
        classes = []
        for c in L["classes"]:
            cid = c["id"]
            tp = confusion.get(cid, {}).get(cid, 0)
            classes.append(
                ClassMetric(
                    id=cid, name=c["name"], support=c["total"],
                    predicted=pred_counts.get(cid, 0), tp=tp,
                )
            )
        layers.append(
            LayerMetric(
                id=L["id"], name=L["name"], accuracy=L["accuracy"],
                correct=L["correct"], total=L["total"], label_ids=list(L["label_ids"]),
                names=names, confusion=confusion, classes=classes,
            )
        )

    conf = res.get("confidence", {}) or {}
    readiness = OperationalReadiness(
        auto_total=conf.get("confident_total", 0),
        auto_correct=conf.get("confident_correct", 0),
        review_total=conf.get("flagged_total", 0),
        review_correct=conf.get("flagged_correct", 0),
    )

    ov = res.get("overall", {})
    return PerfMetrics(
        n=res.get("n", 0),
        model=res.get("model"),
        errors=res.get("errors", 0),
        overall_accuracy=ov.get("accuracy", 0.0),
        overall_correct=ov.get("correct", 0),
        overall_total=ov.get("total", 0),
        layers=layers,
        readiness=readiness,
        latency_ms_avg=res.get("latency_ms_avg", 0.0),
        target=target,
    )
