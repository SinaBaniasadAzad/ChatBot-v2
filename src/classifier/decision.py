"""
منطق تصمیم‌گیریِ Ambiguity-Driven.

به‌جای تکیه بر «عدد confidence» (که در LLM کالیبره نیست)، ابهام را بر پایهٔ
*شواهد عینی* تشخیص می‌دهیم:
  یک لایه مبهم است اگر:
    - مدل خودش needs_clarification=true داده باشد، یا
    - کاندیدای برترش هیچ شاهد متنی‌ای نداشته باشد (گارد ضدِ بیش‌اعتمادی).

تصمیم نهایی، با در نظر گرفتن بودجهٔ سوال:
  - هیچ لایهٔ مبهمی نیست            -> DONE
  - لایهٔ مبهم هست و بودجه باقی است -> ASK (یک سوال هدفمند)
  - لایهٔ مبهم هست و بودجه تمام شد  -> FALLBACK (بهترین حدس + flag بازبینی)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.classifier.schema import ClassificationOutput, LayerOutput
from src.taxonomy import Taxonomy


class Action(str, Enum):
    DONE = "done"
    ASK = "ask"
    FALLBACK = "fallback"


@dataclass
class LayerDecision:
    layer_id: str
    label: str | None
    ambiguous: bool
    evidence: list[str] = field(default_factory=list)


@dataclass
class Decision:
    action: Action
    layer_decisions: dict[str, LayerDecision]
    question: str | None = None

    @property
    def labels(self) -> dict[str, str | None]:
        return {lid: d.label for lid, d in self.layer_decisions.items()}

    @property
    def needs_review(self) -> bool:
        return self.action == Action.FALLBACK


def _is_ambiguous(lo: LayerOutput) -> bool:
    top = lo.top
    if top is None:
        return True
    if not top.evidence:  # هیچ شاهد عینی -> اطلاعات احتمالاً کم است
        return True
    return lo.needs_clarification


def _fallback_question(tax: Taxonomy, ambiguous_layer_ids: list[str]) -> str:
    """اگر مدل سوالی نساخت، یک سوال عمومیِ هدفمند می‌سازیم."""
    names = []
    for lid in ambiguous_layer_ids:
        layer = tax.get_layer(lid)
        if layer:
            opts = " یا ".join(lbl.name for lbl in layer.labels)
            names.append(f"«{layer.name}» ({opts})")
    target = " و ".join(names) if names else "موضوع"
    return f"برای دسته‌بندی دقیق‌تر، لطفاً کمی بیشتر دربارهٔ {target} توضیح دهید."


def decide(
    output: ClassificationOutput,
    tax: Taxonomy,
    questions_asked: int,
    max_questions: int,
) -> Decision:
    layer_decisions: dict[str, LayerDecision] = {}
    ambiguous_ids: list[str] = []

    for layer in tax.layers:
        lo = output.layers.get(layer.id) or LayerOutput()
        amb = _is_ambiguous(lo)
        top = lo.top
        layer_decisions[layer.id] = LayerDecision(
            layer_id=layer.id,
            label=top.label if top else None,
            ambiguous=amb,
            evidence=top.evidence if top else [],
        )
        if amb:
            ambiguous_ids.append(layer.id)

    if not ambiguous_ids:
        return Decision(Action.DONE, layer_decisions)

    if questions_asked < max_questions:
        question = output.clarifying_question or _fallback_question(tax, ambiguous_ids)
        return Decision(Action.ASK, layer_decisions, question=question)

    # بودجهٔ سوال تمام شد: بهترین حدس + علامت بازبینی انسانی.
    return Decision(Action.FALLBACK, layer_decisions)
