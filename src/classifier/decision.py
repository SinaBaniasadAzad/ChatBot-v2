"""
منطق تصمیم‌گیریِ Ambiguity-Driven + گِیتِ اطمینانِ مبتنی بر سابقه (kNN).

به‌جای تکیه بر «عدد confidence» (که در LLM کالیبره نیست)، ابهام را بر پایهٔ
*شواهد عینی* تشخیص می‌دهیم. یک لایه مبهم است اگر:
  ۱) مدل خودش needs_clarification=true بدهد، یا
  ۲) کاندیدای برترش هیچ شاهدِ متنیِ *راستی‌آزمایی‌شده* نداشته باشد — شاهدی که
     واقعاً در متنِ تیکت نیست (توهم)، شاهد حساب نمی‌شود (گاردِ ضدِ بیش‌اعتمادی)، یا
  ۳) «نظرِ دومِ» kNN مخالف باشد: اگر همسایگیِ خالصِ تیکت‌های مشابهِ تاریخی
     (purity ≥ آستانه) برچسبِ دیگری بدهد، اعتمادِ LLM مشکوک است → سوال بپرس.
     (این همان درمانِ «مسیریابیِ مطمئن در عینِ ندانستن» است: در دادهٔ واقعی،
     همسایگی‌های با خلوصِ ≥۰.۸ حدودِ ۹۷٪ با برچسبِ درست هم‌خوانند.)

تصمیم نهایی، با در نظر گرفتن بودجهٔ سوال (حداکثر ۲):
  - هیچ لایهٔ مبهمی نیست            -> DONE
  - لایهٔ مبهم هست و بودجه باقی است -> ASK (یک سوال هدفمند)
  - لایهٔ مبهم هست و بودجه تمام شد  -> FALLBACK (بهترین حدس + flag بازبینی)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.classifier.schema import ClassificationOutput, LayerOutput
from src.taxonomy import Taxonomy
from src.utils.normalize import contains_cue


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
    reasons: list[str] = field(default_factory=list)  # چرا مبهم شد (برای لاگ/تحلیل)


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

    @property
    def ambiguity_reasons(self) -> dict[str, list[str]]:
        return {lid: d.reasons for lid, d in self.layer_decisions.items() if d.reasons}


def _verified_evidence(evidence: list[str], ticket_text: str) -> list[str]:
    """فقط شواهدی که واقعاً در متنِ تیکت هستند (با نرمال‌سازی، مقاوم به نیم‌فاصله)."""
    return [e for e in evidence if e and contains_cue(ticket_text, e)]


def _is_ambiguous(lo: LayerOutput) -> bool:
    """ابهامِ خودگزارش‌شده/بدونِ شاهد (سازگار با نسخهٔ قبلی؛ evalها از این استفاده می‌کنند)."""
    top = lo.top
    if top is None:
        return True
    if not top.evidence:  # هیچ شاهد عینی -> اطلاعات احتمالاً کم است
        return True
    return lo.needs_clarification


def _layer_ambiguity(
    lo: LayerOutput,
    ticket_text: str,
    knn_vote: dict | None,
    knn_disagree_purity: float,
    verify_evidence: bool,
) -> tuple[bool, list[str], list[str]]:
    """خروجی: (مبهم است؟، دلایل، شواهدِ معتبر)."""
    top = lo.top
    if top is None:
        return True, ["no_candidate"], []

    evidence = top.evidence
    reasons: list[str] = []
    if verify_evidence and ticket_text:
        evidence = _verified_evidence(top.evidence, ticket_text)
        if top.evidence and not evidence:
            reasons.append("hallucinated_evidence")
    if not evidence:
        if "hallucinated_evidence" not in reasons:
            reasons.append("no_evidence")
    if lo.needs_clarification:
        reasons.append("model_requested")

    # گِیتِ سابقه: همسایگیِ خالصِ تاریخی که مخالفِ LLM رای می‌دهد = علامتِ خطر.
    if (
        knn_vote
        and knn_vote.get("label")
        and knn_vote.get("purity", 0.0) >= knn_disagree_purity
        and knn_vote["label"] != top.label
    ):
        reasons.append(
            f"knn_disagreement(precedent={knn_vote['label']},purity={knn_vote['purity']:.2f})"
        )

    return bool(reasons), reasons, evidence


def _fallback_question(tax: Taxonomy, ambiguous_layer_ids: list[str]) -> str:
    """اگر مدل سوالی نساخت، یک سوال عمومیِ هدفمند می‌سازیم (گزینه‌ها را نام می‌برد —
    برای حوزه عملاً یعنی «کدام سامانه: ERP یا Staff؟»)."""
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
    *,
    ticket_text: str = "",
    knn_votes: dict[str, dict] | None = None,
    knn_disagree_purity: float | None = None,
    verify_evidence: bool = True,
) -> Decision:
    """پارامترهای جدید keyword-only هستند؛ فراخوانی‌های قدیمی بدونِ تغییر کار می‌کنند.

    knn_votes: {layer_id: {"label": str, "purity": float, "n": int}} — از retriever.
    """
    if knn_disagree_purity is None:
        from config.settings import settings

        knn_disagree_purity = settings.knn_disagree_purity

    layer_decisions: dict[str, LayerDecision] = {}
    ambiguous_ids: list[str] = []

    for layer in tax.layers:
        lo = output.layers.get(layer.id) or LayerOutput()
        amb, reasons, evidence = _layer_ambiguity(
            lo,
            ticket_text,
            (knn_votes or {}).get(layer.id),
            knn_disagree_purity,
            verify_evidence,
        )
        top = lo.top
        layer_decisions[layer.id] = LayerDecision(
            layer_id=layer.id,
            label=top.label if top else None,
            ambiguous=amb,
            evidence=evidence,
            reasons=reasons,
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
