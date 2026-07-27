"""
هستهٔ دسته‌بندی: یک «دور» کامل (یک تیکت -> یک ClassificationOutput).

این کلاس از وضعیت مکالمه و سقف سوال‌ها چیزی نمی‌داند؛ آن منطق در conversation/manager است.

Retrieval-augmented: اگر بازیابِ سابقه فعال باشد، پیش از فراخوانیِ LLM تیکت‌های
مشابهِ تاریخی بازیابی می‌شوند و (۱) به‌عنوانِ سابقهٔ برچسب‌خورده واردِ پیامِ کاربر
می‌شوند، (۲) رایِ kNNشان همراهِ متادیتا برمی‌گردد تا لایهٔ تصمیم برای گِیتِ
اطمینان استفاده کند. نبودِ ایندکس/وابستگی‌ها = رفتارِ قبلی، بدونِ خطا.
"""
from __future__ import annotations

from config.settings import settings
from src.classifier.few_shot import build_demonstrations
from src.classifier.output_parser import label_key, parse_and_validate
from src.classifier.schema import ClassificationOutput
from src.llm.client import DeepSeekClient
from src.llm.prompts import build_system_prompt, build_user_prompt
from src.taxonomy import Taxonomy, load_taxonomy
from src.utils.logging import get_logger

log = get_logger("classifier")

_AUTO = object()  # sentinel: ساختِ خودکارِ بازیاب طبقِ settings


class Classifier:
    def __init__(
        self,
        client: DeepSeekClient | None = None,
        taxonomy: Taxonomy | None = None,
        retriever=_AUTO,  # TicketRetriever | None | _AUTO
    ) -> None:
        self.taxonomy = taxonomy or load_taxonomy()
        self.client = client or DeepSeekClient()
        # per_combo=5 → مثال‌های متضادِ متوازن برای هر ترکیب (system prompt کش می‌شود).
        self.demonstrations = build_demonstrations(self.taxonomy, per_combo=5)
        self._system = build_system_prompt(self.taxonomy, self.demonstrations)
        if retriever is _AUTO:
            from src.retrieval.retriever import maybe_build_retriever

            self.retriever = maybe_build_retriever()
        else:
            self.retriever = retriever

    def classify(
        self,
        summary: str,
        description: str,
        clarifications: list[tuple[str, str]] | None = None,
        exclude_keys: frozenset[str] | set[str] | None = None,  # ارزیابی: حذفِ خودِ تیکت
    ) -> tuple[ClassificationOutput, dict]:
        """خروجی: (نتیجهٔ دسته‌بندی، متادیتای LLM/retrieval برای لاگ و تصمیم)."""
        retrieval = None
        if self.retriever is not None:
            try:
                retrieval = self.retriever.retrieve(
                    summary, description, clarifications,
                    exclude_keys=exclude_keys,
                    drop_self_sim=0.995 if exclude_keys else None,
                )
            except Exception as e:  # retrieval هرگز نباید دسته‌بندی را بیندازد
                log.warning("retrieval ناموفق بود؛ بدونِ سابقه ادامه می‌دهیم: %s", e)

        user = build_user_prompt(
            summary, description, clarifications,
            precedents=retrieval.demos if retrieval else None,
        )

        # مسیر عادی: یک فراخوانی.
        if not settings.enable_self_consistency:
            resp = self.client.complete_json(self._system, user)
            output = parse_and_validate(resp.data, self.taxonomy)
            meta = self._meta(resp, retrieval)
            return output, meta

        # مسیر اختیاری: self-consistency برای سنجش پایداری.
        resp, agreement = self.client.majority_vote(
            self._system,
            user,
            key_fn=lambda data: label_key(data, self.taxonomy),
            n=settings.self_consistency_samples,
        )
        output = parse_and_validate(resp.data, self.taxonomy)
        # اگر توافق پایین بود، همهٔ لایه‌ها را به‌عنوان مبهم علامت بزن.
        if agreement < 0.6:
            log.info("توافق self-consistency پایین بود (%.0f%%) -> ابهام.", agreement * 100)
            for lo in output.layers.values():
                lo.needs_clarification = True
        meta = self._meta(resp, retrieval)
        meta["agreement"] = agreement
        return output, meta

    @staticmethod
    def _meta(resp, retrieval) -> dict:
        return {
            "model": resp.model,
            "latency_ms": round(resp.latency_ms, 1),
            "usage": resp.usage,
            "retrieval": retrieval.meta() if retrieval else None,
            "knn_votes": retrieval.votes_dict() if retrieval else None,
        }
