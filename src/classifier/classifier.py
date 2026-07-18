"""
هستهٔ دسته‌بندی: یک «دور» کامل (یک تیکت -> یک ClassificationOutput).
zaza
این کلاس از وضعیت مکالمه و سقف سوال‌ها چیزی نمی‌داند؛ آن منطق در conversation/manager است.
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


class Classifier:
    def __init__(
        self,
        client: DeepSeekClient | None = None,
        taxonomy: Taxonomy | None = None,
    ) -> None:
        self.taxonomy = taxonomy or load_taxonomy()
        self.client = client or DeepSeekClient()
        # per_combo=5 → مثال‌های متضادِ متوازن برای هر ترکیب (system prompt کش می‌شود).
        self.demonstrations = build_demonstrations(self.taxonomy, per_combo=5)
        self._system = build_system_prompt(self.taxonomy, self.demonstrations)

    def classify(
        self,
        summary: str,
        description: str,
        clarifications: list[tuple[str, str]] | None = None,
    ) -> tuple[ClassificationOutput, dict]:
        """خروجی: (نتیجهٔ دسته‌بندی، متادیتای LLM برای لاگ)."""
        user = build_user_prompt(summary, description, clarifications)

        # مسیر عادی: یک فراخوانی.
        if not settings.enable_self_consistency:
            resp = self.client.complete_json(self._system, user)
            output = parse_and_validate(resp.data, self.taxonomy)
            meta = {"model": resp.model, "latency_ms": round(resp.latency_ms, 1), "usage": resp.usage}
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
        meta = {
            "model": resp.model,
            "latency_ms": round(resp.latency_ms, 1),
            "usage": resp.usage,
            "agreement": agreement,
        }
        return output, meta
