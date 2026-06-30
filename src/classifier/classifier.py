"""
Classification core: one full "round" (a single ticket -> one ClassificationOutput).

This class knows nothing about conversation state or the question budget;
that logic lives in conversation/manager.
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
        # per_combo=5 → balanced contrasting examples per combo (the system prompt is cached).
        self.demonstrations = build_demonstrations(self.taxonomy, per_combo=5)
        self._system = build_system_prompt(self.taxonomy, self.demonstrations)

    def classify(
        self,
        summary: str,
        description: str,
        clarifications: list[tuple[str, str]] | None = None,
    ) -> tuple[ClassificationOutput, dict]:
        """Returns: (classification result, LLM metadata for logging)."""
        user = build_user_prompt(summary, description, clarifications)

        # Normal path: a single call.
        if not settings.enable_self_consistency:
            resp = self.client.complete_json(self._system, user)
            output = parse_and_validate(resp.data, self.taxonomy)
            meta = {"model": resp.model, "latency_ms": round(resp.latency_ms, 1), "usage": resp.usage}
            return output, meta

        # Optional path: self-consistency to gauge stability.
        resp, agreement = self.client.majority_vote(
            self._system,
            user,
            key_fn=lambda data: label_key(data, self.taxonomy),
            n=settings.self_consistency_samples,
        )
        output = parse_and_validate(resp.data, self.taxonomy)
        # If agreement was low, mark every layer as ambiguous.
        if agreement < 0.6:
            log.info("Self-consistency agreement was low (%.0f%%) -> ambiguous.", agreement * 100)
            for lo in output.layers.values():
                lo.needs_clarification = True
        meta = {
            "model": resp.model,
            "latency_ms": round(resp.latency_ms, 1),
            "usage": resp.usage,
            "agreement": agreement,
        }
        return output, meta
