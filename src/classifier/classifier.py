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
from src import observability as obs
from src.classifier.few_shot import build_demonstrations
from src.classifier.output_parser import label_key, parse_and_validate
from src.classifier.schema import ClassificationOutput
from src.llm.client import DeepSeekClient
from src.llm.prompts import _PRECEDENT_DESC_CHARS, build_system_prompt, build_user_prompt
from src.observability.fingerprint import compute_fingerprint
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
        # اثرانگشتِ پیکربندی: هر trace با این مهر می‌خورد تا نسخه‌های config قابلِ‌مقایسه باشند.
        self.fingerprint = compute_fingerprint(self._system)
        if retriever is _AUTO:
            from src.retrieval.retriever import maybe_build_retriever

            self.retriever = maybe_build_retriever()
        else:
            self.retriever = retriever

    def config_snapshot(self) -> dict:
        """نمای قابلِ‌بازرسیِ پیکربندیِ مؤثر — برای /api/debug/config و traceِ راه‌اندازی."""
        layer_ids = [layer.id for layer in self.taxonomy.layers]
        return {
            "fingerprint": self.fingerprint,
            "model": settings.model,
            "system_prompt_chars": len(self._system),
            "retrieval_active": self.retriever is not None,
            "few_shot": {
                "count": len(self.demonstrations),
                "demos": [
                    {
                        "summary": d["input"].get("summary", ""),
                        "labels": {
                            lid: (d["output"]["layers"].get(lid, {}).get("candidates") or [{}])[0].get("label")
                            for lid in layer_ids
                        },
                    }
                    for d in self.demonstrations
                ],
            },
        }

    def classify(
        self,
        summary: str,
        description: str,
        clarifications: list[tuple[str, str]] | None = None,
        exclude_keys: frozenset[str] | set[str] | None = None,  # ارزیابی: حذفِ خودِ تیکت
    ) -> tuple[ClassificationOutput, dict]:
        """خروجی: (نتیجهٔ دسته‌بندی، متادیتای LLM/retrieval برای لاگ و تصمیم)."""
        with obs.span(
            "classify",
            input={"summary": summary, "description": description,
                   "clarifications": clarifications or []},
            metadata={"config_fingerprint": self.fingerprint["fingerprint"]},
        ) as cls_span:
            retrieval = None
            explain: dict = {}
            if self.retriever is not None:
                with obs.span("retrieval", as_type="retriever") as ret_span:
                    try:
                        retrieval = self.retriever.retrieve(
                            summary, description, clarifications,
                            exclude_keys=exclude_keys,
                            drop_self_sim=0.995 if exclude_keys else None,
                            explain=explain,
                        )
                    except Exception as e:  # retrieval هرگز نباید دسته‌بندی را بیندازد
                        log.warning("retrieval ناموفق بود؛ بدونِ سابقه ادامه می‌دهیم: %s", e)
                        explain["error"] = str(e)
                    ret_span.update(
                        input={k: explain.get(k) for k in
                               ("query_text", "k_demos", "purity_k", "sim_floor", "model")},
                        output=self._retrieval_payload(retrieval, explain),
                    )

            user = build_user_prompt(
                summary, description, clarifications,
                precedents=retrieval.demos if retrieval else None,
            )

            # مسیر عادی: یک فراخوانی.
            if not settings.enable_self_consistency:
                resp = self.client.complete_json(self._system, user)
                output = parse_and_validate(resp.data, self.taxonomy)
                meta = self._meta(resp, retrieval, explain)
                self._finish_span(cls_span, output)
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
            meta = self._meta(resp, retrieval, explain)
            meta["agreement"] = agreement
            self._finish_span(cls_span, output)
            return output, meta

    @staticmethod
    def _retrieval_payload(retrieval, explain: dict) -> dict:
        """خروجیِ کاملِ retrieval برای trace: همسایه‌های واقعاً تزریق‌شده + رای‌ها + دلیلِ کناره‌گیری."""
        if retrieval is None:
            return {
                "abstained": True,
                "abstain_reason": explain.get("abstain_reason") or explain.get("error"),
                "top_similarity": explain.get("top_similarity"),
            }
        return {
            "abstained": False,
            "top_similarity": round(retrieval.top_similarity, 4),
            # هر همسایه دقیقاً همان‌طور که به prompt تزریق شد (توضیحات کوتاه‌شده)
            "neighbors": [
                {
                    "key": d.get("key"),
                    "similarity": round(s, 4),
                    "labels": {k: d.get(k) for k in ("layer1", "layer2") if d.get(k)},
                    "summary": d.get("summary", ""),
                    "description": (d.get("description") or "")[:_PRECEDENT_DESC_CHARS],
                }
                for d, s in zip(retrieval.demos, retrieval.demo_sims)
            ],
            "knn_votes": retrieval.votes_dict(),
        }

    def _finish_span(self, cls_span, output: ClassificationOutput) -> None:
        cls_span.update(
            output={
                "labels": {
                    lid: (lo.top.label if lo.top else None)
                    for lid, lo in output.layers.items()
                },
                "needs_clarification": {
                    lid: lo.needs_clarification for lid, lo in output.layers.items()
                },
                "clarifying_question": output.clarifying_question,
                "reasoning": output.reasoning,
            }
        )

    def _meta(self, resp, retrieval, explain: dict | None = None) -> dict:
        return {
            "model": resp.model,
            "latency_ms": round(resp.latency_ms, 1),
            "usage": resp.usage,
            "retrieval": retrieval.meta() if retrieval else None,
            "retrieval_explain": explain or None,
            "knn_votes": retrieval.votes_dict() if retrieval else None,
            "config_fingerprint": self.fingerprint["fingerprint"],
        }
