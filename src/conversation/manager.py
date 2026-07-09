"""
Orchestrator مکالمه — مغز سیستم.

جریان: classify -> decide -> (پایان | یک سوال | fallback). سقف سوال‌ها اینجا اعمال
می‌شود، نه در مدل. وضعیت هر جلسه با session_id نگه‌داری می‌شود تا پاسخِ سوال تکمیلی
به همان context بچسبد.

Observability: هر دور یک traceِ ریشه («classification-round») می‌سازد که با
session_id گروه‌بندی می‌شود؛ زیرـspanها (retrieval/generation/decision) داخلِ
Classifier و DeepSeekClient ساخته می‌شوند. جلساتِ needs_review خودکار واردِ
صفِ بازبینیِ انسانی می‌شوند.
"""
from __future__ import annotations

from config.settings import settings
from src import observability as obs
from src.classifier.classifier import Classifier
from src.classifier.decision import Action, decide
from src.conversation.state import Session, Status
from src.review.store import ReviewStore, maybe_build_review_store
from src.utils.interaction_log import InteractionLogger
from src.utils.logging import get_logger

log = get_logger("conversation")

_AUTO = object()  # sentinel: ساختِ خودکارِ صفِ بازبینی طبقِ settings


class ConversationManager:
    def __init__(
        self,
        classifier: Classifier | None = None,
        interaction_logger: InteractionLogger | None = None,
        review_store: ReviewStore | None = _AUTO,  # None = خاموش
    ) -> None:
        self.classifier = classifier or Classifier()
        self.taxonomy = self.classifier.taxonomy
        self.interaction_logger = interaction_logger or InteractionLogger(
            settings.interaction_log_enabled, settings.interaction_log_path
        )
        self.review_store = (
            maybe_build_review_store() if review_store is _AUTO else review_store
        )
        self._sessions: dict[str, Session] = {}  # تولید: Redis/DB

    # ---- API عمومی ----
    def start(self, summary: str, description: str) -> dict:
        session = Session(summary=summary, description=description)
        self._sessions[session.session_id] = session
        return self._run(session)

    def answer(self, session_id: str, user_answer: str) -> dict:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("session_id نامعتبر است یا منقضی شده.")
        if session.pending_question is None:
            return self._response(session)  # چیزی پرسیده نشده بود
        session.add_clarification(session.pending_question, user_answer)
        return self._run(session)

    # ---- منطق داخلی ----
    def _run(self, session: Session) -> dict:
        round_input = {
            "summary": session.summary,
            "description": session.description,
            "clarifications": [{"q": q, "a": a} for q, a in session.clarifications],
        }
        # صفاتِ trace (v4: قبل از ساختِ span تعیین می‌شوند) → گروه‌بندیِ session در UI.
        with obs.trace_context(
            session_id=session.session_id,
            trace_name="ticket-classification",
            metadata={
                "config_fingerprint": str(
                    getattr(self.classifier, "fingerprint", {}).get("fingerprint", "")
                ),
                "turn_index": str(session.questions_asked),
            },
        ), obs.span("classification-round", input=round_input) as root:
            output, meta = self.classifier.classify(
                session.summary, session.description, session.clarifications or None
            )
            # متنِ کامل برای راستی‌آزماییِ شواهد (پاسخ‌های کاربر هم منبعِ معتبرِ شاهدند).
            ticket_text = "\n".join(
                [session.summary, session.description, *(a for _q, a in session.clarifications)]
            )
            with obs.span(
                "decision",
                as_type="chain",
                input={
                    "questions_asked": session.questions_asked,
                    "max_questions": settings.max_questions,
                    "knn_votes": meta.get("knn_votes"),
                    "evidence_verification": settings.evidence_verification,
                },
            ) as dec_span:
                decision = decide(
                    output,
                    self.taxonomy,
                    session.questions_asked,
                    settings.max_questions,
                    ticket_text=ticket_text,
                    knn_votes=meta.get("knn_votes"),
                    verify_evidence=settings.evidence_verification,
                )
                dec_span.update(
                    output={
                        "action": decision.action.value,
                        "labels": decision.labels,
                        "ambiguity_reasons": decision.ambiguity_reasons,
                        "verified_evidence": {
                            lid: d.evidence for lid, d in decision.layer_decisions.items()
                        },
                        "question": decision.question,
                        "needs_review": decision.needs_review,
                    }
                )
            self._annotate_trace(root, round_input, decision, meta)
            session.last_trace_id = obs.current_trace_id() or session.last_trace_id

        log.info(
            "session=%s action=%s labels=%s asked=%d",
            session.session_id, decision.action.value, decision.labels, session.questions_asked,
        )

        # ثبت ماندگارِ این دور (تیکت، شواهد، تصمیم، سوال، متادیتای LLM).
        self.interaction_logger.log_round(session, output, decision, meta)

        if decision.action == Action.ASK:
            session.pending_question = decision.question
            session.status = Status.NEED_INFO
            session.result = None
        else:
            session.status = (
                Status.COMPLETED if decision.action == Action.DONE else Status.COMPLETED_LOW_CONF
            )
            session.pending_question = None
            session.result = self._build_result(decision, output)
            self.interaction_logger.log_final(session)  # رکورد نهایی جلسه
            if decision.needs_review:
                self._enqueue_review(session)

        return self._response(session)

    # ---- observability ----
    def _annotate_trace(self, root, round_input: dict, decision, meta: dict) -> None:
        """خروجی روی spanِ ریشه + I/O trace + scoreهای خودکار (خوراکِ داشبورد/هشدارِ drift)."""
        retrieval = meta.get("retrieval") or {}
        explain = meta.get("retrieval_explain") or {}
        retriever_active = getattr(self.classifier, "retriever", None) is not None
        abstained = retriever_active and not retrieval
        round_output = {
            "action": decision.action.value,
            "labels": decision.labels,
            "question": decision.question,
            "needs_review": decision.needs_review,
        }
        # ورودی/خروجیِ trace از spanِ ریشه مشتق می‌شود (set_trace_io در v4 منسوخ است).
        root.update(
            output=round_output,
            metadata={
                "model": meta.get("model"),
                "retrieval_abstain_reason": explain.get("abstain_reason"),
            },
        )
        # scoreهای عددی: روندشان در Langfuse قابلِ‌رسم و هشدار است.
        obs.score_current_trace("asked_clarification", int(decision.action == Action.ASK))
        obs.score_current_trace(
            "needs_review", int(decision.needs_review),
            comment="; ".join(
                f"{lid}: {', '.join(rs)}" for lid, rs in decision.ambiguity_reasons.items()
            ) or None,
        )
        if retriever_active:
            obs.score_current_trace("retrieval_abstained", int(abstained))
        top_sim = retrieval.get("top_similarity", explain.get("top_similarity"))
        if top_sim is not None:
            obs.score_current_trace("retrieval_top_similarity", float(top_sim))
        knn_votes = meta.get("knn_votes") or {}
        if knn_votes:
            agree = [
                int(v.get("label") == decision.labels.get(lid))
                for lid, v in knn_votes.items()
                if v.get("label") and decision.labels.get(lid)
            ]
            if agree:
                obs.score_current_trace("knn_llm_agreement", sum(agree) / len(agree))
        halluc = any(
            "hallucinated_evidence" in rs for rs in decision.ambiguity_reasons.values()
        )
        obs.score_current_trace("hallucinated_evidence", int(halluc))

    def _enqueue_review(self, session: Session) -> None:
        """جلسهٔ needs_review → صفِ بازبینیِ انسانی. خطا فقط لاگ می‌شود."""
        if self.review_store is None or session.result is None:
            return
        try:
            item_id = self.review_store.enqueue(
                source="production",
                session_id=session.session_id,
                trace_id=session.last_trace_id,
                summary=session.summary,
                description=session.description,
                clarifications=[{"q": q, "a": a} for q, a in session.clarifications],
                predicted_labels=session.result.get("labels") or {},
                ambiguity_reasons=session.result.get("ambiguity_reasons") or {},
                evidence=session.result.get("evidence") or {},
                model_reasoning=session.result.get("reasoning") or "",
            )
            if item_id is not None:
                log.info("session=%s به صفِ بازبینی افزوده شد (item=%s)", session.session_id, item_id)
        except Exception as e:
            log.warning("افزودن به صفِ بازبینی ناموفق بود: %s", e)

    def _build_result(self, decision, output) -> dict:
        return {
            "labels": decision.labels,                    # {layer_id: label_id}
            "evidence": {lid: d.evidence for lid, d in decision.layer_decisions.items()},
            "suggested_summary": output.suggested_summary,
            "needs_review": decision.needs_review,
            "ambiguity_reasons": decision.ambiguity_reasons,  # چرا مبهم/بازبینی (تحلیل‌پذیر)
            "reasoning": output.reasoning,
        }

    def _response(self, session: Session) -> dict:
        return {
            "session_id": session.session_id,
            "status": session.status.value if session.status else None,
            "question": session.pending_question,
            "questions_asked": session.questions_asked,
            "result": session.result,
        }
