"""
Orchestrator مکالمه — مغز سیستم.

جریان: classify -> decide -> (پایان | یک سوال | fallback). سقف سوال‌ها اینجا اعمال
می‌شود، نه در مدل. وضعیت هر جلسه با session_id نگه‌داری می‌شود تا پاسخِ سوال تکمیلی
به همان context بچسبد.
"""
from __future__ import annotations

from config.settings import settings
from src.classifier.classifier import Classifier
from src.classifier.decision import Action, decide
from src.conversation.state import Session, Status
from src.utils.interaction_log import InteractionLogger
from src.utils.logging import get_logger

log = get_logger("conversation")


class ConversationManager:
    def __init__(
        self,
        classifier: Classifier | None = None,
        interaction_logger: InteractionLogger | None = None,
    ) -> None:
        self.classifier = classifier or Classifier()
        self.taxonomy = self.classifier.taxonomy
        self.interaction_logger = interaction_logger or InteractionLogger(
            settings.interaction_log_enabled, settings.interaction_log_path
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
        output, meta = self.classifier.classify(
            session.summary, session.description, session.clarifications or None
        )
        # متنِ کامل برای راستی‌آزماییِ شواهد (پاسخ‌های کاربر هم منبعِ معتبرِ شاهدند).
        ticket_text = "\n".join(
            [session.summary, session.description, *(a for _q, a in session.clarifications)]
        )
        decision = decide(
            output,
            self.taxonomy,
            session.questions_asked,
            settings.max_questions,
            ticket_text=ticket_text,
            knn_votes=meta.get("knn_votes"),
            verify_evidence=settings.evidence_verification,
        )
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

        return self._response(session)

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
