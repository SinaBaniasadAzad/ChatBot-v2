"""
Conversation orchestrator — the brain of the system.

Flow: classify -> decide -> (finish | one question | fallback). The question
budget is enforced here, not in the model. Each session's state is kept by
session_id so a follow-up answer attaches to the same context.
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
        self._sessions: dict[str, Session] = {}  # production: Redis/DB

    # ---- Public API ----
    def start(self, summary: str, description: str) -> dict:
        session = Session(summary=summary, description=description)
        self._sessions[session.session_id] = session
        return self._run(session)

    def answer(self, session_id: str, user_answer: str) -> dict:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("session_id is invalid or has expired.")
        if session.pending_question is None:
            return self._response(session)  # nothing was asked
        session.add_clarification(session.pending_question, user_answer)
        return self._run(session)

    # ---- Internal logic ----
    def _run(self, session: Session) -> dict:
        output, meta = self.classifier.classify(
            session.summary, session.description, session.clarifications or None
        )
        decision = decide(
            output, self.taxonomy, session.questions_asked, settings.max_questions
        )
        log.info(
            "session=%s action=%s labels=%s asked=%d",
            session.session_id, decision.action.value, decision.labels, session.questions_asked,
        )

        # Persist this round (ticket, evidence, decision, question, LLM metadata).
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
            self.interaction_logger.log_final(session)  # final session record

        return self._response(session)

    def _build_result(self, decision, output) -> dict:
        return {
            "labels": decision.labels,                    # {layer_id: label_id}
            "evidence": {lid: d.evidence for lid, d in decision.layer_decisions.items()},
            "suggested_summary": output.suggested_summary,
            "needs_review": decision.needs_review,
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
