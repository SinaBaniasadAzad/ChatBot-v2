"""State of a single conversation/session. In production this could live in Redis."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    NEED_INFO = "need_info"            # awaiting a follow-up answer
    COMPLETED = "completed"            # classified confidently
    COMPLETED_LOW_CONF = "completed_low_confidence"  # guessed, needs review


@dataclass
class Session:
    summary: str
    description: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    clarifications: list[tuple[str, str]] = field(default_factory=list)
    questions_asked: int = 0
    status: Status | None = None
    pending_question: str | None = None
    result: dict | None = None

    def add_clarification(self, question: str, answer: str) -> None:
        self.clarifications.append((question, answer))
        self.questions_asked += 1
        self.pending_question = None
