"""وضعیت یک مکالمه/جلسه — درون‌حافظه‌ای با TTL (الزام: یک workerِ uvicorn).

جلسه‌ها کوتاه‌عمرند (چند دقیقه)؛ manager جلسه‌های بی‌حرکت را پس از
SESSION_TTL_MINUTES پاک می‌کند. اگر روزی چند پروسه لازم شد، این لایه با Redis
جایگزین می‌شود بدون تغییرِ رابط.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    NEED_INFO = "need_info"            # منتظر پاسخ سوال تکمیلی
    COMPLETED = "completed"            # با اطمینان دسته‌بندی شد
    COMPLETED_LOW_CONF = "completed_low_confidence"  # حدس زده شد، نیازمند بازبینی


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
    touched_at: float = field(default_factory=time.monotonic)  # برای evictionِ TTL

    def touch(self) -> None:
        self.touched_at = time.monotonic()

    def add_clarification(self, question: str, answer: str) -> None:
        self.clarifications.append((question, answer))
        self.questions_asked += 1
        self.pending_question = None
