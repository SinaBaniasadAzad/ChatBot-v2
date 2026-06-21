"""
ثبت ماندگارِ تعاملات به فرمت JSONL (هر خط یک رویداد).

چرا JSONL؟ append-only، مقاوم، و تحلیلش با pandas/jq ساده است.
چه چیزی ذخیره می‌شود؟ تیکت، سوال‌های تکمیلی + پاسخ کاربر، خروجی خام مدل (کاندیدا +
شواهد)، تصمیم نهایی، و متادیتای LLM (مدل، latency، مصرف توکن). این داده‌ها سرمایهٔ
ساخت Gold Set، تحلیل خطا، و تیون prompt هستند.

نکتهٔ امنیتی: تیکت‌ها حاوی دادهٔ پرسنلی (نام، کد پرسنلی) هستند؛ این لاگ‌ها PII دارند —
پوشهٔ logs/ در .gitignore است و باید سیاست نگه‌داری/دسترسی برایش تعریف شود.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from src.utils.logging import get_logger

log = get_logger("interaction_log")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class InteractionLogger:
    def __init__(self, enabled: bool, path: Path) -> None:
        self.enabled = enabled
        self.path = Path(path)
        self._lock = threading.Lock()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict) -> None:
        if not self.enabled:
            return
        line = json.dumps(record, ensure_ascii=False)
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:  # لاگ نباید جریان اصلی را بشکند
            log.warning("نوشتن لاگ تعامل ناموفق بود: %s", e)

    @staticmethod
    def _clarifications(session) -> list[dict]:
        return [{"q": q, "a": a} for q, a in session.clarifications]

    def log_round(self, session, output, decision, meta: dict) -> None:
        """یک دورِ classify->decide (شامل سوالی که قرار است پرسیده شود)."""
        self._write(
            {
                "ts": _now(),
                "event": "round",
                "session_id": session.session_id,
                "turn_index": session.questions_asked,
                "input": {
                    "summary": session.summary,
                    "description": session.description,
                    "clarifications": self._clarifications(session),
                },
                "model_output": output.model_dump(),
                "decision": {
                    "action": decision.action.value,
                    "labels": decision.labels,
                    "evidence": {lid: d.evidence for lid, d in decision.layer_decisions.items()},
                    "question": decision.question,
                    "needs_review": decision.needs_review,
                },
                "llm": meta,
            }
        )

    def log_final(self, session) -> None:
        """رکورد نهاییِ جلسه — مفیدترین آرتیفکت برای تحلیل دقت."""
        self._write(
            {
                "ts": _now(),
                "event": "session_final",
                "session_id": session.session_id,
                "status": session.status.value if session.status else None,
                "questions_asked": session.questions_asked,
                "input": {
                    "summary": session.summary,
                    "description": session.description,
                    "clarifications": self._clarifications(session),
                },
                "result": session.result,
            }
        )
