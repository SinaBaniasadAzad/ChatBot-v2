"""
ثبتِ ماندگارِ تعاملات در جدولِ interactions (SQLite).

شکلِ رکورد (payload) دقیقاً همان JSONLِ قبلی است؛ ابزارهای تحلیل (cost_report و
pandas/jq) با scripts/export_interactions.py همان فایلِ JSONL را می‌گیرند.
چه چیزی ذخیره می‌شود؟ تیکت، سوال‌های تکمیلی + پاسخ کاربر، خروجی خام مدل (کاندیدا +
شواهد)، تصمیم نهایی، و متادیتای LLM (مدل، latency، مصرف توکن).

نکتهٔ امنیتی: این داده‌ها PII دارند (متنِ تیکت)؛ سیاستِ نگه‌داری در
src/db/maintenance.py اعمال می‌شود (پیش‌فرض: حذف پس از ۹۰ روز).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src.db.database import Database
from src.utils.logging import get_logger

log = get_logger("interaction_log")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class InteractionLogger:
    def __init__(self, enabled: bool, db: Database) -> None:
        self.enabled = enabled
        self.db = db

    def _write(self, record: dict) -> None:
        if not self.enabled:
            return
        try:
            self.db.conn.execute(
                "INSERT INTO interactions(ts, event, session_id, payload) VALUES (?,?,?,?)",
                (
                    record["ts"],
                    record["event"],
                    record.get("session_id", ""),
                    json.dumps(record, ensure_ascii=False),
                ),
            )
        except Exception as e:  # لاگ نباید جریان اصلی را بشکند
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
