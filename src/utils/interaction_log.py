"""
Persist interactions in JSONL format (one event per line).

Why JSONL? Append-only, robust, and easy to analyze with pandas/jq.
What is stored? The ticket, follow-up questions + user answers, the raw model
output (candidates + evidence), the final decision, and LLM metadata (model,
latency, token usage). This data is the foundation for building a Gold Set,
error analysis, and prompt tuning.

Security note: tickets contain personal data (names, employee IDs); these logs
hold PII — the logs/ folder is in .gitignore and needs a retention/access policy.
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
        except OSError as e:  # logging must not break the main flow
            log.warning("Failed to write interaction log: %s", e)

    @staticmethod
    def _clarifications(session) -> list[dict]:
        return [{"q": q, "a": a} for q, a in session.clarifications]

    def log_round(self, session, output, decision, meta: dict) -> None:
        """One classify->decide round (including the question about to be asked)."""
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
        """Final session record — the most useful artifact for accuracy analysis."""
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
