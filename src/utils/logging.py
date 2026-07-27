"""لاگ ساختاریافته — json (production) یا text (توسعه)، طبقِ LOG_FORMAT/LOG_LEVEL.

خروجی به stdout می‌رود و چرخش/نگه‌داریِ فایل به میزبان (journald/Docker) سپرده
می‌شود؛ همین «rotation-friendly» است. UTF-8: ensure_ascii=False تا فارسی خوانا بماند.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    """هر خط یک شیء JSON: ts (UTC ISO-8601)، level، logger، msg، و exc در صورت خطا."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def _configure() -> None:
    from config.settings import settings

    stream = sys.stdout
    try:  # کنسول ویندوز ممکن است cp1252 باشد؛ متن فارسی نباید خطا بدهد
        stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    handler = logging.StreamHandler(stream)
    if settings.log_format.lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
    root = logging.getLogger("chatbot")
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    root.propagate = False


def get_logger(name: str = "chatbot") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        _configure()
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("chatbot") else f"chatbot.{name}")
