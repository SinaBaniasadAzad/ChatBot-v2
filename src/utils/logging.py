"""لاگ ساختاریافتهٔ ساده (JSON-friendly). در تولید می‌توان به فایل/سرویس هدایت کرد."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "chatbot") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root = logging.getLogger("chatbot")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("chatbot") else f"chatbot.{name}")
