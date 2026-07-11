"""بارگذاری پیکربندی از .env و مسیرهای پروژه."""
from __future__ import annotations
.01010101
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    # --- DeepSeek ---
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    reasoner_model: str = os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")

    # --- Behavior ---
    max_questions: int = int(os.getenv("MAX_QUESTIONS", "2"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))

    # --- Robustness ---
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "60"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))

    # --- Ambiguity detection ---
    enable_self_consistency: bool = _get_bool("ENABLE_SELF_CONSISTENCY", False)
    self_consistency_samples: int = int(os.getenv("SELF_CONSISTENCY_SAMPLES", "3"))

    # --- Interaction logging (برای تحلیل دقت و ساخت Gold Set) ---
    interaction_log_enabled: bool = _get_bool("INTERACTION_LOG_ENABLED", True)
    interaction_log_path: Path = PROJECT_ROOT / os.getenv(
        "INTERACTION_LOG_PATH", "logs/interactions.jsonl"
    )

    # --- Ticket submissions (خروجی نهایی؛ حاوی PII، داخل logs/ بماند) ---
    tickets_log_path: Path = PROJECT_ROOT / os.getenv(
        "TICKETS_LOG_PATH", "logs/tickets.jsonl"
    )

    # --- Paths ---
    taxonomy_path: Path = PROJECT_ROOT / "config" / "taxonomy.yaml"
    examples_path: Path = PROJECT_ROOT / "data" / "examples.jsonl"

    def require_api_key(self) -> None:
        if not self.deepseek_api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY تنظیم نشده است. فایل .env.example را به .env کپی کرده "
                "و کلید خود را وارد کنید."
            )


settings = Settings()
