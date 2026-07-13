"""بارگذاری پیکربندی از .env و مسیرهای پروژه."""
from __future__ import annotations

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
    # پس از این تعداد خطای پیاپیِ LLM، مدارشکن باز می‌شود و به‌جای انتظارِ طولانی،
    # بلافاصله «سرویسِ دسته‌بندی در دسترس نیست» برمی‌گردد (ثبتِ دستیِ تیکت ممکن می‌ماند).
    cb_failure_threshold: int = int(os.getenv("CB_FAILURE_THRESHOLD", "3"))
    cb_cooldown_seconds: float = float(os.getenv("CB_COOLDOWN_SECONDS", "30"))
    # استخرِ اتصالِ HTTP به DeepSeek — با فرضِ بارِ اوج ~۱۰ فراخوانِ همزمان، ۱۶ کافی است.
    llm_max_connections: int = int(os.getenv("LLM_MAX_CONNECTIONS", "16"))

    # --- Sessions (درون‌حافظه‌ای؛ الزام: یک workerِ uvicorn) ---
    session_ttl_minutes: float = float(os.getenv("SESSION_TTL_MINUTES", "60"))

    # --- HTTP (SPA هم‌مبدأ است؛ CORS فقط اگر مبدأ جدا لازم شد) ---
    cors_origins: list[str] = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()
    ]
    # توکنِ اندپوینت‌های ادمین (جستجو/بازیابیِ تیکت). خالی = اندپوینت‌ها غیرفعال.
    admin_api_token: str = os.getenv("ADMIN_API_TOKEN", "")

    # --- Logging ---
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "text")  # production: json (در ایمیج تنظیم شده)

    # --- Ambiguity detection ---
    enable_self_consistency: bool = _get_bool("ENABLE_SELF_CONSISTENCY", False)
    self_consistency_samples: int = int(os.getenv("SELF_CONSISTENCY_SAMPLES", "3"))

    # --- Retrieval-augmented classification (precedent few-shot + kNN gating) ---
    # اگر ایندکس/وابستگی‌ها موجود نباشند، سیستم خودکار بدونِ retrieval کار می‌کند.
    retrieval_enabled: bool = _get_bool("RETRIEVAL_ENABLED", True)
    retrieval_index_path: Path = PROJECT_ROOT / os.getenv(
        "RETRIEVAL_INDEX_PATH", "data/retrieval/index.npz"
    )
    retrieval_pool_path: Path = PROJECT_ROOT / os.getenv(
        "RETRIEVAL_POOL_PATH", "data/retrieval/tickets_clean.jsonl"
    )
    retrieval_k_demos: int = int(os.getenv("RETRIEVAL_K_DEMOS", "6"))
    retrieval_purity_k: int = int(os.getenv("RETRIEVAL_PURITY_K", "15"))
    # کفِ شباهت: پایین‌تر از این، retrieval کلاً کنار می‌کشد (تیکتِ بی‌سابقه)
    retrieval_sim_floor: float = float(os.getenv("RETRIEVAL_SIM_FLOOR", "0.40"))
    # اگر همسایگیِ خالص (purity ≥ این آستانه) با برچسبِ LLM مخالف بود → ابهام → سوال
    knn_disagree_purity: float = float(os.getenv("KNN_DISAGREE_PURITY", "0.80"))
    # راستی‌آزماییِ شواهد: شاهدی که در متنِ تیکت نباشد، شاهد حساب نمی‌شود
    evidence_verification: bool = _get_bool("EVIDENCE_VERIFICATION", True)

    # --- Interaction logging (جدول interactions در DB؛ برای تحلیل دقت و Gold Set) ---
    interaction_log_enabled: bool = _get_bool("INTERACTION_LOG_ENABLED", True)

    # --- Datastore (SQLite/WAL: تیکت‌ها + لاگِ تعاملات؛ حاوی PII) ---
    db_path: Path = PROJECT_ROOT / os.getenv("APP_DB_PATH", "data/app.db")

    # --- Data retention (اجرای روزانه داخلِ اپ + CLI: python -m scripts.db_maintenance) ---
    retention_enabled: bool = _get_bool("RETENTION_ENABLED", True)
    # حذفِ کاملِ ردیف‌های interactions قدیمی‌تر از این (روز). 0 = هرگز.
    interaction_retention_days: int = int(os.getenv("INTERACTION_RETENTION_DAYS", "90"))
    # ناشناس‌سازیِ هویتِ تیکت‌ها (نام/کد پرسنلی) پس از این مدت (روز). 0 = هرگز.
    ticket_anonymize_days: int = int(os.getenv("TICKET_ANONYMIZE_DAYS", "365"))
    # حذفِ کاملِ تیکت‌های قدیمی (روز). 0 = هرگز (پیش‌فرض: نگه می‌داریم، فقط ناشناس می‌شوند).
    ticket_delete_days: int = int(os.getenv("TICKET_DELETE_DAYS", "0"))

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
