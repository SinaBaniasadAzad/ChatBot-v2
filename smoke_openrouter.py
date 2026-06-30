"""
اسموک‌تستِ اولیه روی OpenRouter — یک تیکت می‌فرستد و دستهٔ تشخیص‌داده‌شده را چاپ می‌کند.

هدف: تأیید این‌که (۱) کلید OpenRouter، (۲) مدل دیپ‌سیک، و (۳) مسیر دسته‌بندیِ JSON
درست کار می‌کنند. این اسکریپت همان هستهٔ واقعی پروژه (Classifier) را صدا می‌زند؛
یعنی دقیقاً یک «دور» کامل دسته‌بندی — بدون حلقهٔ سوال تکمیلی.

پیش‌نیاز (داخل فایل .env):
    DEEPSEEK_API_KEY=sk-or-v1-...                  # کلید OpenRouter
    DEEPSEEK_BASE_URL=https://openrouter.ai/api/v1
    DEEPSEEK_MODEL=deepseek/deepseek-chat-v3-0324  # یا نسخهٔ رایگان: ...:free

اجرا:
    python -m scripts.smoke_openrouter
    python -m scripts.smoke_openrouter "خلاصهٔ تیکت" "شرح کامل مشکل"
"""
from __future__ import annotations

import sys
from pathlib import Path

# اجازهٔ اجرا از ریشهٔ پروژه (مثل scripts/evaluate.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from src.classifier.classifier import Classifier  # noqa: E402

# تیکت نمونه (برگرفته از داده‌های واقعی data/examples.jsonl)
SAMPLE_SUMMARY = "مشکل ثبت ورود و خروج"
SAMPLE_DESCRIPTION = (
    "با سلام، طبق پیوست پانچ ورود و خروج برای تاریخ ۱۹ مرداد ماه در سامانهٔ ERP "
    "به درستی ثبت نشده است. لطفاً بررسی نمایید."
)


def main() -> None:
    # روی کنسول ویندوز، خروجی را UTF-8 کن تا متن فارسی درست نمایش داده شود.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    argv = sys.argv[1:]
    summary = argv[0] if len(argv) >= 1 else SAMPLE_SUMMARY
    description = argv[1] if len(argv) >= 2 else SAMPLE_DESCRIPTION

    settings.require_api_key()  # اگر کلید نباشد، پیام راهنمای فارسی می‌دهد

    print("=" * 60)
    print("اسموک‌تستِ OpenRouter / DeepSeek")
    print("=" * 60)
    print(f"Endpoint : {settings.deepseek_base_url}")
    print(f"Model    : {settings.model}")
    print(f"Ticket   : {summary} | {description}")
    print("-" * 60)

    clf = Classifier()  # prompt + few-shot یک‌بار ساخته می‌شود
    output, meta = clf.classify(summary, description)

    print("نتیجهٔ دسته‌بندی (یک دور):")
    for layer in clf.taxonomy.layers:
        lo = output.layers.get(layer.id)
        top = lo.top if lo else None
        label = top.label if top else "—"
        evidence = "، ".join(top.evidence) if (top and top.evidence) else "—"
        flag = "  ⚠️ مبهم (نیازمند سوال تکمیلی)" if (lo and lo.needs_clarification) else ""
        print(f"  • {layer.id} ({layer.name}): {label}   [شواهد: {evidence}]{flag}")

    print(f"\nخلاصهٔ پیشنهادی: {output.suggested_summary}")
    print(
        f"متادیتای LLM → model={meta.get('model')}  "
        f"latency={meta.get('latency_ms')}ms  usage={meta.get('usage')}"
    )
    print("\n✅ تماس با OpenRouter موفق بود و خروجی JSON معتبر دریافت شد.")


if __name__ == "__main__":
    main()
