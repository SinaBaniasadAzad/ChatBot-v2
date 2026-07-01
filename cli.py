"""
CLI تعاملی برای تست دستی هسته (بدون نیاز به API).

اجرا:  python cli.py
سپس summary و description را وارد کنید؛ اگر چت‌بات سوال تکمیلی داشت، پاسخ دهید
"""
from __future__ import annotations

import sys

# روی کنسول ویندوز، خروجی/ورودی را UTF-8 کن تا متن فارسی درست نمایش داده شود.
for _stream in (sys.stdout, sys.stdin, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from src.conversation.manager import ConversationManager


def _print_result(resp: dict) -> None:
    print("\n" + "=" * 60)
    print(f"وضعیت: {resp['status']}   |   سوال‌های پرسیده‌شده: {resp['questions_asked']}")
    result = resp["result"]
    if result:
        print("برچسب‌ها:")
        for layer_id, label in result["labels"].items():
            ev = "، ".join(result["evidence"].get(layer_id, [])) or "—"
            print(f"   • {layer_id}: {label}   (شواهد: {ev})")
        print(f"خلاصهٔ پیشنهادی: {result['suggested_summary']}")
        if result["needs_review"]:
            print("⚠️  اطمینان پایین — نیازمند بازبینی انسانی.")
    print("=" * 60 + "\n")


def main() -> None:
    print("چت‌بات مسیریابی تیکت — برای خروج Ctrl+C\n")
    manager = ConversationManager()

    while True:
        try:
            summary = input("Summary> ").strip()
            description = input("Description> ").strip()
            if not summary and not description:
                continue

            resp = manager.start(summary, description)

            # حلقهٔ سوال‌های تکمیلی
            while resp["status"] == "need_info":
                print(f"\n🤖 سوال: {resp['question']}")
                ans = input("پاسخ شما> ").strip()
                resp = manager.answer(resp["session_id"], ans)

            _print_result(resp)
        except (KeyboardInterrupt, EOFError):
            print("\nخداحافظ!")
            sys.exit(0)


if __name__ == "__main__":
    main()
