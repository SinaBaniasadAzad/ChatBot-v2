"""لانچرِ ویندوز/PyCharm برای رابطِ Gradioِ Ticket Assistant.

تجربهٔ کاربری کاملاً همان `app_gradio.py` است (شناسایی کاربر → جستجوی FAQ →
دسته‌بندی → ثبت با شمارهٔ پیگیری). این فایل فقط یک نقطهٔ‌ورودِ جداست تا Run
Configurationِ فعلیِ شما در PyCharm دست‌نخورده کار کند.

★ منطقِ رابط کاربری فقط در `app_gradio.py` است (single source of truth)؛ برای
تغییرِ UI همان فایل را ویرایش کنید، نه این یکی را.

نحوهٔ اجرا در PyCharm:
  ۱) Interpreter پروژه را روی .venv تنظیم کنید (نیازمند gradio, openai, …).
  ۲) کلیدِ DEEPSEEK_API_KEY را در فایلِ .env در ریشهٔ پروژه بگذارید.
  ۳) روی همین فایل راست‌کلیک → Run 'app_gradio_windows'. مرورگر خودکار باز می‌شود.
"""
from __future__ import annotations

import os
import sys

# مسیرِ پروژه (Windows-safe) قبل از importِ ماژولِ مشترک.
PROJECT = os.path.dirname(os.path.abspath(__file__))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

# app_gradio خودش .env را بارگذاری می‌کند، بک‌اند را می‌سازد و `demo` را آماده می‌کند.
from app_gradio import ASSETS, demo  # noqa: E402

if __name__ == "__main__":
    # اجرای محلی: بدونِ tunnelِ عمومی، مرورگر خودکار باز می‌شود.
    demo.queue().launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
        show_error=True,
        allowed_paths=[ASSETS],
    )
