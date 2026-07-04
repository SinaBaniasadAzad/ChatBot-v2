"""اجرای رابطِ اصلیِ وب (FastAPI + SPA) به‌صورتِ محلی از داخلِ PyCharm.

روی همین فایل راست‌کلیک → Run 'run_web'، سپس در مرورگر باز کنید:
    http://127.0.0.1:8000/
مستنداتِ API:  http://127.0.0.1:8000/docs

این همان رابطی است که روی سرورِ شرکت مستقر می‌شود (پوشهٔ web/). برای تستِ سریعِ
Gradio از app_gradio_windows.py استفاده کنید.
"""
from __future__ import annotations

import os
import sys

PROJECT = os.path.dirname(os.path.abspath(__file__))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(PROJECT, ".env"))
except Exception:
    pass

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="127.0.0.1", port=8000, reload=False)
