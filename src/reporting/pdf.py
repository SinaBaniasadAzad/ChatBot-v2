"""
تبدیلِ HTML به PDF با موتورِ مرورگر (Chromium/Playwright).

چرا مرورگر و نه کتابخانه‌های سبک؟ گزارش‌های ما از CSSِ مدرن استفاده می‌کنند
(gradient روی متن، color-mix، grid/flex، radial-gradient). فقط یک موتورِ واقعیِ
مرورگر این‌ها را **دقیقاً** مثلِ همان چیزی که در صفحه می‌بینی رندر می‌کند.

خروجی **عمداً تیره** است (مثلِ خودِ HTML): با `emulate_media("screen")` استایلِ چاپ
اعمال نمی‌شود و با `print_background=True` پس‌زمینه حفظ می‌شود. PDF یک صفحهٔ بلندِ
پیوسته است تا دقیقاً مثلِ اسکرین‌شات باشد (بدونِ برشِ صفحه‌ایِ کارت‌ها).

نیازمندی (یک‌بار):  pip install playwright  &&  playwright install chromium
روی Kaggle مرورگر معمولاً از پیش هست؛ اگر نبود همان دو دستور را اجرا کن.
"""
from __future__ import annotations

import glob
from pathlib import Path

# مسیرهای رایجِ مرورگرِ از پیش‌نصب (این محیط/CI). اگر خالی بود، لانچِ پیش‌فرض امتحان می‌شود.
_CHROME_GLOBS = (
    "/opt/pw-browsers/chromium-*/chrome-linux/chrome",
    "/opt/pw-browsers/chromium-*/chrome-linux/headless_shell",
    "/opt/pw-browsers/chromium_headless_shell-*/chrome-linux/headless_shell",
)


def _find_chrome() -> str | None:
    for pat in _CHROME_GLOBS:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def html_to_pdf(html_path: str | Path, pdf_path: str | Path, *, width_px: int = 1240) -> Path:
    """یک فایلِ HTML را به PDFِ تیره و تک‌صفحهٔ بلند تبدیل می‌کند و مسیر را برمی‌گرداند."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:  # راهنماییِ شفاف به‌جای خطای مبهم
        raise RuntimeError(
            "برای خروجیِ PDF به Playwright نیاز است. یک‌بار اجرا کن:\n"
            "    pip install playwright && playwright install chromium"
        ) from e

    html_path = Path(html_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        exe = _find_chrome()
        kwargs: dict = {"args": ["--no-sandbox"]}
        if exe:
            kwargs["executable_path"] = exe
        try:
            browser = p.chromium.launch(**kwargs)
        except Exception as e:
            raise RuntimeError(
                "مرورگرِ Chromium برای Playwright پیدا نشد. یک‌بار اجرا کن:\n"
                "    playwright install chromium"
            ) from e
        try:
            page = browser.new_page(viewport={"width": width_px, "height": 1200})
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            page.emulate_media(media="screen")  # حفظِ تمِ تیره (نه CSSِ چاپ)
            height_px = page.evaluate("Math.ceil(document.body.scrollHeight)")
            page.pdf(
                path=str(pdf_path),
                width=f"{width_px}px",
                height=f"{int(height_px) + 24}px",  # یک صفحهٔ پیوسته
                print_background=True,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            )
        finally:
            browser.close()
    return pdf_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="HTML → PDF (تیره، مثلِ همان HTML).")
    ap.add_argument("html")
    ap.add_argument("pdf")
    ap.add_argument("--width", type=int, default=1240)
    a = ap.parse_args()
    out = html_to_pdf(a.html, a.pdf, width_px=a.width)
    print(f"saved: {out}")
