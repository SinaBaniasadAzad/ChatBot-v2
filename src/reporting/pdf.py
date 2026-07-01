"""
تبدیلِ HTML به PDF — با **زنجیرهٔ چند‌موتوره** تا در هر محیطی چیزی کار کند.

ترتیبِ تلاش (خودکار):
  ۱) Playwright + Chromium   → بهترین فیدلیتی؛ PDF یک صفحهٔ بلندِ پیوسته و تیره (مثلِ HTML).
  ۲) مرورگرِ سیستمی (chrome/chromium) با --headless --print-to-pdf → بدونِ نصبِ مرورگرِ Playwright.
  ۳) WeasyPrint              → **بدونِ هیچ مرورگری**، فقط pip؛ خروجیِ تیره و صفحه‌بندی‌شدهٔ A4.

اگر هیچ‌کدام نبود، `PdfEngineUnavailable` بالا می‌آید تا فراخواننده بتواند به یک خروجیِ
جایگزین (مثلاً PDFِ داشبوردِ matplotlib) برگردد.

نصبِ ساده روی Kaggle (بدونِ مرورگر):  pip install weasyprint
"""
from __future__ import annotations

import glob
import shutil
import subprocess
from pathlib import Path


class PdfEngineUnavailable(RuntimeError):
    """هیچ موتورِ تولیدِ PDF در دسترس نبود."""


# مسیرها/نام‌های رایجِ مرورگر
_CHROME_GLOBS = (
    "/opt/pw-browsers/chromium-*/chrome-linux/chrome",
    "/opt/pw-browsers/chromium-*/chrome-linux/headless_shell",
)
_CHROME_NAMES = ("google-chrome-stable", "google-chrome", "chromium", "chromium-browser", "chrome")

# اصلاحِ CSS مخصوصِ WeasyPrint: عنوانِ گرادیانی را قابل‌دیدن کن و کلِ صفحه را تیره کن.
_WEASY_CSS = """
.h-grad{ color:#8ab4ff !important; background:none !important; }
@page{ size:A4; margin:12mm 9mm; background:#0b1020; }
body{ padding:0 !important; }
"""


def _find_chrome() -> str | None:
    for pat in _CHROME_GLOBS:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    for name in _CHROME_NAMES:
        p = shutil.which(name)
        if p:
            return p
    return None


def _via_playwright(html_path: Path, pdf_path: Path, width_px: int) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise PdfEngineUnavailable("playwright نصب نیست") from e
    with sync_playwright() as p:
        exe = _find_chrome()
        kwargs: dict = {"args": ["--no-sandbox"]}
        if exe:
            kwargs["executable_path"] = exe
        try:
            browser = p.chromium.launch(**kwargs)
        except Exception as e:
            raise PdfEngineUnavailable(f"مرورگرِ Playwright در دسترس نیست: {e}") from e
        try:
            page = browser.new_page(viewport={"width": width_px, "height": 1200})
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            page.emulate_media(media="screen")  # حفظِ تمِ تیره
            height = int(page.evaluate("Math.ceil(document.body.scrollHeight)"))
            page.pdf(path=str(pdf_path), width=f"{width_px}px", height=f"{height + 24}px",
                     print_background=True, margin={"top": "0", "right": "0", "bottom": "0", "left": "0"})
        finally:
            browser.close()


def _via_chrome_cli(html_path: Path, pdf_path: Path, width_px: int) -> None:
    exe = _find_chrome()
    if not exe:
        raise PdfEngineUnavailable("مرورگرِ سیستمی (chrome/chromium) پیدا نشد")
    cmd = [
        exe, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri(),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=120)
    except Exception as e:
        raise PdfEngineUnavailable(f"اجرای مرورگرِ سیستمی ناموفق: {e}") from e
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        # نسخه‌های قدیمی‌تر --headless=new را نمی‌شناسند
        cmd[1] = "--headless"
        subprocess.run(cmd, capture_output=True, timeout=120)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise PdfEngineUnavailable(f"مرورگرِ سیستمی PDF نساخت (rc={r.returncode})")


def _via_weasyprint(html_path: Path, pdf_path: Path, width_px: int) -> None:
    try:
        from weasyprint import CSS, HTML
    except (ImportError, OSError) as e:  # OSError = کتابخانه‌های سیستمیِ pango/cairo نبود
        raise PdfEngineUnavailable(f"WeasyPrint در دسترس نیست: {e}") from e
    HTML(filename=str(html_path)).write_pdf(str(pdf_path), stylesheets=[CSS(string=_WEASY_CSS)])


_ENGINES = {
    "playwright": _via_playwright,
    "chrome": _via_chrome_cli,
    "weasyprint": _via_weasyprint,
}
_DEFAULT_ORDER = ("playwright", "chrome", "weasyprint")


def html_to_pdf(html_path: str | Path, pdf_path: str | Path, *,
                width_px: int = 1240, engine: str = "auto") -> Path:
    """
    HTML → PDF با انتخابِ خودکارِ موتور. `engine` می‌تواند
    "auto" (پیش‌فرض) یا یکی از "playwright" / "chrome" / "weasyprint" باشد.
    خروجی: مسیرِ PDF. اگر هیچ موتوری نبود: `PdfEngineUnavailable`.
    """
    html_path = Path(html_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    order = _DEFAULT_ORDER if engine == "auto" else (engine,)
    problems = []
    for name in order:
        fn = _ENGINES.get(name)
        if fn is None:
            raise ValueError(f"موتورِ ناشناخته: {engine}")
        try:
            fn(html_path, pdf_path, width_px)
            return pdf_path
        except PdfEngineUnavailable as e:
            problems.append(f"{name}: {e}")
            continue
    raise PdfEngineUnavailable(
        "هیچ موتورِ PDF در دسترس نبود. یکی را نصب کن:\n"
        "    pip install weasyprint            # بدونِ مرورگر (پیشنهادی روی Kaggle)\n"
        "    pip install playwright && playwright install chromium\n"
        "جزئیات → " + " | ".join(problems)
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="HTML → PDF (چند‌موتوره).")
    ap.add_argument("html")
    ap.add_argument("pdf")
    ap.add_argument("--engine", default="auto", choices=["auto", "playwright", "chrome", "weasyprint"])
    ap.add_argument("--width", type=int, default=1240)
    a = ap.parse_args()
    print("saved:", html_to_pdf(a.html, a.pdf, width_px=a.width, engine=a.engine))
