"""
رابط گرافیکیِ چت‌بات مسیریابی تیکت — نسخهٔ Gradio (برای اجرا روی Kaggle).

این فقط یک «نما» روی همان ConversationManager موجود است؛ هیچ منطقی در بک‌اند عوض نمی‌شود.

ویژگی‌ها:
  • گفت‌وگوی چت‌مانند با حلقهٔ سوال تکمیلی (تا سقف MAX_QUESTIONS).
  • نمایش دستهٔ نهایی (نوع + حوزه) به‌صورت نشان‌های رنگی.
  • هشدارِ «نیاز به بازبینی انسانی» + دلیلِ کوتاه (کدام لایه مبهم ماند).
  • چیدمان راست‌به‌چپ (RTL) و فونت فارسی (Vazirmatn).
  • سوییچِ نمایش JSON خام برای دیباگ.

اجرا روی Kaggle:
  ۱) این فایل را در ریشهٔ پروژه بگذار:  /kaggle/working/Test-ChatBot/app_gradio.py
  ۲) اینترنتِ نوت‌بوک را روشن کن و کلید را در Secrets بگذار (نام: OPENROUTER_API_KEY).
  ۳) سلول نصب:   !pip -q install -U gradio openai pydantic PyYAML python-dotenv
  ۴) سلول اجرا:  %run /kaggle/working/Test-ChatBot/app_gradio.py
  لینکِ عمومیِ Gradio (share) در خروجی چاپ می‌شود.

  مدل را می‌توانی پایین (یا با متغیر محیطی DEEPSEEK_MODEL) عوض کنی.
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# ۱) پیکربندی — باید قبل از import پروژه انجام شود (settings فقط یک‌بار خوانده می‌شود)
# ---------------------------------------------------------------------------
# کلید OpenRouter: اول از Kaggle Secrets، بعد از متغیر محیطی.
_KEY = ""
try:
    from kaggle_secrets import UserSecretsClient  # type: ignore

    _KEY = UserSecretsClient().get_secret("OPENROUTER_API_KEY")
except Exception:
    _KEY = os.environ.get("OPENROUTER_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))

os.environ["DEEPSEEK_API_KEY"] = _KEY
os.environ["DEEPSEEK_BASE_URL"] = os.environ.get("DEEPSEEK_BASE_URL", "https://openrouter.ai/api/v1")
# ★ مدل را اینجا عوض کن (مثلاً "deepseek/deepseek-chat-v3-0324" یا "openai/gpt-oss-120b:free")
os.environ["DEEPSEEK_MODEL"] = os.environ.get("DEEPSEEK_MODEL", "deepseek/deepseek-chat-v3-0324")

# مسیر پروژه = محلِ همین فایل (با fallback برای اجرای paste-شده در سلول)
try:
    PROJECT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT = "/kaggle/working/Test-ChatBot"
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import gradio as gr  # noqa: E402

from src.conversation.manager import ConversationManager  # noqa: E402

# ---------------------------------------------------------------------------
# ۲) نمونهٔ مشترکِ بک‌اند (prompt و few-shot یک‌بار ساخته می‌شوند)
# ---------------------------------------------------------------------------
try:
    MANAGER = ConversationManager()
except Exception as exc:  # معمولاً نبودِ کلید یا اینترنت
    raise SystemExit(
        f"راه‌اندازی ناموفق بود: {exc}\n"
        "→ مطمئن شو Secret با نام OPENROUTER_API_KEY ساخته شده و اینترنتِ نوت‌بوک روشن است."
    )
TAX = MANAGER.taxonomy

# پالتِ رنگ برای برچسب‌ها؛ برچسب‌های ناشناخته رنگِ پایدار (هش‌شده) می‌گیرند.
_COLORS = {
    "incident": "#e5484d",         # قرمز — مشکل
    "service_request": "#3b82f6",  # آبی — درخواست
    "erp": "#0d9488",              # سبزآبی
    "staff": "#8b5cf6",            # بنفش
}
_PALETTE = ["#0ea5e9", "#f59e0b", "#10b981", "#ec4899", "#6366f1", "#14b8a6"]


def _color(label_id: str | None) -> str:
    if not label_id:
        return "#9ca3af"
    return _COLORS.get(label_id) or _PALETTE[sum(map(ord, label_id)) % len(_PALETTE)]


def _disp(name: str) -> str:
    """نام نمایشیِ کوتاه: اگر دوزبانه بود («Type / نوع») بخشِ بعد از / را بردار."""
    return name.split("/")[-1].strip() if "/" in name else name.strip()


def _result_card(resp: dict) -> str:
    """کارتِ HTML نتیجهٔ نهایی: نشان‌های رنگی + بنرِ وضعیت/هشدار."""
    result = resp.get("result")
    if not result:
        return ""
    labels = result.get("labels", {})
    badges = []
    for layer in TAX.layers:
        label_id = labels.get(layer.id)
        label = layer.get_label(label_id) if label_id else None
        badges.append(
            f"<div class='badge' style='background:{_color(label_id)}'>"
            f"<div class='cap'>{_disp(layer.name)}</div>"
            f"<div class='val'>{label.name if label else '—'}</div></div>"
        )
    html = f"<div class='cards'>{''.join(badges)}</div>"

    if result.get("needs_review"):
        # دلیلِ کوتاه: کدام لایه‌ها بدونِ شاهد (مبهم) ماندند
        amb = [
            _disp(TAX.get_layer(lid).name)
            for lid, ev in result.get("evidence", {}).items()
            if not ev and TAX.get_layer(lid)
        ]
        reason = "، ".join(amb) if amb else "یکی از لایه‌ها"
        html += (
            "<div class='warn'>⚠️ <b>نیاز به بازبینی انسانی</b> — حتی پس از سوال‌های تکمیلی، "
            f"شواهدِ کافی برای این بخش پیدا نشد: <b>{reason}</b></div>"
        )
    else:
        html += "<div class='ok'>✅ با اطمینان دسته‌بندی شد</div>"
    return html


def _bubble(resp: dict) -> str:
    """خلاصهٔ یک‌خطیِ نتیجه برای نمایش داخل حبابِ چت."""
    r = resp["result"]
    parts = []
    for layer in TAX.layers:
        label_id = r["labels"].get(layer.id)
        label = layer.get_label(label_id) if label_id else None
        parts.append(f"{_disp(layer.name)}: **{label.name if label else '—'}**")
    head = "⚠️ نتیجه (نیاز به بازبینی)" if r.get("needs_review") else "✅ نتیجه"
    return head + " — " + " | ".join(parts)


def _handle(resp: dict, history: list):
    """خروجی مشترک: [chat, session, result_html, answer_row, ans, raw]."""
    sid = resp["session_id"]
    need = resp["status"] == "need_info"
    if need:
        history = history + [{"role": "assistant", "content": f"❓ {resp['question']}"}]
        return history, sid, "", gr.update(visible=True), "", resp
    history = history + [{"role": "assistant", "content": _bubble(resp)}]
    return history, sid, _result_card(resp), gr.update(visible=False), "", resp


def start_ticket(summary: str, description: str, history: list):
    history = history or []
    if not (summary or "").strip() and not (description or "").strip():
        return history, None, "", gr.update(visible=False), "", {}
    history = history + [
        {"role": "user", "content": f"**خلاصه:** {summary or '—'}\n\n**شرح:** {description or '—'}"}
    ]
    try:
        resp = MANAGER.start(summary or "", description or "")
    except Exception as e:  # خطای API/شبکه
        history = history + [{"role": "assistant", "content": f"❌ خطا در فراخوانی مدل: {e}"}]
        return history, None, "", gr.update(visible=False), "", {"error": str(e)}
    return _handle(resp, history)


def answer_question(answer: str, session_id: str, history: list):
    history = history or []
    if not session_id:
        return history, session_id, "", gr.update(visible=False), "", {}
    history = history + [{"role": "user", "content": answer or "—"}]
    try:
        resp = MANAGER.answer(session_id, answer or "")
    except Exception as e:
        history = history + [{"role": "assistant", "content": f"❌ خطا: {e}"}]
        return history, session_id, "", gr.update(visible=False), "", {"error": str(e)}
    return _handle(resp, history)


def reset():
    return [], None, "", gr.update(visible=False), "", {}, "", ""


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700&display=swap');
*, .gradio-container * { font-family: 'Vazirmatn', Tahoma, sans-serif !important; }
.gradio-container { direction: rtl; }
.hdr { text-align:center; padding:16px; border-radius:16px; margin-bottom:10px;
       background:linear-gradient(135deg,#1e3a8a,#0d9488); color:#fff; }
.hdr h1 { margin:0; font-size:1.55rem; }
.hdr p  { margin:6px 0 0; opacity:.92; font-size:.95rem; }
.cards { display:flex; gap:12px; flex-wrap:wrap; margin-top:6px; }
.badge { flex:1; min-width:140px; padding:14px 16px; border-radius:16px; color:#fff;
         text-align:center; box-shadow:0 6px 16px rgba(0,0,0,.14); }
.badge .cap { font-size:.82rem; opacity:.92; }
.badge .val { font-size:1.3rem; font-weight:700; margin-top:3px; }
.ok   { margin-top:10px; padding:11px 14px; border-radius:12px; text-align:center;
        background:#dcfce7; color:#166534; font-weight:600; }
.warn { margin-top:10px; padding:12px 14px; border-radius:12px; line-height:1.8;
        background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }
"""

with gr.Blocks(css=_CSS, title="چت‌بات مسیریابی تیکت") as demo:
    gr.HTML(
        "<div class='hdr'><h1>🎫 چت‌بات مسیریابی تیکت</h1>"
        "<p>مشکل یا درخواستت را بنویس؛ بات تشخیص می‌دهد مربوط به کدام بخش است.</p></div>"
    )
    session = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            summary = gr.Textbox(label="خلاصه", rtl=True, placeholder="مثلاً: مشکل ثبت ورود و خروج")
            description = gr.Textbox(label="شرح", lines=5, rtl=True, placeholder="جزئیات کامل تیکت…")
            with gr.Row():
                send = gr.Button("ارسال تیکت", variant="primary")
                clear = gr.Button("گفت‌وگوی جدید")
            gr.Examples(
                examples=[
                    ["مشکل ثبت ورود و خروج",
                     "پانچ ورود و خروج برای تاریخ ۱۹ مرداد در سامانه ERP به درستی ثبت نشده است."],
                    ["دسترسی تایم‌شیت اپرور",
                     "لطفاً برای کارمندِ جدید واحد فنی دسترسی تایم‌شیت اپرور ایجاد گردد. با تشکر"],
                    ["خطا در ثبت درخواست وام",
                     "برای ثبت وام کوتاه‌مدت صندوق خطای «دو ضامن» می‌گیرم؛ لطفاً بررسی بفرمایید."],
                    ["مشکل در ارزیابی",
                     "در بخش ارزیابی مشکل دارم، لطفاً راهنمایی کنید."],  # عمداً مبهم → سوال تکمیلی
                ],
                inputs=[summary, description],
                label="نمونه‌ها (کلیک کن تا پر شود)",
            )
        with gr.Column(scale=3):
            chat = gr.Chatbot(label="گفت‌وگو", type="messages", height=420, rtl=True)
            with gr.Row(visible=False) as answer_row:
                ans = gr.Textbox(label="پاسخ شما به سوالِ بات", rtl=True, scale=4)
                ans_send = gr.Button("ارسال پاسخ", variant="primary", scale=1)
            result_html = gr.HTML()
            with gr.Accordion("🔧 JSON خام (برای دیباگ)", open=False):
                raw = gr.JSON()

    _start_out = [chat, session, result_html, answer_row, ans, raw]
    send.click(start_ticket, [summary, description, chat], _start_out)
    summary.submit(start_ticket, [summary, description, chat], _start_out)
    ans_send.click(answer_question, [ans, session, chat], _start_out)
    ans.submit(answer_question, [ans, session, chat], _start_out)
    clear.click(reset, None, [chat, session, result_html, answer_row, ans, raw, summary, description])


if __name__ == "__main__":
    # روی Kaggle حتماً share=True (لوکال‌هاست مستقیم در دسترس نیست).
    demo.queue().launch(share=True, show_error=True)
