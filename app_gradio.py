"""
رابط گرافیکیِ Ticket Routing Assistant — نسخهٔ Gradio (برای اجرا روی Kaggle).

نمایی حرفه‌ای روی همان ConversationManager موجود؛ بک‌اند دست‌نخورده است.

ویژگی‌ها:
  • UI انگلیسیِ حرفه‌ای؛ محتوای فارسی (تیکت/سوال‌ها) در نواحیِ RTL.
  • تمِ Dark/Light قابلِ سوییچ (بدون رفرش، تاریخچهٔ چت حفظ می‌شود).
  • آواتارِ اختصاصی کنارِ هر پیام (کاربر / دستیار).
  • اندیکاتورِ «Analyzing…» هنگام پردازش (تابع‌های generator).
  • نشان‌های رنگیِ نتیجه + هشدارِ «نیاز به بازبینی» با دلیلِ کوتاه.
  • جای‌گاهِ لوگوی شرکت در هدر + سوییچِ JSON خام برای دیباگ.

اجرا روی Kaggle:
  ۱) این فایل را در ریشهٔ پروژه بگذار: /kaggle/working/Test-ChatBot/app_gradio.py
  ۲) اینترنتِ نوت‌بوک روشن + Secret به نام OPENROUTER_API_KEY.
  ۳) نصب:  !pip -q install -U gradio openai pydantic PyYAML python-dotenv
  ۴) اجرا:  %run /kaggle/working/Test-ChatBot/app_gradio.py
"""
from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# ۱) پیکربندی — قبل از import پروژه
# ---------------------------------------------------------------------------
_KEY = ""
try:
    from kaggle_secrets import UserSecretsClient  # type: ignore

    _KEY = UserSecretsClient().get_secret("DEEPSEEK_API_KEY")
except Exception:
    _KEY = os.environ.get("OPENROUTER_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))

os.environ["DEEPSEEK_API_KEY"] = _KEY
os.environ["DEEPSEEK_BASE_URL"] = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# ★ مدلِ تست (طبق درخواست):
os.environ["DEEPSEEK_MODEL"] = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

# ★ لوگوی شرکت: مسیرِ فایل (مثلاً "assets/logo.png") یا URL. خالی = نشانِ پیش‌فرض.
LOGO_SRC = "data/logo.png"

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
# ۲) بک‌اندِ مشترک
# ---------------------------------------------------------------------------
try:
    MANAGER = ConversationManager()
except Exception as exc:
    raise SystemExit(
        f"Startup failed: {exc}\n"
        "→ Ensure the OPENROUTER_API_KEY secret exists and notebook internet is ON."
    )
TAX = MANAGER.taxonomy

# ---------------------------------------------------------------------------
# ۳) آواتارها — SVG آفلاین (پایدار روی همهٔ سیستم‌ها)
# ---------------------------------------------------------------------------
ASSETS = os.path.join(PROJECT, "assets")
os.makedirs(ASSETS, exist_ok=True)

_BOT_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>"
    "<stop offset='0' stop-color='#4f46e5'/><stop offset='1' stop-color='#0d9488'/>"
    "</linearGradient></defs>"
    "<rect width='64' height='64' rx='16' fill='url(#g)'/>"
    "<rect x='30' y='11' width='4' height='8' rx='2' fill='#fff'/>"
    "<circle cx='32' cy='10' r='3' fill='#fff'/>"
    "<rect x='15' y='21' width='34' height='26' rx='8' fill='#fff'/>"
    "<circle cx='26' cy='34' r='3.6' fill='#4f46e5'/>"
    "<circle cx='38' cy='34' r='3.6' fill='#0d9488'/>"
    "<rect x='25' y='41' width='14' height='3' rx='1.5' fill='#c7d2fe'/>"
    "</svg>"
)
_USER_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='16' fill='#475569'/>"
    "<circle cx='32' cy='25' r='11' fill='#e2e8f0'/>"
    "<path d='M13 53c0-10.5 8.5-17 19-17s19 6.5 19 17z' fill='#e2e8f0'/>"
    "</svg>"
)


def _write_asset(name: str, content: str) -> str:
    path = os.path.join(ASSETS, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


BOT_AVATAR = _write_asset("avatar_bot.svg", _BOT_SVG)
USER_AVATAR = _write_asset("avatar_user.svg", _USER_SVG)

# ---------------------------------------------------------------------------
# ۴) رنگ‌ها و کمک‌تابع‌های نمایش
# ---------------------------------------------------------------------------
_COLORS = {
    "incident": "#e5484d",
    "service_request": "#3b82f6",
    "erp": "#0d9488",
    "staff": "#8b5cf6",
}
_PALETTE = ["#0ea5e9", "#f59e0b", "#10b981", "#ec4899", "#6366f1", "#14b8a6"]


def _color(label_id: str | None) -> str:
    if not label_id:
        return "#9ca3af"
    return _COLORS.get(label_id) or _PALETTE[sum(map(ord, label_id)) % len(_PALETTE)]


def _cap(name: str) -> str:
    """عنوانِ انگلیسیِ لایه: از «Type / نوع درخواست» بخشِ قبل از / را برمی‌دارد."""
    return name.split("/")[0].strip() if "/" in name else name.strip()


def _logo_tag() -> str:
    src = (LOGO_SRC or "").strip()
    if not src:
        return "<div class='logo-fallback'>🎫</div>"
    if src.startswith("http"):
        return f"<img class='logo-img' src='{src}' alt='logo'/>"
    import base64
    import mimetypes

    p = src if os.path.isabs(src) else os.path.join(PROJECT, src)
    if not os.path.exists(p):
        return "<div class='logo-fallback'>🎫</div>"
    mime = mimetypes.guess_type(p)[0] or "image/png"
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"<img class='logo-img' src='data:{mime};base64,{b64}' alt='logo'/>"


def _result_card(resp: dict) -> str:
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
            f"<div class='cap'>{_cap(layer.name)}</div>"
            f"<div class='val'>{label.name if label else '—'}</div></div>"
        )
    html = f"<div class='cards'>{''.join(badges)}</div>"

    if result.get("needs_review"):
        amb = [
            _cap(TAX.get_layer(lid).name)
            for lid, ev in result.get("evidence", {}).items()
            if not ev and TAX.get_layer(lid)
        ]
        reason = ", ".join(amb) if amb else "one of the layers"
        html += (
            "<div class='warn'>⚠️ <b>Needs human review</b> — insufficient evidence to confidently "
            f"determine: <b>{reason}</b>. Showing the best guess after clarification.</div>"
        )
    else:
        html += "<div class='ok'>✅ Classified with high confidence</div>"
    return html


def _bubble(resp: dict) -> str:
    r = resp["result"]
    parts = []
    for layer in TAX.layers:
        label_id = r["labels"].get(layer.id)
        label = layer.get_label(label_id) if label_id else None
        parts.append(f"{_cap(layer.name)}: **{label.name if label else '—'}**")
    head = "⚠️ Needs review" if r.get("needs_review") else "✅ Classification"
    return head + " — " + " | ".join(parts)


def _handle(resp: dict, history: list):
    """خروجی مشترک: [chat, session, result_html, answer_row, ans, raw]."""
    sid = resp["session_id"]
    if resp["status"] == "need_info":
        history = history + [{"role": "assistant", "content": f"❓ {resp['question']}"}]
        return history, sid, "", gr.update(visible=True), "", resp
    history = history + [{"role": "assistant", "content": _bubble(resp)}]
    return history, sid, _result_card(resp), gr.update(visible=False), "", resp


# ---------------------------------------------------------------------------
# ۵) هندلرها (generator → اندیکاتورِ «Analyzing…»)
# ---------------------------------------------------------------------------
def start_ticket(summary: str, description: str, history: list):
    history = history or []
    if not (summary or "").strip() and not (description or "").strip():
        yield history, None, "", gr.update(visible=False), "", {}
        return
    history = history + [
        {"role": "user", "content": f"**Subject:** {summary or '—'}\n\n**Description:** {description or '—'}"}
    ]
    yield history + [{"role": "assistant", "content": "🔎 _Analyzing your ticket_"}], None, "", gr.update(
        visible=False
    ), "", {}
    try:
        resp = MANAGER.start(summary or "", description or "")
    except Exception as e:
        yield history + [
            {"role": "assistant", "content": f"❌ Error contacting the model: {e}"}
        ], None, "", gr.update(visible=False), "", {"error": str(e)}
        return
    yield _handle(resp, history)


def answer_question(answer: str, session_id: str, history: list):
    history = history or []
    if not session_id:
        yield history, session_id, "", gr.update(visible=False), "", {}
        return
    history = history + [{"role": "user", "content": answer or "—"}]
    yield history + [{"role": "assistant", "content": "🔎 _Reviewing your reply…_"}], session_id, "", gr.update(
        visible=False
    ), "", {}
    try:
        resp = MANAGER.answer(session_id, answer or "")
    except Exception as e:
        yield history + [
            {"role": "assistant", "content": f"❌ Error: {e}"}
        ], session_id, "", gr.update(visible=False), "", {"error": str(e)}
        return
    yield _handle(resp, history)


def reset():
    return [], None, "", gr.update(visible=False), "", {}, "", ""


# ---------------------------------------------------------------------------
# ۶) ظاهر (CSS) — شاملِ حالتِ تیره
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Vazirmatn:wght@400;500;700&display=swap');
*, .gradio-container * { font-family: 'Inter','Vazirmatn',system-ui,sans-serif; }
.hdr { display:flex; align-items:center; justify-content:space-between; gap:16px;
       padding:18px 22px; border-radius:18px; margin-bottom:12px; color:#fff;
       background:linear-gradient(135deg,#4f46e5,#0d9488);
       box-shadow:0 10px 30px rgba(79,70,229,.25); }
.brand { display:flex; align-items:center; gap:14px; }
.logo-plate { background:#fff; border-radius:14px; padding:8px 10px; display:flex;
              align-items:center; justify-content:center; min-width:54px; min-height:54px;
              box-shadow:0 4px 12px rgba(0,0,0,.18); }
.logo-img { height:38px; display:block; }
.logo-fallback { font-size:30px; line-height:1; }
.titles h1 { margin:0; font-size:1.5rem; font-weight:700; letter-spacing:.2px; }
.titles p  { margin:5px 0 0; opacity:.92; font-size:.92rem; max-width:560px; }
.theme-btn { background:rgba(255,255,255,.18); color:#fff; border:1px solid rgba(255,255,255,.4);
             border-radius:10px; padding:9px 14px; cursor:pointer; font-size:.9rem;
             font-weight:500; white-space:nowrap; transition:background .15s; }
.theme-btn:hover { background:rgba(255,255,255,.30); }
.cards { display:flex; gap:12px; flex-wrap:wrap; margin-top:6px; }
.badge { flex:1; min-width:150px; padding:14px 16px; border-radius:16px; color:#fff;
         text-align:center; box-shadow:0 6px 16px rgba(0,0,0,.16); }
.badge .cap { font-size:.8rem; opacity:.92; text-transform:uppercase; letter-spacing:.6px; }
.badge .val { font-size:1.32rem; font-weight:700; margin-top:3px; }
.ok   { margin-top:10px; padding:11px 14px; border-radius:12px; text-align:center;
        background:#dcfce7; color:#166534; font-weight:600; }
.warn { margin-top:10px; padding:12px 14px; border-radius:12px; line-height:1.8;
        background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }
.dark .ok   { background:#0f2e1d; color:#86efac; }
.dark .warn { background:#3a2c08; color:#fcd34d; border-color:#a16207; }
@media (max-width:680px){ .hdr{flex-direction:column; align-items:flex-start;} .titles p{display:none;} }
"""

_THEME_JS = (
    "var d=document.body.classList.contains('dark');"
    "[document.documentElement,document.body,document.querySelector('gradio-app')]"
    ".forEach(function(e){if(e){e.classList.toggle('dark',!d);}});"
)

_HEADER = f"""
<div class='hdr'>
  <div class='brand'>
    <div class='logo-plate'>{_logo_tag()}</div>
    <div class='titles'>
      <h1>Ticket Routing Assistant</h1>
      <p>Automated triage that classifies each request by type and routing domain, flagging low-confidence cases for human review.</p>
    </div>
  </div>
  <button class='theme-btn' onclick="{_THEME_JS}">🌗 Theme</button>
</div>
"""

# ---------------------------------------------------------------------------
# ۷) چیدمان
# ---------------------------------------------------------------------------
with gr.Blocks(css=_CSS, theme=gr.themes.Soft(primary_hue="indigo"), title="Ticket Routing Assistant") as demo:
    gr.HTML(_HEADER)
    session = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            summary = gr.Textbox(label="Subject", rtl=True, placeholder="Brief summary of the issue or request")
            description = gr.Textbox(
                label="Description", lines=5, rtl=True,
                placeholder="Describe the problem in detail — system, dates, error messages, etc.",
            )
            with gr.Row():
                send = gr.Button("Submit Ticket", variant="primary")
                clear = gr.Button("Reset")
            gr.Examples(
                examples=[
                    ["مشکل ثبت ورود و خروج",
                     "پانچ ورود و خروج برای تاریخ ... در سامانه ERP به درستی ثبت نشده است."],
                    ["دسترسی تایم‌شیت اپرور",
                     "لطفاً برای کارمندِ جدید واحد فنی دسترسی تایم‌شیت اپرور ایجاد گردد. با تشکر"],
                    ["خطا در ثبت درخواست وام",
                     "برای ثبت وام کوتاه‌مدت صندوق خطای «دو ضامن» می‌گیرم؛ لطفاً بررسی بفرمایید."],
                    ["مشکل در ارزیابی",
                     "در بخش ارزیابی مشکل دارم، لطفاً راهنمایی کنید."],
                ],
                inputs=[summary, description],
                label="Sample tickets — click to autofill",
            )
        with gr.Column(scale=3):
            chat = gr.Chatbot(
                label="Conversation", type="messages", height=440, rtl=True,
                avatar_images=(USER_AVATAR, BOT_AVATAR), show_copy_button=True,
            )
            with gr.Row(visible=False) as answer_row:
                ans = gr.Textbox(label="Your reply", rtl=True, scale=4, placeholder="Type your reply…")
                ans_send = gr.Button("Send", variant="primary", scale=1)
            result_html = gr.HTML()
            with gr.Accordion("Raw response (debug)", open=False):
                raw = gr.JSON()

    _out = [chat, session, result_html, answer_row, ans, raw]
    send.click(start_ticket, [summary, description, chat], _out)
    summary.submit(start_ticket, [summary, description, chat], _out)
    ans_send.click(answer_question, [ans, session, chat], _out)
    ans.submit(answer_question, [ans, session, chat], _out)
    clear.click(reset, None, [chat, session, result_html, answer_row, ans, raw, summary, description])


if __name__ == "__main__":
    demo.queue().launch(share=True, show_error=True, allowed_paths=[ASSETS])
