"""رابط گرافیکیِ Ticket Assistant — نسخهٔ Gradio (برای تست، از جمله روی Kaggle).

همان تجربهٔ کاربریِ SPA (پوشهٔ web/) روی همان بک‌اند، در قالب Gradio:
  ۱) مشخصات کاربر (کد پرسنلی + نام + نام خانوادگی) — یک‌بار، با اعتبارسنجی.
  ۲) جستجوی FAQ (۲۰ قالبِ آماده از data/faq.json) → پرکردنِ خودکارِ تیکت.
  ۳) توضیحِ درخواست → دسته‌بندی با حداکثر ۲ سوالِ تکمیلی.
  ۴) ثبتِ نهایی → شمارهٔ پیگیری TKT-YYYY-NNNNN (در logs/tickets.jsonl).

★ نسخهٔ Gradio: این UI با API نسخهٔ ۴/۵ نوشته شده (type="messages"، css/theme در
Blocks). در Gradio ۶ این‌ها حذف شده‌اند؛ پس نصب را به `gradio>=4,<6` مقید کنید.

اجرا روی Kaggle:
  ۱) این فایل در ریشهٔ پروژه: /kaggle/working/ChatBot-v2/app_gradio.py
  ۲) اینترنتِ نوت‌بوک روشن + Secret به نام DEEPSEEK_API_KEY.
  ۳) نصب:  !pip -q install -U "gradio>=4,<6" openai pydantic PyYAML python-dotenv
  ۴) اجرا:  %run /kaggle/working/ChatBot-v2/app_gradio.py
"""
from __future__ import annotations

import os
import sys
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# ۱) پیکربندی — قبل از import پروژه
# ---------------------------------------------------------------------------
# بارگذاریِ .env برای اجرای محلی/PyCharm (روی Kaggle که .env ندارد، بی‌اثر است).
# باید قبل از خواندنِ کلید باشد وگرنه os.environ خالی می‌ماند و بعداً load_dotenv
# مقدارِ خالی را بازنویسی نمی‌کند.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    pass

_KEY = os.getenv("DEEPSEEK_API_KEY", "")
try:
    from kaggle_secrets import UserSecretsClient
    _KEY = UserSecretsClient().get_secret("DEEPSEEK_API_KEY")
except Exception:from dotenv import load_dotenv

os.environ["DEEPSEEK_API_KEY"] = _KEY
os.environ["DEEPSEEK_BASE_URL"] = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
# ★ مدلِ تست (طبق درخواست):
os.environ["DEEPSEEK_MODEL"] = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")

# ★ لوگوی شرکت: مسیرِ فایل یا URL. خالی = نشانِ پیش‌فرض.
LOGO_SRC = "data/logo.png"

try:
    PROJECT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT = "C:/Users/sina/PycharmProjects/ChatBot-v2/.claude/worktrees/wizardly-agnesi-6d8824"
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

import re  # noqa: E402

import gradio as gr  # noqa: E402

from src.conversation.manager import ConversationManager  # noqa: E402
from src.db.database import get_database  # noqa: E402
from src.faq import load_faq, search_faq  # noqa: E402
from src.tickets.store import TicketStore  # noqa: E402

# ---------------------------------------------------------------------------
# ۲) بک‌اندِ مشترک
# ---------------------------------------------------------------------------
try:
    MANAGER = ConversationManager()
except Exception as exc:
    raise SystemExit(
        f"Startup failed: {exc}\n"
        "→ Ensure the DEEPSEEK_API_KEY secret exists and notebook internet is ON."
    )
TAX = MANAGER.taxonomy
STORE = TicketStore(get_database())
FAQ_CATEGORIES, FAQ_ITEMS = load_faq()
_FAQ_BY_QUESTION = {it.question: it for it in FAQ_ITEMS}

# ---------------------------------------------------------------------------
# ۳) آواتارها — SVG آفلاین
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
    "incident": "#dc2626",
    "service_request": "#2563eb",
    "erp": "#0d9488",
    "staff": "#7c3aed",
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


def _valid_identity(employee_id: str, first: str, last: str) -> str | None:
    """None = معتبر؛ در غیر این صورت پیامِ خطا."""
    if not re.fullmatch(r"\d{3,10}", (employee_id or "").strip()):
        return "Enter a numeric Employee ID (3–10 digits)."
    if len((first or "").strip()) < 2 or len((last or "").strip()) < 2:
        return "Enter your first and last name."
    return None


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
        html += (
            "<div class='warn'>⚠️ <b>Will be double-checked.</b> The assistant wasn't fully sure, "
            "so a support agent will verify the routing after you submit.</div>"
        )
    else:
        html += "<div class='ok'>✅ Classified with high confidence — ready to submit.</div>"
    return html


def _ref_banner(record: dict) -> str:
    labels = record.get("labels", {})
    pills = []
    for layer in TAX.layers:
        label_id = labels.get(layer.id)
        label = layer.get_label(label_id) if label_id else None
        pills.append(
            f"<span class='pill' style='background:{_color(label_id)}'>"
            f"{label.name if label else '—'}</span>"
        )
    return (
        "<div class='refbox'>"
        "<div class='refhead'>✅ Ticket submitted — keep this reference:</div>"
        f"<div class='refnum'>{record['reference']}</div>"
        f"<div class='refmeta'>{' '.join(pills)}</div>"
        "</div>"
    )


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
    """خروجی مشترک: [chat, session, result_html, answer_row, ans, raw, submit_row, ref_html]."""
    sid = resp["session_id"]
    if resp["status"] == "need_info":
        n = int(resp.get("questions_asked") or 0) + 1
        history = history + [
            {"role": "assistant", "content": f"❓ {resp['question']}\n\n_Quick question {n} of 2._"}
        ]
        return history, sid, "", gr.update(visible=True), "", resp, gr.update(visible=False), ""
    history = history + [{"role": "assistant", "content": _bubble(resp)}]
    return history, sid, _result_card(resp), gr.update(visible=False), "", resp, gr.update(visible=True), ""


# ---------------------------------------------------------------------------
# ۵) هندلرها
# ---------------------------------------------------------------------------
_IDLE = ("", None, "", gr.update(visible=False), "", {}, gr.update(visible=False), "")


def filter_faq(query: str):
    hits = search_faq(FAQ_ITEMS, query or "")
    label = f"Matching templates ({len(hits)})" if (query or "").strip() else "Common requests — pick one to autofill"
    return gr.update(choices=[it.question for it in hits], value=None, label=label)


def apply_faq(question: str | None, summary: str, description: str):
    item = _FAQ_BY_QUESTION.get(question or "")
    if item is None:
        return summary, description
    gr.Info("Template applied — replace the [BRACKETED] parts with your details.")
    return item.summary, item.description


def start_ticket(employee_id: str, first: str, last: str, summary: str, description: str, history: list):
    history = history or []
    err = _valid_identity(employee_id, first, last)
    if err:
        gr.Warning(f"Step 1 — {err}")
        yield (history, *_IDLE[1:])
        return
    if not (summary or "").strip() and not (description or "").strip():
        gr.Warning("Step 3 — fill in the subject or the description first.")
        yield (history, *_IDLE[1:])
        return
    history = history + [
        {"role": "user", "content": f"**Subject:** {summary or '—'}\n\n**Description:** {description or '—'}"}
    ]
    yield history + [{"role": "assistant", "content": "🔎 _Analyzing your ticket…_"}], None, "", gr.update(
        visible=False
    ), "", {}, gr.update(visible=False), ""
    try:
        resp = MANAGER.start(summary or "", description or "")
    except Exception as e:
        yield history + [
            {"role": "assistant", "content": f"❌ Error contacting the model: {e}"}
        ], None, "", gr.update(visible=False), "", {"error": str(e)}, gr.update(visible=False), ""
        return
    yield _handle(resp, history)


def answer_question(answer: str, session_id: str, history: list):
    history = history or []
    if not session_id:
        yield (history, *_IDLE[1:])
        return
    history = history + [{"role": "user", "content": answer or "—"}]
    yield history + [{"role": "assistant", "content": "🔎 _Reviewing your reply…_"}], session_id, "", gr.update(
        visible=False
    ), "", {}, gr.update(visible=False), ""
    try:
        resp = MANAGER.answer(session_id, answer or "")
    except Exception as e:
        yield history + [
            {"role": "assistant", "content": f"❌ Error: {e}"}
        ], session_id, "", gr.update(visible=False), "", {"error": str(e)}, gr.update(visible=False), ""
        return
    yield _handle(resp, history)


def submit_ticket(
    employee_id: str, first: str, last: str,
    summary: str, description: str,
    session_id: str | None, raw_resp: dict, history: list,
):
    result = (raw_resp or {}).get("result") or {}
    if not result:
        gr.Warning("Analyze the ticket first.")
        return history, gr.update(visible=False), ""
    record = STORE.submit(
        employee_id=(employee_id or "").strip(),
        first_name=(first or "").strip(),
        last_name=(last or "").strip(),
        summary=(summary or "").strip(),
        description=(description or "").strip(),
        labels=result.get("labels", {}),
        needs_review=bool(result.get("needs_review")),
        session_id=session_id,
    )
    gr.Info(f"Ticket {record['reference']} submitted.")
    history = (history or []) + [
        {"role": "assistant", "content": f"🎫 Ticket submitted — reference **{record['reference']}**."}
    ]
    return history, gr.update(visible=False), _ref_banner(record)


def reset():
    return (
        [], None, "", gr.update(visible=False), "", {}, gr.update(visible=False), "",
        "", "",  # subject, description
        "", gr.update(choices=[it.question for it in FAQ_ITEMS], value=None),  # search, radio
    )


# ---------------------------------------------------------------------------
# ۶) ظاهر (CSS) — شاملِ حالتِ تیره
# ---------------------------------------------------------------------------
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@400;500;700&display=swap');
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
.titles h1 { margin:0; font-size:1.5rem; font-weight:800; letter-spacing:.2px; }
.titles p  { margin:5px 0 0; opacity:.92; font-size:.92rem; max-width:560px; }
.theme-btn { background:rgba(255,255,255,.18); color:#fff; border:1px solid rgba(255,255,255,.4);
             border-radius:10px; padding:9px 14px; cursor:pointer; font-size:.9rem;
             font-weight:500; white-space:nowrap; transition:background .15s; }
.theme-btn:hover { background:rgba(255,255,255,.30); }
.step-title { font-weight:700; font-size:1.02rem; margin:2px 0 0; }
.cards { display:flex; gap:12px; flex-wrap:wrap; margin-top:6px; }
.badge { flex:1; min-width:150px; padding:14px 16px; border-radius:16px; color:#fff;
         text-align:center; box-shadow:0 6px 16px rgba(0,0,0,.16); }
.badge .cap { font-size:.72rem; opacity:.92; text-transform:uppercase; letter-spacing:.08em; }
.badge .val { font-size:1.32rem; font-weight:800; margin-top:3px; }
.ok   { margin-top:10px; padding:11px 14px; border-radius:12px; text-align:center;
        background:#dcfce7; color:#166534; font-weight:600; }
.warn { margin-top:10px; padding:12px 14px; border-radius:12px; line-height:1.8;
        background:#fef3c7; color:#92400e; border:1px solid #fcd34d; }
.refbox { margin-top:10px; padding:18px; border-radius:16px; text-align:center;
          background:#ecfdf5; border:2px dashed #34d399; color:#065f46; }
.refhead { font-weight:600; }
.refnum  { font-family:Consolas,monospace; font-size:1.6rem; font-weight:700;
           letter-spacing:.06em; margin:8px 0; }
.refmeta .pill { display:inline-block; margin:0 4px; padding:3px 12px; border-radius:999px;
                 color:#fff; font-size:.78rem; font-weight:700; }
.dark .ok   { background:#0f2e1d; color:#86efac; }
.dark .warn { background:#3a2c08; color:#fcd34d; border-color:#a16207; }
.dark .refbox { background:#052e22; border-color:#065f46; color:#6ee7b7; }
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
      <h1>Service Desk — Ticket Assistant</h1>
      <p>Describe your issue or pick a common request; the assistant routes it to the right team and gives you a tracking reference.</p>
    </div>
  </div>
  <button class='theme-btn' onclick="{_THEME_JS}">🌗 Theme</button>
</div>
"""

# ---------------------------------------------------------------------------
# ۷) چیدمان
# ---------------------------------------------------------------------------
with gr.Blocks(css=_CSS, theme=gr.themes.Soft(primary_hue="indigo"), title="Service Desk — Ticket Assistant") as demo:
    gr.HTML(_HEADER)
    session = gr.State(None)

    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML("<p class='step-title'>1 · Your details</p>")
            with gr.Row():
                emp_id = gr.Textbox(label="Employee ID", placeholder="e.g. 263669", scale=1)
                first_name = gr.Textbox(label="First name", scale=1)
                last_name = gr.Textbox(label="Last name", scale=1)

            gr.HTML("<p class='step-title'>2 · Start from a common request <small>(optional)</small></p>")
            faq_search = gr.Textbox(
                label="Search common requests", placeholder="e.g. punch, loan, timesheet, وام…"
            )
            faq_radio = gr.Radio(
                choices=[it.question for it in FAQ_ITEMS], value=None,
                label="Common requests — pick one to autofill",
            )

            gr.HTML("<p class='step-title'>3 · Describe your request</p>")
            summary = gr.Textbox(label="Subject", placeholder="Brief summary of the issue or request")
            description = gr.Textbox(
                label="Description", lines=6,
                placeholder="Describe the problem in detail — system, dates, employee IDs, exact error messages… (Persian or English)",
            )
            with gr.Row():
                send = gr.Button("Analyze & continue", variant="primary")
                clear = gr.Button("Start over")

        with gr.Column(scale=3):
            chat = gr.Chatbot(
                label="Assistant", type="messages", height=430, rtl=False,
                avatar_images=(USER_AVATAR, BOT_AVATAR), show_copy_button=True,
            )
            with gr.Row(visible=False) as answer_row:
                ans = gr.Textbox(label="Your answer", scale=4, placeholder="Type your answer…")
                ans_send = gr.Button("Send", variant="primary", scale=1)
            result_html = gr.HTML()
            with gr.Row(visible=False) as submit_row:
                submit_btn = gr.Button("🎫 Submit ticket", variant="primary")
            ref_html = gr.HTML()
            with gr.Accordion("Raw response (debug)", open=False):
                raw = gr.JSON()

    _out = [chat, session, result_html, answer_row, ans, raw, submit_row, ref_html]

    faq_search.change(filter_faq, faq_search, faq_radio)
    faq_radio.change(apply_faq, [faq_radio, summary, description], [summary, description])

    _in = [emp_id, first_name, last_name, summary, description, chat]
    send.click(start_ticket, _in, _out)
    description.submit(start_ticket, _in, _out)
    ans_send.click(answer_question, [ans, session, chat], _out)
    ans.submit(answer_question, [ans, session, chat], _out)
    submit_btn.click(
        submit_ticket,
        [emp_id, first_name, last_name, summary, description, session, raw, chat],
        [chat, submit_row, ref_html],
    )
    clear.click(reset, None, _out + [summary, description, faq_search, faq_radio])


if __name__ == "__main__":
    demo.queue().launch(share=True, show_error=True, allowed_paths=[ASSETS])
