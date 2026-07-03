"""
دیزاین‌سیستمِ مشترکِ گزارش‌های HTML — CSS و کامپوننت‌های قابل‌استفادهٔ مجدد.

هم گزارشِ هزینه (`scripts/cost_report.py`) و هم گزارشِ ترکیبیِ عملکرد+هزینه
(`scripts/perf_report.py`) از همین اجزا استفاده می‌کنند تا کاملاً هم‌زبان باشند:
یک پالت، یک تایپوگرافی، یک زبانِ بصری. خروجی همیشه **خودبسنده** است (CSS/SVG امبد،
بدونِ وابستگیِ شبکه) تا با یک دابل‌کلیک در مرورگر باز شود و با Print → PDF بشود.
"""
from __future__ import annotations

import html
from datetime import date

# --- پالت ---
INDIGO = "#6366f1"
TEAL = "#14b8a6"
AMBER = "#f59e0b"
GREEN = "#22c55e"
RED = "#ef4444"
VIOLET = "#a78bfa"
SLATE = "#64748b"


def grade(v: float, *, good: float = 0.90, mid: float = 0.80) -> str:
    """رنگِ نمره: سبز/کهربایی/قرمز."""
    if v >= good:
        return GREEN
    if v >= mid:
        return AMBER
    return RED


# ---------------------------------------------------------------------------
# قالب‌بندیِ اعداد
# ---------------------------------------------------------------------------
def esc(s) -> str:
    return html.escape(str(s))


def usd(x: float, decimals: int | None = None) -> str:
    if decimals is None:
        a = abs(x)
        decimals = 2 if a == 0 else 6 if a < 0.001 else 4 if a < 1 else 2
    return f"${x:,.{decimals}f}"


def intc(n: float) -> str:
    return f"{int(round(n)):,}"


def pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}%"


# ---------------------------------------------------------------------------
# کامپوننت‌ها
# ---------------------------------------------------------------------------
def kpi_card(label: str, value: str, sub: str = "", accent: str = INDIGO,
             badge: str = "", badge_color: str | None = None, extra_class: str = "") -> str:
    bc = badge_color or accent
    badge_html = f'<span class="badge" style="--bc:{bc}">{esc(badge)}</span>' if badge else ""
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi {extra_class}" style="--c:{accent}">'
        f'<div class="kpi-label">{esc(label)}{badge_html}</div>'
        f'<div class="kpi-value">{esc(value)}</div>'
        f"{sub_html}</div>"
    )


def section(title: str, inner: str, *, index: str = "", accent: str = INDIGO,
            status: str = "", status_color: str = INDIGO) -> str:
    """بلوکِ سطح‌بالا با سرتیترِ شماره‌دار و وضعیتِ اختیاری."""
    chip = f'<span class="sec-i" style="--c:{accent}">{esc(index)}</span>' if index else ""
    badge = f'<span class="sec-badge" style="--bc:{status_color}">{esc(status)}</span>' if status else ""
    return (
        '<section class="sec">'
        f'<div class="sec-h">{chip}<span class="sec-t">{esc(title)}</span>{badge}</div>'
        f'<div class="sec-body">{inner}</div></section>'
    )


def panel(title: str, inner: str, *, accent: str = TEAL) -> str:
    return (
        '<div class="panel">'
        f'<p class="p-title"><span class="d" style="background:{accent}"></span>{esc(title)}</p>'
        f"{inner}</div>"
    )


def stacked_bar(segments: list[tuple[str, float, str]], *, money: bool = False) -> str:
    """نوارِ افقیِ انباشته با افسانهٔ خطی. segments = [(label, value, color), ...]."""
    total = sum(max(v, 0) for _, v, _ in segments) or 1.0
    fmt = usd if money else intc
    bars = "".join(
        f'<div class="seg" style="width:{max(v, 0) / total * 100:.4f}%;background:{c}" '
        f'title="{esc(label)}: {fmt(v)}"></div>'
        for label, v, c in segments
    )
    legend = "".join(
        f'<div class="lg-item"><span class="dot" style="background:{c}"></span>'
        f'<span class="lg-label">{esc(label)}</span>'
        f'<span class="lg-val">{fmt(v)} · {v / total * 100:.1f}%</span></div>'
        for label, v, c in segments
    )
    return f'<div class="bar">{bars}</div><div class="legend">{legend}</div>'


def donut(parts: list[tuple[str, float, str]], center_top: str, center_bot: str,
          *, money: bool = True) -> str:
    """دوناتِ SVG. parts = [(label, value, color), ...]."""
    total = sum(max(v, 0) for _, v, _ in parts) or 1.0
    fmt = usd if money else intc
    r = 52.0
    circ = 2 * 3.141592653589793 * r
    offset = 0.0
    rings = ""
    for _, v, c in parts:
        dash = max(v, 0) / total * circ
        rings += (
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="{c}" stroke-width="20" '
            f'stroke-dasharray="{dash:.3f} {circ - dash:.3f}" stroke-dashoffset="{-offset:.3f}" '
            f'transform="rotate(-90 70 70)" />'
        )
        offset += dash
    legend = "".join(
        f'<div class="lg-item"><span class="dot" style="background:{c}"></span>'
        f'<span class="lg-label">{esc(label)}</span>'
        f'<span class="lg-val">{fmt(v)} · {v / total * 100:.1f}%</span></div>'
        for label, v, c in parts
    )
    return (
        '<div class="donut-wrap">'
        f'<svg viewBox="0 0 140 140" class="donut">{rings}'
        f'<text x="70" y="64" class="d-top">{esc(center_top)}</text>'
        f'<text x="70" y="84" class="d-bot">{esc(center_bot)}</text></svg>'
        f'<div class="legend">{legend}</div></div>'
    )


def table(headers: list[str], rows: list[list[str]], *, right_from: int = 1,
          total_row: list[str] | None = None) -> str:
    head = "".join(f'<th class="{"r" if i >= right_from else "l"}">{esc(h)}</th>'
                   for i, h in enumerate(headers))
    body = ""
    for row in rows:
        body += "<tr>" + "".join(
            f'<td class="{"r" if i >= right_from else "l"}">{c}</td>' for i, c in enumerate(row)
        ) + "</tr>"
    if total_row:
        body += '<tr class="tot">' + "".join(
            f'<td class="{"r" if i >= right_from else "l"}">{c}</td>' for i, c in enumerate(total_row)
        ) + "</tr>"
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def progress(value: float, color: str, label: str | None = None) -> str:
    """نوارِ پیشرفتِ درون‌سلولی (۰..۱) با برچسبِ روی آن."""
    txt = label if label is not None else pct(value)
    w = max(0.0, min(value, 1.0)) * 100
    return (
        '<div class="prog">'
        f'<div class="prog-fill" style="width:{w:.2f}%;background:{color}"></div>'
        f'<span class="prog-txt">{esc(txt)}</span></div>'
    )


def heatmap(label_ids: list[str], names: dict, confusion: dict, *, title: str = "") -> str:
    """
    ماتریسِ درهم‌ریختگیِ بصری. سطر=واقعی، ستون=پیش‌بینی (+ ستونِ none در صورتِ وجود).
    قطر (درست) با شدتِ سبز، خارجِ قطر (خطا) با شدتِ قرمز، none با خاکستری رنگ می‌گیرد.
    """
    has_none = any(confusion.get(t, {}).get(None, 0) for t in label_ids)
    cols = list(label_ids) + ([None] if has_none else [])
    col_lbl = [names.get(c, c) if c is not None else "∅ none" for c in cols]

    head = '<th class="hm-corner">true ╲ pred</th>' + "".join(
        f'<th class="hm-h">{esc(c)}</th>' for c in col_lbl
    )
    body = ""
    for t in label_ids:
        row_cells = dict(confusion.get(t, {}))
        row_sum = sum(row_cells.values()) or 1
        cells = f'<th class="hm-rh">{esc(names.get(t, t))}</th>'
        for c in cols:
            n = row_cells.get(c, 0)
            frac = n / row_sum
            if c == t:
                base = "34,197,94"      # سبز = درست
            elif c is None:
                base = "100,116,139"    # خاکستری = بدونِ پیش‌بینی
            else:
                base = "239,68,68"      # قرمز = خطا
            alpha = 0.06 + 0.80 * frac
            strong = frac > 0.45
            cells += (
                f'<td class="hm-c" style="background:rgba({base},{alpha:.3f})">'
                f'<span class="hm-n {"on" if strong else ""}">{n}</span>'
                f'<span class="hm-p">{frac * 100:.0f}%</span></td>'
            )
        body += f"<tr>{cells}</tr>"
    cap = f'<div class="hm-title">{esc(title)}</div>' if title else ""
    return f'{cap}<table class="hm"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# ---------------------------------------------------------------------------
# CSS — یک طرحِ واحد برای همهٔ گزارش‌ها
# ---------------------------------------------------------------------------
CSS = """
:root{
  --bg:#0b1020; --card:#141b2e; --card2:#111827; --line:#243049;
  --ink:#e9eefc; --mut:#93a4c8; --mut2:#64748b;
}
*{box-sizing:border-box}
body{margin:0;background:
  radial-gradient(1100px 560px at 12% -8%,#1a2748 0%,transparent 55%),
  radial-gradient(1000px 520px at 100% 0%,#10263a 0%,transparent 50%),
  var(--bg);
  color:var(--ink);font-family:'Segoe UI',Roboto,Helvetica,Arial,system-ui,sans-serif;
  -webkit-font-smoothing:antialiased;padding:40px 22px;
  -webkit-print-color-adjust:exact;print-color-adjust:exact}
.wrap{max-width:1180px;margin:0 auto}

/* هدر */
.head{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;
  border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:8px}
.h-title{font-size:30px;font-weight:800;letter-spacing:.2px;margin:0;line-height:1.15}
.h-grad{background:linear-gradient(90deg,#818cf8,#22d3ee 60%,#2dd4bf);
  -webkit-background-clip:text;background-clip:text;color:transparent}
.h-sub{color:var(--mut);font-size:13.5px;margin-top:9px;line-height:1.75}
.h-sub b{color:var(--ink);font-weight:600}
.pill{display:inline-block;padding:3px 11px;border:1px solid var(--line);border-radius:999px;
  font-size:11.5px;color:var(--mut);margin-left:6px;background:rgba(255,255,255,.02)}

/* سکشن سطح‌بالا */
.sec{margin-top:34px}
.sec-h{display:flex;align-items:center;gap:12px;margin-bottom:16px}
.sec-i{width:26px;height:26px;border-radius:8px;flex:none;display:flex;align-items:center;
  justify-content:center;font-size:13px;font-weight:800;color:var(--c);
  border:1px solid var(--c);background:color-mix(in srgb,var(--c) 16%,transparent)}
.sec-t{font-size:17px;font-weight:800;letter-spacing:.2px}
.sec-badge{margin-left:auto;font-size:12px;font-weight:800;color:var(--bc);
  border:1px solid var(--bc);border-radius:999px;padding:3px 12px;
  background:color-mix(in srgb,var(--bc) 14%,transparent)}

/* گرید */
.grid{display:grid;gap:16px}
.k4{grid-template-columns:repeat(4,1fr)}
.k3{grid-template-columns:repeat(3,1fr)}
.k2{grid-template-columns:repeat(2,1fr)}
.c2{grid-template-columns:1.35fr 1fr}
.c2b{grid-template-columns:1fr 1fr}

/* کارتِ KPI */
.kpi{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);
  border-radius:16px;padding:18px 18px 16px;position:relative;overflow:hidden}
.kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--c)}
.kpi-label{color:var(--mut);font-size:12px;font-weight:600;display:flex;align-items:center;
  justify-content:space-between;gap:8px;text-transform:uppercase;letter-spacing:.5px}
.kpi-value{font-size:30px;font-weight:800;margin-top:10px;letter-spacing:.3px}
.kpi-sub{color:var(--mut2);font-size:12px;margin-top:7px;line-height:1.5}
.badge{font-size:10.5px;font-weight:800;color:var(--bc);border:1px solid var(--bc);
  border-radius:999px;padding:1px 9px;text-transform:none;letter-spacing:0;
  background:color-mix(in srgb,var(--bc) 16%,transparent)}
.hero{padding:22px}
.hero .kpi-value{font-size:44px}
.pass{background:linear-gradient(135deg,rgba(34,197,94,.16),rgba(20,184,166,.08));
  border-color:rgba(34,197,94,.35)}
.fail{background:linear-gradient(135deg,rgba(239,68,68,.15),rgba(245,158,11,.07));
  border-color:rgba(239,68,68,.35)}
.save{background:linear-gradient(135deg,rgba(34,197,94,.16),rgba(20,184,166,.10));
  border-color:rgba(34,197,94,.35)}
.save .kpi-value{color:#4ade80}

/* پنل */
.panel{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);
  border-radius:16px;padding:20px}
.p-title{font-size:13.5px;font-weight:700;margin:0 0 16px;display:flex;align-items:center;gap:9px;
  text-transform:uppercase;letter-spacing:.5px;color:var(--ink)}
.p-title .d{width:9px;height:9px;border-radius:3px}

/* نوار انباشته + افسانه */
.bar{display:flex;height:30px;border-radius:9px;overflow:hidden;border:1px solid var(--line)}
.seg{height:100%}
.legend{margin-top:16px;display:flex;flex-direction:column;gap:11px}
.lg-item{display:flex;align-items:center;gap:10px;font-size:13px}
.dot{width:11px;height:11px;border-radius:3px;flex:none}
.lg-label{color:var(--mut)}
.lg-val{margin-left:auto;font-variant-numeric:tabular-nums;font-weight:600}

/* دونات */
.donut-wrap{display:flex;gap:20px;align-items:center}
.donut{width:140px;height:140px;flex:none}
.d-top{fill:var(--ink);font-size:17px;font-weight:800;text-anchor:middle}
.d-bot{fill:var(--mut);font-size:9.5px;text-anchor:middle;text-transform:uppercase;letter-spacing:1px}
.donut-wrap .legend{flex:1}

/* جدول */
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:10px 12px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{color:var(--mut);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.4px}
td.l,th.l{text-align:left}
td.r,th.r{text-align:right}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:rgba(255,255,255,.02)}
.tot td{font-weight:800;border-top:1px solid var(--line);border-bottom:none}

/* نوارِ پیشرفتِ درون‌سلولی */
.prog{position:relative;display:inline-block;width:158px;height:22px;border-radius:7px;
  background:rgba(255,255,255,.05);overflow:hidden;vertical-align:middle}
.prog-fill{position:absolute;left:0;top:0;bottom:0;border-radius:7px;opacity:.85}
.prog-txt{position:absolute;left:10px;top:0;line-height:22px;font-size:11.5px;font-weight:700;
  color:var(--ink);text-shadow:0 1px 2px rgba(0,0,0,.5)}

/* heatmap */
.hm-title{font-size:12.5px;font-weight:700;color:var(--mut);margin:0 0 10px}
table.hm{border-collapse:separate;border-spacing:4px;width:auto}
.hm th,.hm td{border:none;padding:0}
.hm-h{font-size:11px;color:var(--mut);text-align:center;padding:0 4px 6px!important;font-weight:600}
.hm-corner{font-size:10.5px;color:var(--mut2);text-align:right;padding-right:8px!important}
.hm-rh{font-size:11.5px;color:var(--ink);text-align:right;padding-right:8px!important;
  font-weight:600;white-space:nowrap}
.hm-c{width:92px;height:58px;border-radius:10px;text-align:center;vertical-align:middle;
  border:1px solid var(--line)!important;position:relative}
.hm-n{display:block;font-size:18px;font-weight:800;color:var(--ink)}
.hm-n.on{color:#fff}
.hm-p{display:block;font-size:10px;color:var(--mut);margin-top:1px}
.hm-n.on + .hm-p{color:rgba(255,255,255,.8)}

/* پاورقی */
.foot{margin-top:30px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--mut2);font-size:11.5px;line-height:1.85}
.foot b{color:var(--mut)}
.mt{margin-top:16px}

@media print{
  body{padding:0}
  .kpi,.panel,.sec{break-inside:avoid}
}
"""


def page(*, title: str, header_html: str, body_html: str) -> str:
    """سند کاملِ HTML با CSSِ امبد."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{header_html}
{body_html}
</div>
</body>
</html>"""


def header(*, title_html: str, subtitle_html: str, pills: list[str]) -> str:
    pill_html = "".join(f'<span class="pill">{esc(p)}</span>' for p in pills)
    return (
        '<div class="head"><div>'
        f'<h1 class="h-title">{title_html}</h1>'
        f'<div class="h-sub">{subtitle_html}</div></div>'
        f'<div style="text-align:right">{pill_html}'
        f'<div class="h-sub" style="margin-top:10px">{date.today().isoformat()}</div></div></div>'
    )
