"""
HTML cost & token report — a single standalone file, ready to present to management.

Fed by **real** data (not a hypothetical calculator). Two sources:
  1) The production log (no API needed):
        python -m scripts.cost_report --from-log logs/interactions.jsonl --out cost_report.html
  2) A real model run over the dataset (requires DEEPSEEK_API_KEY):
        python -m scripts.cost_report tests/Ticketing_DB.jsonl --frac 0.2 --workers 6 \
            --out cost_report.html

The output is a self-contained HTML file (embedded CSS/SVG, no network
dependency) that opens in a browser and turns into a slide/report via
Print → Save as PDF.

Design: every number comes from `src.reporting.cost.CostBreakdown` so it
matches the visual dashboard (`scripts/report.py`) exactly.
"""
from __future__ import annotations

import argparse
import html
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reporting.cost import (  # noqa: E402
    CostBreakdown,
    Pricing,
    aggregate_log,
    breakdown_from_eval,
)

# Palette — consistent with scripts/report.py
_INDIGO = "#6366f1"
_TEAL = "#14b8a6"
_AMBER = "#f59e0b"
_GREEN = "#22c55e"

# Sample monthly volumes for the projection table (from the measured cost per ticket).
_PROJECTION_VOLUMES = (1_000, 10_000, 50_000, 100_000)


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------
def _usd(x: float, decimals: int | None = None) -> str:
    """USD with adaptive precision (tiny amounts get more digits)."""
    if decimals is None:
        a = abs(x)
        if a == 0:
            decimals = 2
        elif a < 0.001:
            decimals = 6
        elif a < 1:
            decimals = 4
        else:
            decimals = 2
    return f"${x:,.{decimals}f}"


def _int(n: float) -> str:
    return f"{int(round(n)):,}"


def _pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}%"


def _esc(s: str) -> str:
    return html.escape(str(s))


# ---------------------------------------------------------------------------
# Visual components (embedded SVG/HTML)
# ---------------------------------------------------------------------------
def _kpi_card(label: str, value: str, sub: str = "", accent: str = _INDIGO, badge: str = "") -> str:
    badge_html = f'<span class="badge" style="--c:{accent}">{_esc(badge)}</span>' if badge else ""
    sub_html = f'<div class="kpi-sub">{_esc(sub)}</div>' if sub else ""
    return (
        f'<div class="kpi" style="--c:{accent}">'
        f'<div class="kpi-label">{_esc(label)}{badge_html}</div>'
        f'<div class="kpi-value">{_esc(value)}</div>'
        f"{sub_html}</div>"
    )


def _stacked_bar(segments: list[tuple[str, float, str]]) -> str:
    """A stacked horizontal bar. segments = [(label, value, color), ...]."""
    total = sum(max(v, 0) for _, v, _ in segments) or 1.0
    bars = "".join(
        f'<div class="seg" style="width:{max(v, 0) / total * 100:.4f}%;background:{c}" '
        f'title="{_esc(label)}: {_int(v)}"></div>'
        for label, v, c in segments
    )
    legend = "".join(
        f'<div class="lg-item"><span class="dot" style="background:{c}"></span>'
        f'<span class="lg-label">{_esc(label)}</span>'
        f'<span class="lg-val">{_int(v)} · {v / total * 100:.1f}%</span></div>'
        for label, v, c in segments
    )
    return f'<div class="bar">{bars}</div><div class="legend">{legend}</div>'


def _donut(parts: list[tuple[str, float, str]], center_top: str, center_bot: str) -> str:
    """An SVG donut for cost composition. parts = [(label, value, color), ...]."""
    total = sum(max(v, 0) for _, v, _ in parts) or 1.0
    r = 52.0
    circ = 2 * 3.141592653589793 * r
    offset = 0.0
    rings = ""
    for _, v, c in parts:
        frac = max(v, 0) / total
        dash = frac * circ
        rings += (
            f'<circle cx="70" cy="70" r="{r}" fill="none" stroke="{c}" stroke-width="20" '
            f'stroke-dasharray="{dash:.3f} {circ - dash:.3f}" stroke-dashoffset="{-offset:.3f}" '
            f'transform="rotate(-90 70 70)" />'
        )
        offset += dash
    legend = "".join(
        f'<div class="lg-item"><span class="dot" style="background:{c}"></span>'
        f'<span class="lg-label">{_esc(label)}</span>'
        f'<span class="lg-val">{_usd(v)} · {v / total * 100:.1f}%</span></div>'
        for label, v, c in parts
    )
    return (
        '<div class="donut-wrap">'
        f'<svg viewBox="0 0 140 140" class="donut">{rings}'
        f'<text x="70" y="64" class="d-top">{_esc(center_top)}</text>'
        f'<text x="70" y="84" class="d-bot">{_esc(center_bot)}</text></svg>'
        f'<div class="legend">{legend}</div></div>'
    )


def _table(headers: list[str], rows: list[list[str]], right_from: int = 1) -> str:
    head = "".join(
        f'<th class="{"r" if i >= right_from else "l"}">{_esc(h)}</th>' for i, h in enumerate(headers)
    )
    body = ""
    for row in rows:
        cells = "".join(
            f'<td class="{"r" if i >= right_from else "l"}">{c}</td>' for i, c in enumerate(row)
        )
        body += f"<tr>{cells}</tr>"
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_CSS = """
:root{
  --bg:#0b1020; --bg2:#0f172a; --card:#141b2e; --card2:#111827;
  --line:#243049; --ink:#e5edff; --mut:#8aa0c6; --mut2:#64748b;
}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 600px at 20% -10%,#15213b 0%,var(--bg) 55%);
  color:var(--ink);font-family:'Segoe UI',Roboto,Helvetica,Arial,system-ui,sans-serif;
  -webkit-font-smoothing:antialiased;padding:36px 20px}
.wrap{max-width:1120px;margin:0 auto}
.head{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;
  border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:26px}
.h-title{font-size:27px;font-weight:800;letter-spacing:.2px;margin:0}
.h-grad{background:linear-gradient(90deg,#818cf8,#2dd4bf);-webkit-background-clip:text;
  background-clip:text;color:transparent}
.h-sub{color:var(--mut);font-size:13.5px;margin-top:8px;line-height:1.7}
.h-sub b{color:var(--ink);font-weight:600}
.pill{display:inline-block;padding:3px 10px;border:1px solid var(--line);border-radius:999px;
  font-size:11.5px;color:var(--mut);margin-right:6px}
.grid{display:grid;gap:16px}
.k4{grid-template-columns:repeat(4,1fr)}
.c2{grid-template-columns:1.35fr 1fr}
.c2b{grid-template-columns:1fr 1fr}
.kpi{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);
  border-radius:16px;padding:18px 18px 16px;position:relative;overflow:hidden}
.kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--c)}
.kpi-label{color:var(--mut);font-size:12.5px;font-weight:600;display:flex;align-items:center;
  justify-content:space-between;gap:8px;text-transform:uppercase;letter-spacing:.4px}
.kpi-value{font-size:30px;font-weight:800;margin-top:10px;letter-spacing:.3px}
.kpi-sub{color:var(--mut2);font-size:12px;margin-top:6px}
.badge{font-size:10.5px;font-weight:700;color:var(--c);border:1px solid var(--c);
  border-radius:999px;padding:1px 8px;text-transform:none;letter-spacing:0;
  background:color-mix(in srgb,var(--c) 14%,transparent)}
.panel{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);
  border-radius:16px;padding:20px}
.p-title{font-size:14px;font-weight:700;margin:0 0 16px;display:flex;align-items:center;gap:9px}
.p-title .d{width:9px;height:9px;border-radius:3px}
.bar{display:flex;height:30px;border-radius:9px;overflow:hidden;border:1px solid var(--line)}
.seg{height:100%}
.legend{margin-top:16px;display:flex;flex-direction:column;gap:11px}
.lg-item{display:flex;align-items:center;gap:10px;font-size:13px}
.dot{width:11px;height:11px;border-radius:3px;flex:none}
.lg-label{color:var(--mut)}
.lg-val{margin-left:auto;font-variant-numeric:tabular-nums;font-weight:600}
.donut-wrap{display:flex;gap:20px;align-items:center}
.donut{width:140px;height:140px;flex:none}
.d-top{fill:var(--ink);font-size:17px;font-weight:800;text-anchor:middle}
.d-bot{fill:var(--mut);font-size:9.5px;text-anchor:middle;text-transform:uppercase;letter-spacing:1px}
.donut-wrap .legend{flex:1}
.save{background:linear-gradient(135deg,rgba(34,197,94,.16),rgba(20,184,166,.10));
  border:1px solid rgba(34,197,94,.35)}
.save .kpi-value{color:#4ade80}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:10px 12px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
th{color:var(--mut);font-weight:600;font-size:11.5px;text-transform:uppercase;letter-spacing:.4px}
td.l,th.l{text-align:left}
td.r,th.r{text-align:right}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:rgba(255,255,255,.02)}
.tot td{font-weight:800;border-top:1px solid var(--line)}
.mt{margin-top:16px}
.foot{margin-top:26px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--mut2);font-size:11.5px;line-height:1.8}
.foot b{color:var(--mut)}
@media print{body{background:#fff;color:#0f172a;padding:0}
  .kpi,.panel{break-inside:avoid}}
"""


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def render_html(b: CostBreakdown, *, dataset_name: str = "", title: str = "Token & Cost Report") -> str:
    p = b.pricing
    model = b.model or "—"

    # --- KPI row ---
    savings_badge = f"−{_pct(b.cache_savings_pct, 0)}" if b.cache_savings > 0 else ""
    kpis = "".join(
        [
            _kpi_card("Total cost", _usd(b.cost_total), f"over {_int(b.n_tickets)} tickets", _INDIGO),
            _kpi_card("Cost / ticket", _usd(b.cost_per_ticket), f"{_int(b.tokens_per_ticket)} tokens/ticket", _TEAL),
            _kpi_card("Total tokens", _int(b.total_tokens),
                      f"in {_int(b.prompt_tokens)} · out {_int(b.completion_tokens)}", _AMBER),
            _kpi_card("Cache hit rate", _pct(b.cache_hit_rate), f"of {_int(b.prompt_tokens)} input tokens",
                      _GREEN, badge=savings_badge),
        ]
    )

    # --- Token composition panel ---
    token_bar = _stacked_bar(
        [
            ("Cached input (hit)", b.cache_hit_tokens, _GREEN),
            ("Uncached input (miss)", b.cache_miss_tokens, _AMBER),
            ("Output (completion)", b.completion_tokens, _INDIGO),
        ]
    )
    token_panel = (
        '<div class="panel">'
        f'<p class="p-title"><span class="d" style="background:{_TEAL}"></span>Token composition</p>'
        f"{token_bar}</div>"
    )

    # --- Cost composition panel (donut) ---
    cost_donut = _donut(
        [("Input cost", b.cost_input, _AMBER), ("Output cost", b.cost_output, _INDIGO)],
        _usd(b.cost_total),
        "total",
    )
    cost_panel = (
        '<div class="panel">'
        f'<p class="p-title"><span class="d" style="background:{_INDIGO}"></span>Cost composition</p>'
        f"{cost_donut}</div>"
    )

    # --- Cost breakdown table ---
    cost_rows = [
        ["Cached input", _int(b.cache_hit_tokens), _usd(p.cache_hit_per_m), _usd(b.cost_cache_hit)],
        ["Uncached input", _int(b.cache_miss_tokens), _usd(p.input_per_m), _usd(b.cost_cache_miss)],
        ["Output", _int(b.completion_tokens), _usd(p.output_per_m), _usd(b.cost_output)],
    ]
    cost_table = _table(["Component", "Tokens", "Rate / 1M", "Cost"], cost_rows, right_from=1)
    cost_table += (
        '<table><tbody><tr class="tot"><td class="l">Total</td>'
        f'<td class="r">{_int(b.total_tokens)}</td><td class="r"></td>'
        f'<td class="r">{_usd(b.cost_total)}</td></tr></tbody></table>'
    )
    breakdown_panel = (
        '<div class="panel">'
        f'<p class="p-title"><span class="d" style="background:{_AMBER}"></span>Cost breakdown</p>'
        f"{cost_table}</div>"
    )

    # --- Cache savings panel ---
    savings_panel = _kpi_card(
        "Saved by prompt caching",
        _usd(b.cache_savings),
        f"vs. {_usd(b.cost_without_cache)} no-cache baseline ({_pct(b.cache_savings_pct)})",
        _GREEN,
    ).replace('class="kpi"', 'class="kpi save"')

    # --- Unit economics ---
    unit_rows = [
        ["Cost per ticket", _usd(b.cost_per_ticket)],
        ["Cost per API call", _usd(b.cost_per_call)],
        ["Tokens per ticket", _int(b.tokens_per_ticket)],
        ["API calls per ticket", f"{b.calls_per_ticket:.2f}"],
        ["Avg latency", f"{b.latency_ms_avg:,.0f} ms"],
    ]
    unit_panel = (
        '<div class="panel">'
        f'<p class="p-title"><span class="d" style="background:{_TEAL}"></span>Unit economics</p>'
        f"{_table(['Metric', 'Value'], unit_rows, right_from=1)}</div>"
    )

    # --- Monthly cost projection (from the measured cost per ticket) ---
    proj_rows = [
        [f"{_int(v)} tickets / mo", _usd(b.project(v)), _usd(b.project(v) * 12)]
        for v in _PROJECTION_VOLUMES
    ]
    proj_panel = (
        '<div class="panel">'
        f'<p class="p-title"><span class="d" style="background:{_INDIGO}"></span>'
        "Projected cost at scale</p>"
        f"{_table(['Monthly volume', 'Monthly', 'Annual'], proj_rows, right_from=1)}"
        '<div class="kpi-sub mt">Extrapolated from the measured cost-per-ticket above '
        "(assumes a comparable cache-hit rate).</div></div>"
    )

    # --- Footer ---
    foot = (
        f"<b>Pricing assumptions</b> (USD per 1M tokens): "
        f"cached input ${p.cache_hit_per_m} · uncached input ${p.input_per_m} · output ${p.output_per_m}. "
        "Verify against current DeepSeek pricing. &nbsp;·&nbsp; "
        "Figures are derived from real measured token usage (no estimates). &nbsp;·&nbsp; "
        f"Generated {date.today().isoformat()}."
    )

    sub = f'<b>Model:</b> {_esc(model)} &nbsp;·&nbsp; <b>Tickets:</b> {_int(b.n_tickets)} ' \
          f"&nbsp;·&nbsp; <b>API calls:</b> {_int(b.n_calls)}"
    if dataset_name:
        sub += f" &nbsp;·&nbsp; <b>Source:</b> {_esc(dataset_name)}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{_esc(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <div>
      <h1 class="h-title"><span class="h-grad">Token &amp; Cost Report</span></h1>
      <div class="h-sub">Ticket Triage Chatbot · DeepSeek<br/>{sub}</div>
    </div>
    <div style="text-align:right">
      <span class="pill">DeepSeek API</span>
      <span class="pill">3-tier pricing</span>
      <div class="h-sub" style="margin-top:10px">{date.today().isoformat()}</div>
    </div>
  </div>

  <div class="grid k4">{kpis}</div>

  <div class="grid c2 mt">{token_panel}{cost_panel}</div>

  <div class="grid c2b mt">{breakdown_panel}<div class="grid" style="gap:16px">{savings_panel}{unit_panel}</div></div>

  <div class="mt">{proj_panel}</div>

  <div class="foot">{foot}</div>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="HTML cost/token report from real data.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("data_path", nargs="?", help="raw dataset (JSONL) for a real model run")
    src.add_argument("--from-log", metavar="PATH", help="aggregate from logs/interactions.jsonl (no API)")

    ap.add_argument("--out", default="cost_report.html", help="HTML output path")
    ap.add_argument("--frac", type=float, default=None, help="sample fraction per combo (e.g. 0.2)")
    ap.add_argument("--balanced", type=int, default=None, help="at most N tickets per combo")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--price-in", type=float, default=Pricing().input_per_m)
    ap.add_argument("--price-cache", type=float, default=Pricing().cache_hit_per_m)
    ap.add_argument("--price-out", type=float, default=Pricing().output_per_m)
    args = ap.parse_args()

    pricing = Pricing(input_per_m=args.price_in, cache_hit_per_m=args.price_cache, output_per_m=args.price_out)

    if args.from_log:
        breakdown = aggregate_log(args.from_log, pricing=pricing)
        dataset_name = Path(args.from_log).name
    else:
        # Real model run (requires an API key) — lazy import so the log path stays lightweight.
        from scripts.eval_incdb import run_evaluation

        res = run_evaluation(
            args.data_path, limit=args.limit, balanced=args.balanced, frac=args.frac,
            seed=args.seed, workers=args.workers,
        )
        breakdown = breakdown_from_eval(res, pricing=pricing)
        dataset_name = Path(args.data_path).name

    out = Path(args.out)
    out.write_text(render_html(breakdown, dataset_name=dataset_name), encoding="utf-8")
    print(f"saved: {out}")
    print(f"  model={breakdown.model}  tickets={breakdown.n_tickets}  "
          f"cost=${breakdown.cost_total:.4f}  cost/ticket=${breakdown.cost_per_ticket:.6f}")


if __name__ == "__main__":
    main()
