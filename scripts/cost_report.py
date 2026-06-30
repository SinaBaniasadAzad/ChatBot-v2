"""
گزارشِ HTMLِ هزینه و توکن — تک‌فایلِ مستقل، آمادهٔ ارائه به مدیریت.

از دادهٔ **واقعی** تغذیه می‌شود (نه ماشین‌حساب فرضی). دو منبع:
  ۱) لاگِ تولید (بدونِ نیاز به API):
        python -m scripts.cost_report --from-log logs/interactions.jsonl --out cost_report.html
  ۲) اجرای واقعیِ مدل روی دیتاست (نیازمندِ DEEPSEEK_API_KEY):
        python -m scripts.cost_report tests/Ticketing_DB.jsonl --frac 0.2 --workers 6 \
            --out cost_report.html

اجزای بصری از دیزاین‌سیستمِ مشترک می‌آیند (`src/reporting/html_ui.py`) تا با گزارشِ
ترکیبیِ عملکرد+هزینه دقیقاً هم‌زبان باشد. تابعِ `cost_body()` همان بدنهٔ هزینه را
برمی‌گرداند تا در گزارشِ ترکیبی بازاستفاده شود.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reporting import html_ui as ui  # noqa: E402
from src.reporting.cost import (  # noqa: E402
    CostBreakdown,
    Pricing,
    aggregate_log,
    breakdown_from_eval,
)

# حجم‌های ماهانهٔ نمونه برای جدولِ برون‌یابی (از هزینهٔ اندازه‌گیری‌شدهٔ هر تیکت).
_PROJECTION_VOLUMES = (1_000, 10_000, 50_000, 100_000)


def cost_body(b: CostBreakdown) -> str:
    """بدنهٔ HTMLِ بخشِ هزینه/توکن (بدونِ هدر/پاورقیِ صفحه) — برای بازاستفاده."""
    p = b.pricing

    savings_badge = f"−{ui.pct(b.cache_savings_pct, 0)}" if b.cache_savings > 0 else ""
    kpis = "".join([
        ui.kpi_card("Total cost", ui.usd(b.cost_total), f"over {ui.intc(b.n_tickets)} tickets", ui.INDIGO),
        ui.kpi_card("Cost / ticket", ui.usd(b.cost_per_ticket),
                    f"{ui.intc(b.tokens_per_ticket)} tokens/ticket", ui.TEAL),
        ui.kpi_card("Total tokens", ui.intc(b.total_tokens),
                    f"in {ui.intc(b.prompt_tokens)} · out {ui.intc(b.completion_tokens)}", ui.AMBER),
        ui.kpi_card("Cache hit rate", ui.pct(b.cache_hit_rate),
                    f"of {ui.intc(b.prompt_tokens)} input tokens", ui.GREEN,
                    badge=savings_badge, badge_color=ui.GREEN),
    ])

    token_panel = ui.panel("Token composition", ui.stacked_bar([
        ("Cached input (hit)", b.cache_hit_tokens, ui.GREEN),
        ("Uncached input (miss)", b.cache_miss_tokens, ui.AMBER),
        ("Output (completion)", b.completion_tokens, ui.INDIGO),
    ]), accent=ui.TEAL)

    cost_panel = ui.panel("Cost composition", ui.donut(
        [("Input cost", b.cost_input, ui.AMBER), ("Output cost", b.cost_output, ui.INDIGO)],
        ui.usd(b.cost_total), "total",
    ), accent=ui.INDIGO)

    breakdown_panel = ui.panel("Cost breakdown", ui.table(
        ["Component", "Tokens", "Rate / 1M", "Cost"],
        [
            ["Cached input", ui.intc(b.cache_hit_tokens), ui.usd(p.cache_hit_per_m), ui.usd(b.cost_cache_hit)],
            ["Uncached input", ui.intc(b.cache_miss_tokens), ui.usd(p.input_per_m), ui.usd(b.cost_cache_miss)],
            ["Output", ui.intc(b.completion_tokens), ui.usd(p.output_per_m), ui.usd(b.cost_output)],
        ],
        right_from=1,
        total_row=["Total", ui.intc(b.total_tokens), "", ui.usd(b.cost_total)],
    ), accent=ui.AMBER)

    savings_card = ui.kpi_card(
        "Saved by prompt caching", ui.usd(b.cache_savings),
        f"vs. {ui.usd(b.cost_without_cache)} no-cache baseline ({ui.pct(b.cache_savings_pct)})",
        ui.GREEN, extra_class="save",
    )
    unit_panel = ui.panel("Unit economics", ui.table(
        ["Metric", "Value"],
        [
            ["Cost per ticket", ui.usd(b.cost_per_ticket)],
            ["Cost per API call", ui.usd(b.cost_per_call)],
            ["Tokens per ticket", ui.intc(b.tokens_per_ticket)],
            ["API calls per ticket", f"{b.calls_per_ticket:.2f}"],
            ["Avg latency", f"{b.latency_ms_avg:,.0f} ms"],
        ],
        right_from=1,
    ), accent=ui.TEAL)

    proj_rows = [[f"{ui.intc(v)} tickets / mo", ui.usd(b.project(v)), ui.usd(b.project(v) * 12)]
                 for v in _PROJECTION_VOLUMES]
    proj_panel = ui.panel("Projected cost at scale",
                          ui.table(["Monthly volume", "Monthly", "Annual"], proj_rows, right_from=1)
                          + '<div class="kpi-sub mt">Extrapolated from the measured cost-per-ticket '
                            "above (assumes a comparable cache-hit rate).</div>", accent=ui.INDIGO)

    return (
        f'<div class="grid k4">{kpis}</div>'
        f'<div class="grid c2 mt">{token_panel}{cost_panel}</div>'
        f'<div class="grid c2b mt">{breakdown_panel}'
        f'<div class="grid" style="gap:16px">{savings_card}{unit_panel}</div></div>'
        f'<div class="mt">{proj_panel}</div>'
    )


def cost_footnote(b: CostBreakdown) -> str:
    p = b.pricing
    return (
        f"<b>Pricing assumptions</b> (USD per 1M tokens): cached input ${p.cache_hit_per_m} · "
        f"uncached input ${p.input_per_m} · output ${p.output_per_m}. "
        "Verify against current DeepSeek pricing. &nbsp;·&nbsp; "
        "Figures are derived from real measured token usage (no estimates)."
    )


def render_html(b: CostBreakdown, *, dataset_name: str = "", title: str = "Token & Cost Report") -> str:
    sub = (f"<b>Model:</b> {ui.esc(b.model or '—')} &nbsp;·&nbsp; "
           f"<b>Tickets:</b> {ui.intc(b.n_tickets)} &nbsp;·&nbsp; "
           f"<b>API calls:</b> {ui.intc(b.n_calls)}")
    if dataset_name:
        sub += f" &nbsp;·&nbsp; <b>Source:</b> {ui.esc(dataset_name)}"
    head = ui.header(
        title_html='<span class="h-grad">Token &amp; Cost Report</span>',
        subtitle_html=f"Ticket Triage Chatbot · DeepSeek<br/>{sub}",
        pills=["DeepSeek API", "3-tier pricing"],
    )
    body = cost_body(b) + f'<div class="foot">{cost_footnote(b)}</div>'
    return ui.page(title=title, header_html=head, body_html=body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="گزارشِ HTMLِ هزینه/توکن از دادهٔ واقعی.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("data_path", nargs="?", help="دیتاستِ خام (JSONL) برای اجرای واقعیِ مدل")
    src.add_argument("--from-log", metavar="PATH", help="تجمیع از logs/interactions.jsonl (بدون API)")

    ap.add_argument("--out", default="cost_report.html", help="مسیرِ خروجیِ HTML")
    ap.add_argument("--frac", type=float, default=None, help="نسبتِ نمونه از هر ترکیب (مثلاً 0.2)")
    ap.add_argument("--balanced", type=int, default=None, help="حداکثر N تیکت در هر ترکیب")
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
        from scripts.eval_incdb import run_evaluation  # importِ تنبل (نیازمندِ API)

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
