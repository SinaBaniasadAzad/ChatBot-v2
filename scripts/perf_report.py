"""
گزارشِ HTMLِ دقتِ مدل — «Model Accuracy Report».

یک سندِ خودبسنده و آمادهٔ ارائه به مدیریت که نشان می‌دهد مدل **چقدر خوب** تیکت‌ها را
دسته‌بندی می‌کند. بخش‌ها:
  ۱) خلاصهٔ مدیریتی (دقتِ کل + دقتِ هر لایه)
  ۲) آمادگیِ عملیاتی (auto در برابر needs-review — ترجمهٔ دقت به ارزشِ تجاری)
  ۳) عملکردِ هر کلاس (Precision / Recall / F1)
  ۴) ماتریسِ درهم‌ریختگیِ بصری (heatmap)

اعداد از موتورِ واحدِ `src/reporting/metrics.py` می‌آیند؛ این فایل فقط «نمایش» می‌دهد.
هزینه/توکن یک گزارشِ جداست (`scripts/cost_report.py`).

اجرا (نیازمندِ DEEPSEEK_API_KEY چون به اجرای واقعیِ مدل نیاز دارد):
    python -m scripts.perf_report tests/Ticketing_DB.jsonl --frac 0.2 --workers 6 \
        --out accuracy_report.html
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reporting import html_ui as ui  # noqa: E402
from src.reporting import metrics as M  # noqa: E402


# ---------------------------------------------------------------------------
# بخش ۱ — خلاصهٔ مدیریتی
# ---------------------------------------------------------------------------
def _summary_section(m: M.PerfMetrics) -> str:
    acc = m.overall_accuracy
    color = ui.grade(acc)
    overall = ui.kpi_card(
        "Overall accuracy", ui.pct(acc),
        f"both layers correct · {m.overall_correct}/{m.overall_total}", color,
        extra_class="hero",
    )
    cards = [overall]
    for L in m.layers:
        cards.append(ui.kpi_card(
            f"{L.name} accuracy", ui.pct(L.accuracy),
            f"{L.correct}/{L.total} · macro-F1 {ui.pct(L.macro_f1)}", ui.grade(L.accuracy),
        ))
    r = m.readiness
    if r.has_data:
        cards.append(ui.kpi_card(
            "Auto-classified", ui.pct(r.auto_coverage),
            f"at {ui.pct(r.auto_accuracy)} accuracy", ui.TEAL,
        ))
    else:
        cards.append(ui.kpi_card("Avg latency", f"{m.latency_ms_avg:,.0f} ms", "per ticket", ui.SLATE))

    return ui.section("Executive summary", f'<div class="grid k4">{"".join(cards)}</div>',
                      index="1", accent=color)


# ---------------------------------------------------------------------------
# بخش ۲ — آمادگیِ عملیاتی
# ---------------------------------------------------------------------------
def _readiness_section(m: M.PerfMetrics) -> str:
    r = m.readiness
    if not r.has_data:
        return ""
    narrative = (
        f'<div class="kpi-sub" style="font-size:13.5px;margin:0 0 14px">'
        f'The model can <b style="color:#4ade80">auto-resolve {ui.pct(r.auto_coverage)}</b> of tickets '
        f'at <b>{ui.pct(r.auto_accuracy)}</b> accuracy; the remaining '
        f'<b style="color:#fbbf24">{ui.pct(r.review_share)}</b> are routed to human review.</div>'
    )
    split = ui.stacked_bar([
        ("Auto-classified", r.auto_total, ui.GREEN),
        ("Needs human review", r.review_total, ui.AMBER),
    ])
    workload = ui.panel("Workload split", narrative + split, accent=ui.GREEN)

    tiles = "".join([
        ui.kpi_card("Auto coverage", ui.pct(r.auto_coverage), "handled without a human", ui.GREEN),
        ui.kpi_card("Auto accuracy", ui.pct(r.auto_accuracy),
                    f"{r.auto_correct}/{r.auto_total} correct", ui.grade(r.auto_accuracy)),
        ui.kpi_card("Review share", ui.pct(r.review_share), "routed to humans", ui.AMBER),
        ui.kpi_card("Review accuracy", ui.pct(r.review_accuracy),
                    f"{r.review_correct}/{r.review_total} correct", ui.SLATE),
    ])
    inner = workload + f'<div class="grid k4 mt">{tiles}</div>'
    return ui.section("Operational readiness", inner, index="2", accent=ui.GREEN)


# ---------------------------------------------------------------------------
# بخش ۳ — عملکردِ هر کلاس (Precision / Recall / F1)
# ---------------------------------------------------------------------------
def _per_class_section(m: M.PerfMetrics) -> str:
    panels = []
    for L in m.layers:
        rows = []
        for c in L.classes:
            rows.append([
                c.name, ui.pct(c.precision), ui.pct(c.recall),
                ui.progress(c.f1, ui.grade(c.f1), ui.pct(c.f1)), ui.intc(c.support),
            ])
        total_row = [
            "Macro avg", ui.pct(L.macro_precision), ui.pct(L.macro_recall),
            ui.progress(L.macro_f1, ui.grade(L.macro_f1), ui.pct(L.macro_f1)), ui.intc(L.total),
        ]
        tbl = ui.table(["Class", "Precision", "Recall", "F1", "Support"], rows,
                       right_from=1, total_row=total_row)
        panels.append(ui.panel(f"{L.name}  ·  per-class metrics", tbl, accent=ui.VIOLET))
    return ui.section("Per-class performance", '<div class="grid mt" style="gap:16px">'
                      + "".join(panels) + "</div>", index="3", accent=ui.VIOLET)


# ---------------------------------------------------------------------------
# بخش ۴ — ماتریسِ درهم‌ریختگی
# ---------------------------------------------------------------------------
def _confusion_section(m: M.PerfMetrics) -> str:
    panels = [
        ui.panel(f"{L.name}", ui.heatmap(L.label_ids, L.names, L.confusion), accent=ui.INDIGO)
        for L in m.layers
    ]
    return ui.section("Confusion matrices", f'<div class="grid c2b">{"".join(panels)}</div>',
                      index="4", accent=ui.INDIGO)


# ---------------------------------------------------------------------------
# سرهم‌بندیِ گزارش
# ---------------------------------------------------------------------------
def render_report(res: dict, *, dataset_name: str = "") -> str:
    m = M.from_eval(res)

    sub = (f"<b>Model:</b> {ui.esc(m.model or '—')} &nbsp;·&nbsp; "
           f"<b>Tickets:</b> {ui.intc(m.n)}")
    if dataset_name:
        sub += f" &nbsp;·&nbsp; <b>Dataset:</b> {ui.esc(dataset_name)}"
    head = ui.header(
        title_html='<span class="h-grad">Model Accuracy Report</span>',
        subtitle_html=f"Ticket Triage Chatbot · Executive report<br/>{sub}",
        pills=["DeepSeek", "single-shot eval"],
    )

    body = (
        _summary_section(m)
        + _readiness_section(m)
        + _per_class_section(m)
        + _confusion_section(m)
    )
    foot = (
        f"<b>Methodology</b>: single-shot evaluation (no clarifying questions) · "
        f"{ui.intc(m.n)} tickets · avg latency {m.latency_ms_avg:,.0f} ms/ticket · "
        f"generated {date.today().isoformat()}."
    )
    body += f'<div class="foot">{foot}</div>'
    return ui.page(title="Model Accuracy Report", header_html=head, body_html=body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="گزارشِ HTMLِ دقتِ مدل (نیازمندِ API).")
    ap.add_argument("data_path", help="دیتاستِ خام (JSONL)")
    ap.add_argument("--out", default="accuracy_report.html")
    ap.add_argument("--frac", type=float, default=None, help="نسبتِ نمونه از هر ترکیب (مثلاً 0.2)")
    ap.add_argument("--balanced", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--errors", default=None, help="ذخیرهٔ تیکت‌های اشتباه (JSONL)")
    args = ap.parse_args()

    from scripts.eval_incdb import run_evaluation  # importِ تنبل (نیازمندِ API)

    res = run_evaluation(
        args.data_path, limit=args.limit, balanced=args.balanced, frac=args.frac,
        seed=args.seed, workers=args.workers, errors_out=args.errors,
    )
    out = Path(args.out)
    out.write_text(render_report(res, dataset_name=Path(args.data_path).name), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
