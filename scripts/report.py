"""
داشبوردِ دقتِ حرفه‌ای (برای ارائه) — روی نتایجِ eval_incdb.run_evaluation.
121212
نمایش:
  • کارت‌های KPI: دقتِ کل (هر دو لایه) + دقتِ هر لایه
  • نمودارِ recall هر کلاس (Incident / Service Request / ERP / Staff)
  • Confusion matrix هر لایه (heatmap)
  • نوار پایین: مدل، تعداد نمونه، توکن/هزینه، تاریخ

استفاده روی Kaggle (یک سلول):
    from scripts.report import evaluate_and_report
    res, fig = evaluate_and_report(
        "data/INC_DB.jsonl", balanced=75, workers=6,
        save_path="/kaggle/working/accuracy_report.png",
    )

خروجی هم inline نمایش داده می‌شود، هم PNGِ باکیفیت برای اسلاید ذخیره می‌شود.
"""
from __future__ import annotations

from datetime import date

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from scripts.eval_incdb import print_text_report, run_evaluation
from src.reporting.cost import Pricing, breakdown_from_eval

# پالت
_INK = "#0f172a"
_MUTE = "#64748b"
_GRID = "#e2e8f0"
_LAYER_COLORS = ["#4f46e5", "#0d9488", "#b45309", "#9333ea"]  # برای لایه‌های ۱،۲،…

# رنگِ اجزای هزینه (هم‌خوان با گزارشِ HTML)
_C_HIT = "#0d9488"     # ورودیِ cache-hit
_C_MISS = "#d97706"    # ورودیِ cache-miss
_C_OUT = "#4f46e5"     # خروجی/completion
_C_SAVE = "#16a34a"    # صرفه‌جویی


def _grade(v: float) -> str:
    """رنگِ نمره: خوب/متوسط/ضعیف."""
    if v >= 0.90:
        return "#0d9488"
    if v >= 0.80:
        return "#d97706"
    return "#dc2626"


def _tint(hex_color: str, f: float = 0.12) -> tuple:
    """نسخهٔ روشن (آمیخته با سفید)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


def _draw_kpis(ax, res: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ov = res["overall"]
    tiles = [("Overall accuracy\n(both layers correct)", ov["accuracy"], f"{ov['correct']} / {ov['total']}", True)]
    for i, L in enumerate(res["layers"], 1):
        tiles.append((f"{L['name']}  ·  Layer {i}", L["accuracy"], f"{L['correct']} / {L['total']}", False))

    n = len(tiles)
    gap = 0.025
    w = (1 - gap * (n - 1)) / n
    for i, (label, value, sub, primary) in enumerate(tiles):
        x = i * (w + gap)
        accent = _grade(value)
        if primary:  # سرتیترِ خنثیٰ و مقتدر (سرمه‌ای) + نقطهٔ رنگیِ نمره
            face, edge, lw = "#1f2937", "#1f2937", 0
            num_color, lbl_color, sub_color = "white", "white", "#cbd5e1"
        else:  # کارتِ هر لایه: ته‌رنگِ نمره + عددِ هم‌رنگ
            face, edge, lw = _tint(accent, 0.14), accent, 1.4
            num_color, lbl_color, sub_color = accent, _INK, _MUTE
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.06), w, 0.88,
                boxstyle="round,pad=0,rounding_size=0.035",
                linewidth=lw, edgecolor=edge, facecolor=face, mutation_aspect=0.5,
            )
        )
        ax.text(x + w / 2, 0.78, label, ha="center", va="center", fontsize=11,
                color=lbl_color, fontweight="bold", linespacing=1.25)
        ax.text(x + w / 2, 0.45, f"{value*100:.1f}%", ha="center", va="center",
                fontsize=33, color=num_color, fontweight="bold")
        if primary:  # نقطهٔ رنگیِ نمره روی کارتِ سرمه‌ای
            ax.scatter([x + 0.06], [0.78], s=90, color=accent, zorder=5)
        ax.text(x + w / 2, 0.16, sub, ha="center", va="center", fontsize=11, color=sub_color)


def _draw_recall(ax, res: dict) -> None:
    rows = []  # (class_name, recall, correct, total, color)
    for li, L in enumerate(res["layers"]):
        color = _LAYER_COLORS[li % len(_LAYER_COLORS)]
        for c in L["classes"]:
            rows.append((c["name"], c["recall"], c["correct"], c["total"], color, L["name"]))
    rows.reverse()  # اولین کلاس بالا

    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[4] for r in rows], height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11.5, color=_INK)
    for yi, r in zip(y, rows):
        ax.text(min(r[1] + 0.015, 1.0), yi, f"{r[1]*100:.1f}%  ({r[2]}/{r[3]})",
                va="center", ha="left", fontsize=10.5, color=_INK, fontweight="bold")
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10, color=_MUTE)
    ax.set_title("Per-class recall  ·  share of each true class predicted correctly",
                 fontsize=13, fontweight="bold", color=_INK, loc="left", pad=10)
    ax.xaxis.grid(True, color=_GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(length=0)

    # افسانهٔ لایه‌ها
    handles = [plt.Rectangle((0, 0), 1, 1, color=_LAYER_COLORS[i % len(_LAYER_COLORS)]) for i in range(len(res["layers"]))]
    ax.legend(
    handles,
    [f"{L['name']} (layer {L['id']})" for L in res["layers"]],
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    frameon=False,
    fontsize=9.5
)


def _draw_confusion(ax, L: dict) -> None:
    ids = L["label_ids"]
    name = {c["id"]: c["name"] for c in L["classes"]}
    has_none = any(L["confusion"].get(t, {}).get(None, 0) for t in ids)
    cols = list(ids) + ([None] if has_none else [])
    col_labels = [name[c] if c is not None else "∅ none" for c in cols]

    M = np.array([[L["confusion"].get(tr, {}).get(p, 0) for p in cols] for tr in ids], dtype=float)
    row_sums = M.sum(axis=1, keepdims=True)
    norm = np.divide(M, row_sums, out=np.zeros_like(M), where=row_sums > 0)

    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=9.5, rotation=20, ha="right", color=_INK)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([name[t] for t in ids], fontsize=9.5, color=_INK)
    for i in range(len(ids)):
        for j in range(len(cols)):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if norm[i, j] > 0.5 else _INK, fontweight="bold")
    ax.set_title(f"Confusion — {L['name']}", fontsize=12, fontweight="bold", color=_INK, pad=8)
    ax.set_xlabel("Predicted", fontsize=10, color=_MUTE)
    ax.set_ylabel("True", fontsize=10, color=_MUTE)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def _draw_stack(ax, title: str, segs, fmt) -> None:
    """نوارِ افقیِ انباشتهٔ عمومی با افسانهٔ خطی. segs = [(name, value, color), ...]."""
    total = sum(max(v, 0) for _, v, _ in segs) or 1.0
    left = 0.0
    for _, val, color in segs:
        ax.barh(0, val, left=left, color=color, height=0.8, zorder=3)
        left += val
    ax.set_xlim(0, total)
    ax.set_ylim(-2.95, 0.6)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", color=_INK, loc="left", pad=10)
    # افسانه — هر آیتم در یک خطِ جدا (بدونِ هم‌پوشانی)
    for i, (name, val, color) in enumerate(segs):
        y = -0.95 - i * 0.62
        ax.scatter([total * 0.006], [y], s=70, marker="s", color=color, zorder=4)
        ax.text(total * 0.02, y, f"{name}", va="center", ha="left", fontsize=10.5, color=_MUTE)
        ax.text(total * 0.42, y, f"{fmt(val)}  ·  {val / total * 100:.1f}%",
                va="center", ha="left", fontsize=10.5, color=_INK, fontweight="bold")


def _draw_cost_tokens(ax, b) -> None:
    """نوارِ ترکیبِ توکن: cached / uncached input + output."""
    _draw_stack(ax, "Token composition  ·  cached vs uncached input + output", [
        ("Cached input", b.cache_hit_tokens, _C_HIT),
        ("Uncached input", b.cache_miss_tokens, _C_MISS),
        ("Output", b.completion_tokens, _C_OUT),
    ], fmt=lambda v: f"{v:,.0f}")


def _draw_cost_components(ax, b) -> None:
    """نوارِ ترکیبِ هزینه ($): سهمِ هر جزء از کلِ هزینه."""
    _draw_stack(ax, "Cost by component  ·  where the money goes", [
        ("Cached input", b.cost_cache_hit, _C_HIT),
        ("Uncached input", b.cost_cache_miss, _C_MISS),
        ("Output", b.cost_output, _C_OUT),
    ], fmt=lambda v: f"${v:,.4f}")


def _tiles(ax, items, ncol: int) -> None:
    """شبکهٔ کاشی‌های KPI روی یک محور. items = [(label, value, accent), ...]."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nrow = -(-len(items) // ncol)  # ceil
    gx, gy = 0.035, 0.12
    w = (1 - gx * (ncol - 1)) / ncol
    h = (1 - gy * (nrow - 1)) / nrow
    for idx, (label, val, accent) in enumerate(items):
        r, c = divmod(idx, ncol)
        x = c * (w + gx)
        y = 1 - (r + 1) * h - r * gy
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                linewidth=1.4, edgecolor=accent, facecolor=_tint(accent, 0.13), mutation_aspect=0.42,
            )
        )
        ax.text(x + w / 2, y + h * 0.70, label, ha="center", va="center",
                fontsize=10, color=_INK, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.33, val, ha="center", va="center",
                fontsize=19, color=accent, fontweight="bold")


def _header(fig, title: str, res: dict, dataset_name: str) -> None:
    fig.text(0.07, 0.952, title, fontsize=22, fontweight="bold", color=_INK)
    sub = f"Model: {res.get('model') or '—'}    ·    Tickets evaluated: {res['n']}"
    if dataset_name:
        sub += f"    ·    Dataset: {dataset_name}"
    sub += f"    ·    {date.today().isoformat()}"
    fig.text(0.07, 0.917, sub, fontsize=11.5, color=_MUTE)


def render_accuracy_dashboard(res: dict, *, dataset_name: str = ""):
    """تصویرِ مستقلِ دقت: کارت‌های KPI + recall هر کلاس + ماتریسِ درهم‌ریختگی."""
    layers = res["layers"]
    fig = plt.figure(figsize=(13.0, 12.8), facecolor="white")
    gs = fig.add_gridspec(
        3, 1, height_ratios=[0.74, 1.12, 1.20], hspace=0.46,
        left=0.07, right=0.95, top=0.885, bottom=0.11,
    )
    _header(fig, "Ticket Classification — Accuracy Report", res, dataset_name)

    _draw_kpis(fig.add_subplot(gs[0]), res)
    _draw_recall(fig.add_subplot(gs[1]), res)
    sub_gs = gs[2].subgridspec(1, len(layers), wspace=0.32)
    for i, L in enumerate(layers):
        _draw_confusion(fig.add_subplot(sub_gs[0, i]), L)

    foot = (
        "Single-shot evaluation (no clarifying questions)   ·   "
        "Overall = both layers correct   ·   recall = share of each true class predicted correctly   ·   "
        f"avg latency {res['latency_ms_avg']:.0f} ms/ticket"
    )
    fig.text(0.07, 0.035, foot, fontsize=9.5, color=_MUTE)
    return fig


def render_cost_dashboard(res: dict, *, dataset_name: str = "", prices=(0.14, 0.0028, 0.28)):
    """تصویرِ مستقلِ هزینه/توکن: کارت‌های KPI + ترکیبِ توکن + ترکیبِ هزینه."""
    b = breakdown_from_eval(res, pricing=Pricing.from_tuple(prices))
    fig = plt.figure(figsize=(13.0, 9.6), facecolor="white")
    gs = fig.add_gridspec(
        3, 1, height_ratios=[0.74, 0.62, 0.62], hspace=0.55,
        left=0.07, right=0.95, top=0.865, bottom=0.085,
    )
    _header(fig, "Ticket Classification — Token & Cost Report", res, dataset_name)

    _tiles(
        fig.add_subplot(gs[0]),
        [
            ("Total cost", f"${b.cost_total:,.2f}", _C_OUT),
            ("Cost / ticket", f"${b.cost_per_ticket:.4f}", _C_HIT),
            ("Cache-hit rate", f"{b.cache_hit_rate*100:.0f}%", _C_MISS),
            (f"Saved by cache  (−{b.cache_savings_pct*100:.0f}%)", f"${b.cache_savings:,.2f}", _C_SAVE),
        ],
        ncol=4,
    )
    _draw_cost_tokens(fig.add_subplot(gs[1]), b)
    _draw_cost_components(fig.add_subplot(gs[2]), b)

    p = b.pricing
    foot = (
        f"tokens: input {b.prompt_tokens:,} (hit {b.cache_hit_tokens:,} / miss {b.cache_miss_tokens:,}) / "
        f"output {b.completion_tokens:,}   ·   "
        f"pricing /1M: in ${p.input_per_m} · hit ${p.cache_hit_per_m} · out ${p.output_per_m} "
        f"(verify current pricing)   ·   {b.n_tickets:,} tickets"
    )
    fig.text(0.07, 0.03, foot, fontsize=9.5, color=_MUTE)
    return fig


def evaluate_and_report(
    data_path,
    *,
    limit=None,
    balanced=None,
    frac=None,
    seed=42,
    workers=4,
    out_path=None,
    errors_out=None,
    errors_xlsx=None,
    accuracy_html=None,
    cost_html=None,
    accuracy_png=None,
    cost_png=None,
    dataset_name=None,
    prices=(0.14, 0.0028, 0.28),
    show=True,
):
    """
    یک اجرای واقعی → همهٔ خروجی‌ها (بدونِ مصرفِ دوبارهٔ API). خروجی: (res, figs).
    `figs` یک dict است: {"accuracy": Figure, "cost": Figure}.

    هر دو داشبوردِ تصویری **همیشه** ساخته می‌شوند و با `show=True` (پیش‌فرض) inline
    در نوت‌بوک/Kaggle نمایش داده می‌شوند — مثلِ همیشه، مستقل از اینکه فایلی بخواهی.

    آرگومان‌های ذخیره‌سازی (هرکدام داده شود نوشته می‌شود):
      • accuracy_html : گزارشِ HTMLِ دقت (تمِ تیره؛ خلاصهٔ مدیریتی، P/R/F1، heatmap)
      • cost_html     : گزارشِ HTMLِ هزینه/توکن (تمِ تیره)
      • accuracy_png  : ذخیرهٔ داشبوردِ تصویریِ دقت
      • cost_png      : ذخیرهٔ داشبوردِ تصویریِ هزینه/توکن
      • errors_out    : تیکت‌های اشتباه + متن (JSON)
      • errors_xlsx   : همان تیکت‌های اشتباه (Excel)
      • out_path      : (اختیاری) همهٔ پیش‌بینی‌ها (JSONL)
    """
    from pathlib import Path

    # اگر فقط Excel خواسته شده، یک JSONLِ کناری هم ساخته می‌شود (منبعِ تبدیل).
    eff_errors_out = errors_out
    if errors_xlsx and not eff_errors_out:
        eff_errors_out = str(Path(errors_xlsx).with_suffix(".jsonl"))

    res = run_evaluation(
        data_path, limit=limit, balanced=balanced, frac=frac, seed=seed, workers=workers,
        out_path=out_path, errors_out=eff_errors_out,
    )
    name = dataset_name or str(data_path)
    short = Path(name).name

    # خروجیِ Excelِ تیکت‌های اشتباه (در کنارِ همان JSON)
    if errors_xlsx and eff_errors_out and Path(eff_errors_out).exists():
        from src.reporting.errors_export import jsonl_to_xlsx

        jsonl_to_xlsx(eff_errors_out, errors_xlsx)
        print("saved:", errors_xlsx)

    # --- داشبوردهای تصویری: همیشه ساخته می‌شوند تا inline نمایش داده شوند ---
    figs = {
        "accuracy": render_accuracy_dashboard(res, dataset_name=name),
        "cost": render_cost_dashboard(res, dataset_name=name, prices=prices),
    }
    if accuracy_png:
        figs["accuracy"].savefig(accuracy_png, dpi=160, bbox_inches="tight", facecolor="white")
        print("saved:", accuracy_png)
    if cost_png:
        figs["cost"].savefig(cost_png, dpi=160, bbox_inches="tight", facecolor="white")
        print("saved:", cost_png)

    # --- گزارش‌های HTML (تمِ تیره) ---
    if accuracy_html:
        from scripts.perf_report import render_report  # importِ تنبل

        Path(accuracy_html).write_text(render_report(res, dataset_name=short), encoding="utf-8")
        print("saved:", accuracy_html)
    if cost_html:
        from scripts.cost_report import render_html  # importِ تنبل

        breakdown = breakdown_from_eval(res, pricing=Pricing.from_tuple(prices))
        Path(cost_html).write_text(render_html(breakdown, dataset_name=short), encoding="utf-8")
        print("saved:", cost_html)

    if show:
        plt.show()
    print_text_report(res, *prices)
    return res, figs


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="تولیدِ گزارش‌های جدای دقت و هزینه (PNG و HTML).")
    ap.add_argument("data_path")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--balanced", type=int, default=None)
    ap.add_argument("--frac", type=float, default=None, help="fraction per combo, e.g. 0.2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="همهٔ پیش‌بینی‌ها (JSONL)")
    ap.add_argument("--errors", default="errors.jsonl", help="تیکت‌های اشتباه (JSON)")
    ap.add_argument("--errors-xlsx", default="errors.xlsx", help="تیکت‌های اشتباه (Excel)")
    ap.add_argument("--accuracy-png", default="accuracy_report.png")
    ap.add_argument("--cost-png", default="cost_report.png")
    ap.add_argument("--accuracy-html", default="accuracy_report.html")
    ap.add_argument("--cost-html", default="cost_report.html")
    a = ap.parse_args()
    evaluate_and_report(
        a.data_path, limit=a.limit, balanced=a.balanced, frac=a.frac, seed=a.seed,
        workers=a.workers, out_path=a.out, errors_out=a.errors, errors_xlsx=a.errors_xlsx,
        accuracy_png=a.accuracy_png, cost_png=a.cost_png,
        accuracy_html=a.accuracy_html, cost_html=a.cost_html, show=False,
    )
