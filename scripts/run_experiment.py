"""
اجرای Experiment روی دیتاستِ Langfuse (Score Tracking + مقایسهٔ runها در UI).

هر آیتمِ دیتاست یک‌بار از کلِ خطِ تولید (retrieval → prompt → LLM) می‌گذرد؛ trace
به runِ دیتاست لینک می‌شود و ارزیاب‌ها این scoreها را ثبت می‌کنند:

  layer1_correct / layer2_correct / overall_correct     درستی نسبت به برچسبِ طلایی
  retrieval_agreement_<layer>    سهمِ همسایه‌های تزریق‌شده که هم‌برچسبِ طلایی‌اند
  knn_vote_correct_<layer>       آیا رایِ kNN با برچسبِ طلایی یکی بود؟
  retrieval_abstained            آیا retrieval کناره‌گیری کرد؟

runها با اثرانگشتِ پیکربندی مهر می‌خورند → در UIِ Langfuse (Datasets → Runs)
می‌توان دو نسخهٔ taxonomy/prompt/آستانه را ستون‌به‌ستون مقایسه کرد.

اجرا:
    python -m scripts.run_experiment --dataset ticketing-gold-20pct \
        --run "v1-baseline" --workers 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402

# قبل از ساختِ کلاینت، محیطِ ردیابی را جدا کن تا traceهای آزمایشی با production قاطی نشوند.
settings.observability_environment = "experiment"

from src import observability as obs  # noqa: E402
from src.classifier.classifier import Classifier  # noqa: E402


def _item_field(item, name: str, default=None):
    """آیتم‌های SDK صفت‌محورند؛ dictهای محلی کلیدمحور — هر دو را پشتیبانی کن."""
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def build_task(clf: Classifier):
    def task(*, item, **kwargs):
        inp = _item_field(item, "input") or {}
        meta = _item_field(item, "metadata") or {}
        key = meta.get("key") or _item_field(item, "id")
        out, cmeta = clf.classify(
            inp.get("summary", ""), inp.get("description", ""),
            exclude_keys=frozenset({str(key)}),  # گاردِ نشت: خودِ تیکت سابقهٔ خودش نشود
        )
        return {
            "labels": {lid: (lo.top.label if lo.top else None) for lid, lo in out.layers.items()},
            "retrieval": cmeta.get("retrieval"),
            "reasoning": out.reasoning,
        }

    return task


def correctness_evaluator(*, input, output, expected_output, metadata, **kwargs):
    from langfuse import Evaluation

    pred = (output or {}).get("labels") or {}
    expected = expected_output or {}
    evals = [
        Evaluation(name=f"{lid}_correct", value=int(pred.get(lid) == gold))
        for lid, gold in expected.items()
    ]
    if expected:
        evals.append(
            Evaluation(
                name="overall_correct",
                value=int(all(pred.get(lid) == gold for lid, gold in expected.items())),
                comment=f"pred={pred}",
            )
        )
    return evals


def build_retrieval_evaluator(demos_by_key: dict):
    """کیفیتِ retrieval نسبت به برچسبِ طلایی — پاسخِ مستقیم به «همسایهٔ درست؟»."""

    def retrieval_evaluator(*, input, output, expected_output, metadata, **kwargs):
        from langfuse import Evaluation

        retrieval = (output or {}).get("retrieval")
        expected = expected_output or {}
        evals = [Evaluation(name="retrieval_abstained", value=int(retrieval is None))]
        if not retrieval:
            return evals
        demo_keys = retrieval.get("demo_keys") or []
        for lid, gold in expected.items():
            labels = [
                demos_by_key.get(k, {}).get(lid)
                for k in demo_keys if demos_by_key.get(k, {}).get(lid)
            ]
            if labels:
                evals.append(
                    Evaluation(
                        name=f"retrieval_agreement_{lid}",
                        value=round(sum(int(l == gold) for l in labels) / len(labels), 4),
                        comment=f"neighbors={demo_keys}",
                    )
                )
        for lid, vote in (retrieval.get("votes") or {}).items():
            if lid in expected and vote.get("label"):
                evals.append(
                    Evaluation(
                        name=f"knn_vote_correct_{lid}",
                        value=int(vote["label"] == expected[lid]),
                        comment=f"purity={vote.get('purity')}",
                    )
                )
        return evals

    return retrieval_evaluator


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", default=None, help="نامِ run (پیش‌فرض: model + اثرانگشت)")
    ap.add_argument("--description", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-retrieval", action="store_true")
    args = ap.parse_args()

    lf = obs.get_client()
    if lf is None:
        raise SystemExit(
            "Langfuse در دسترس نیست: OBSERVABILITY_ENABLED و LANGFUSE_PUBLIC_KEY/SECRET_KEY را در .env تنظیم کنید."
        )

    clf = Classifier(retriever=None) if args.no_retrieval else Classifier()
    fp = clf.fingerprint
    run_name = args.run or f"{settings.model}-{fp['fingerprint']}"
    demos_by_key = (
        {r["key"]: r for r in clf.retriever.rows} if clf.retriever is not None else {}
    )

    dataset = lf.get_dataset(args.dataset)
    print(f"run «{run_name}» روی دیتاستِ «{args.dataset}» (fingerprint={fp['fingerprint']})",
          file=sys.stderr)

    result = dataset.run_experiment(
        name=f"triage-{args.dataset}",
        run_name=run_name,
        description=args.description or f"config fingerprint {fp['fingerprint']}",
        task=build_task(clf),
        evaluators=[correctness_evaluator, build_retrieval_evaluator(demos_by_key)],
        max_concurrency=args.workers,
        metadata={
            "config_fingerprint": fp["fingerprint"],
            "model": settings.model,
            "retrieval_active": str(clf.retriever is not None),
            **{f"cfg_{k}": str(v) for k, v in fp["components"].items()},
        },
    )
    obs.flush()

    fmt = getattr(result, "format", None)
    print(fmt() if callable(fmt) else result)
    print(f"\nمقایسهٔ runها: {settings.langfuse_host}  →  Datasets → {args.dataset} → Runs")


if __name__ == "__main__":
    main()
