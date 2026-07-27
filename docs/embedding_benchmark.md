# Embedding Benchmark — Kaggle Guide (cell by cell)

مقایسهٔ مدل‌های embedding برای بازیابیِ تیکت‌های مشابه (پایهٔ retrieval-based few-shot).
هر سلول مستقل اجرا می‌شود؛ نتایجِ هر مدل در `data/retrieval/results/` ذخیره می‌شود.

**پیش‌نیاز Kaggle:** Accelerator = GPU (T4/P100) و Internet = ON (برای دانلود مدل‌ها).

---

## Cell 1 — دریافت کد و نصب وابستگی‌ها (~۲–۳ دقیقه)

```python
!git clone https://github.com/<YOUR_USER>/<YOUR_REPO>.git
%cd <YOUR_REPO>
!pip -q install -r requirements-retrieval.txt
```

## Cell 2 — دیتاست پاک‌شده (اگر در مخزن موجود باشد این سلول skip می‌شود)

```python
!python -m scripts.prepare_retrieval_dataset
```

خروجی: `data/retrieval/tickets_clean.jsonl` + `cleaning_report.json`
(حذفِ سلام/تشکر، نشانه‌گذاریِ پیوستِ Jira، تکراری‌های عینی؛ نگاشتِ برچسب‌ها به id).

## Cell 3 — بیس‌لاین BM25 (CPU، چند ثانیه)

```python
!python -m scripts.benchmark_embeddings --model bm25
```

این خطِ مبناست: هر مدلِ embedding باید از این بهتر باشد تا ارزشِ استقرار داشته باشد.

## Cell 4 — multilingual-e5-large (~۲–۴ دقیقه با دانلود)

```python
!python -m scripts.benchmark_embeddings --model e5-large
```

## Cell 5 — BGE-M3 (dense + sparse + hybrid) (~۳–۶ دقیقه)

```python
!python -m scripts.benchmark_embeddings --model bge-m3
```

چهار واریانت گزارش می‌شود: `dense`، `dense+bm25`، `m3-sparse`، `m3-hybrid`.

## Cell 6 — Qwen3-Embedding-0.6B (~۲–۴ دقیقه)

```python
!python -m scripts.benchmark_embeddings --model qwen3-0.6b
```

## Cell 7 (اختیاری) — gte-multilingual-base

```python
!python -m scripts.benchmark_embeddings --model gte-base
```

مدلِ قوی‌ترِ Qwen3-4B: در `src/retrieval/bench.py` یک entry آماده (کامنت‌شده) دارد.

## Cell 8 — جدولِ مقایسهٔ نهایی

```python
!python -m scripts.benchmark_embeddings --report
```

---

## تفسیرِ معیارها

| معیار | معنا | چرا مهم است |
|---|---|---|
| `L1_acc@10`, `L2_acc@10` | دقتِ دسته‌بندِ kNN (رایِ وزن‌دارِ ۱۰ همسایه، leave-one-out) به‌تفکیکِ لایه | پیش‌بینِ مستقیمِ سودمندیِ همسایه‌ها؛ **لایهٔ ۱ گلوگاهِ دقتِ کلِ سیستم است** |
| `combo@10` | هر دو لایه هم‌زمان درست | معادلِ همان «دقتِ کل» در ارزیابیِ LLM |
| `L1_agree@5` | سهمِ همسایه‌های هم‌برچسب در ۵تای برتر | کیفیتِ مثال‌هایی که به‌عنوان few-shot تزریق می‌شوند |
| `L1_frontier@0.8` | «پوشش%@دقت%» اگر فقط همسایگی‌های با خلوصِ ≥۰.۸ خودکار شوند | همان KPIِ «نرخِ اتوماسیون در دقتِ هدف» |
| `tickets/s`, `query_ms`, `vram_mb` | سرعتِ انکدِ انبوه، تاخیرِ تک‌کوئری، حافظهٔ GPU | هزینهٔ عملیاتی در production |
| `contradictions` | جفت‌های شبه‌تکراری (شباهت ≥۰.۹۵) با برچسبِ متفاوت | کاندیدهای نویزِ برچسب → ورودیِ بازبینیِ انسانی |

**قاعدهٔ انتخابِ برنده:** بیشترین `combo@10` (و در تساوی، `L1_acc@10`)؛ سپس بررسیِ
اینکه واریانتِ hybrid نسبت به dense چقدر اضافه می‌کند و هزینهٔ عملیاتی (سرعت/VRAM)
می‌ارزد یا نه. اختلاف‌های زیر ~۱ درصد را نویز فرض کنید.

## خروجی‌هایی که بعد از اجرا برای تحلیل لازم است

1. جدولِ Cell 8 (یا فایلِ `data/retrieval/results/comparison.csv`)
2. فایلِ `<model>_contradictions.jsonl` مدلِ برنده — فهرستِ کاندیدهای نویزِ برچسب
3. در صورتِ تمایل، کلِ پوشهٔ `data/retrieval/results/` را از Kaggle دانلود کنید

## اجرای محلی (بدونِ GPU)

BM25 روی هر ماشینی چند ثانیه است. مدل‌های dense روی CPU هم اجرا می‌شوند
(kهای batch به‌صورت خودکار کوچک می‌شوند) ولی انکدِ کامل ~۳–۸ دقیقه طول می‌کشد.
تستِ دود: `--limit 200` را به هر فرمان اضافه کنید.
