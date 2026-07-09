# Observability و ارزیابی — Langfuse + صفِ بازبینی

هدف: **هر تصمیمِ چت‌بات شفاف، قابلِ‌توضیح، قابلِ‌اندازه‌گیری و قابلِ‌بازرسی باشد** —
از لحظهٔ ثبتِ تیکت تا نتیجهٔ نهایی: همسایه‌های انتخاب‌شده (retrieval)، مثال‌های
few-shot، پیام‌های LLM، هزینه، تصمیمِ ابهام، و بازخوردِ انسانی.

معماری:

```
کاربر ─▶ ConversationManager ─▶ Classifier ─▶ DeepSeekClient ─▶ DeepSeek API
             │ trace ریشه          │ span retrieval + classify   │ span generation
             │ (session گروه‌بندی)  │ (همسایه‌ها + رای kNN)        │ (پیام‌ها/usage/هزینه)
             ▼
        Langfuse (self-hosted, deploy/langfuse) ◀── scores خودکار + انسانی
             ▲
        صفِ بازبینی (SQLite + /review) ── برچسبِ طلایی → score → Gold Set
```

**اصلِ طراحی:** ردیابی کاملاً اختیاری است. نبودِ بستهٔ `langfuse`، نبودِ کلیدها، یا
`OBSERVABILITY_ENABLED=false` فقط ردیابی را خاموش می‌کند — مسیرِ دسته‌بندی، تست‌ها
و ارزیابی بدونِ هیچ تغییری کار می‌کنند (همان الگوی graceful-degradation بازیاب).

---

## ۱) راه‌اندازیِ Langfuse (یک‌بار)

پیش‌نیاز: Docker Desktop (روی Windows) یا Docker Engine (سرور).

```bash
cd deploy/langfuse
cp .env.example .env      # secretها و LANGFUSE_INIT_* را تنظیم کنید
docker compose up -d
# UI: http://localhost:3000  (کاربر/پروژه/کلیدها با LANGFUSE_INIT_* خودکار ساخته می‌شوند)
```

سپس در `.env` ریشهٔ پروژه (نمونه: `.env.example`):

```env
OBSERVABILITY_ENABLED=true
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-...      # همان LANGFUSE_INIT_PROJECT_PUBLIC_KEY
LANGFUSE_SECRET_KEY=sk-lf-...      # همان LANGFUSE_INIT_PROJECT_SECRET_KEY
OBSERVABILITY_ENVIRONMENT=production
```

و `pip install -r requirements.txt` (بستهٔ `langfuse>=4,<5`).

> **PII:** traceها متنِ کاملِ تیکت (دادهٔ پرسنلی) را دارند؛ Langfuse حتماً باید
> self-hosted و داخلِ شبکهٔ سازمان بماند. `TELEMETRY_ENABLED=false` در compose
> تنظیم شده است. سیاستِ نگه‌داری/دسترسیِ همان logs/ این‌جا هم برقرار است.

## ۲) هر trace چه دارد؟ (Trace visualization)

هر دورِ classify یک trace با نامِ `ticket-classification` می‌سازد که با
`session_id` گروه‌بندی می‌شود (سوال‌وجواب‌های تکمیلیِ همان تیکت = همان session در UI):

| Observation | نوع | محتوا |
|---|---|---|
| `classification-round` | span (ریشه) | ورودیِ کامل (summary/description/شفاف‌سازی‌ها)، خروجیِ نهایی (action/labels/question/needs_review)، مدل، دلیلِ کناره‌گیریِ retrieval |
| `classify` | span | اثرانگشتِ پیکربندی + برچسب‌ها و reasoning مدل |
| `retrieval` | retriever | **همسایه‌های تزریق‌شده** (key، شباهت، برچسب‌ها، متن — دقیقاً همان‌که به prompt رفت)، رایِ kNN + purity، یا `abstain_reason` (`below_sim_floor`/`empty_query`) |
| `deepseek-completion` | generation | پیام‌های system/user کامل (شاملِ few-shot و precedents)، خروجیِ خام، usage با تفکیکِ cache hit/miss، **هزینهٔ دلاری**، latency، تعدادِ retry |
| `decision` | chain | بودجهٔ سوال، رای‌های kNN، action، دلایلِ ابهام per-layer، شواهدِ راستی‌آزمایی‌شده |

**اثرانگشتِ پیکربندی** (`config_fingerprint` در متادیتای trace): هشِ taxonomy +
examples + system prompt + آستانه‌ها + مدل. دو run با اثرانگشتِ یکسان = پیکربندیِ
یکسان → مقایسهٔ نسخه‌ها معتبر است. اجزای مؤثر: `GET /api/debug/config`.

### scoreهای خودکارِ هر trace (خوراکِ داشبورد و پایشِ drift)
`needs_review`، `asked_clarification`، `retrieval_abstained`،
`retrieval_top_similarity`، `knn_llm_agreement`، `hallucinated_evidence`.
در Langfuse → Dashboards روندِ این‌ها را رسم کنید؛ جهشِ ناگهانی در
`retrieval_abstained` یا افتِ `knn_llm_agreement` = علامتِ drift داده/ایندکس.

## ۳) هزینه (Cost tracking)

هزینهٔ هر فراخوانی با قیمت‌گذاریِ سه‌نرخیِ DeepSeek (cache-miss/cache-hit/output —
env: `DEEPSEEK_PRICE_*`) محاسبه و روی generation ثبت می‌شود؛ Langfuse به‌صورتِ
built-in هزینه را به تفکیکِ trace/session/کاربر/روز جمع می‌زند. اعداد با
`scripts/cost_report.py` هم‌منبع‌اند (`src/reporting/cost.py`).

## ۴) دیتاست و Experiment (Dataset management + Score tracking)

```bash
# ۱) آپلودِ همان نمونهٔ استانداردِ ارزیابی (frac=0.2, seed=42) به‌عنوانِ دیتاست
python -m scripts.langfuse_dataset tests/Ticketing_DB.jsonl --name ticketing-gold-20pct --frac 0.2 --seed 42

# ۲) اجرای experiment (هر آیتم از کلِ خطِ تولید می‌گذرد؛ self-key حذف می‌شود)
python -m scripts.run_experiment --dataset ticketing-gold-20pct --run "v1-baseline" --workers 6
```

scoreهای هر آیتم: `layer1_correct` / `layer2_correct` / `overall_correct`،
`retrieval_agreement_<layer>` (سهمِ همسایه‌های هم‌برچسبِ طلایی)،
`knn_vote_correct_<layer>`، `retrieval_abstained`.

در UI: **Datasets → Runs** دو run را ستون‌به‌ستون مقایسه کنید (کدام تیکت fix شد،
کدام شکست) — همان روشِ CLAUDE.md §۳ ولی بصری و کلیک‌پذیر. traceهای experiment با
`environment=experiment` از production جدا هستند.

## ۵) صفِ بازبینی و حاشیه‌نویسی (Annotation & Review queues)

رابط: **`http://localhost:8000/review`** (همان FastAPI؛ بدونِ نیاز به کلیدِ LLM).

- **ورودِ خودکار:** هر جلسه‌ای که `needs_review` شود (fallback پس از ۲ سوال) با
  trace-link واردِ صف می‌شود.
- **ورودِ دستی:** خطاهای ارزیابی:
  `python -m scripts.import_review_items errors.jsonl`
  و خروجیِ ممیزیِ retrieval (`--enqueue`).
- بازبین برچسبِ درست را انتخاب می‌کند → روی traceِ اصلی scoreهای
  `human_reviewed` / `human_correct` / `human_label_<layer>` ثبت می‌شود و آمارِ
  «model↔human agreement» به‌روز می‌شود (ارزیابیِ آنلاینِ واقعی، مکملِ ارزیابیِ آفلاین).
- **خروجیِ Gold Set:** `GET /api/review/export` — برچسب‌های تاییدشدهٔ انسانی برای
  رشدِ `data/examples.jsonl` / Gold Set. ⚠️ قبل از افزودن به few-shot، گاردِ نشت:
  کلیدهای نمونهٔ ارزیابی (frac=0.2, seed=42) را حذف کنید (CLAUDE.md §۴).

ذخیره‌سازی: SQLite در `logs/review.db` (PII → gitignored). API: `/api/review/*`.

## ۶) ممیزیِ کیفیتِ retrieval — «آیا همسایه‌های درست انتخاب می‌شوند؟»

دو لایهٔ تضمین:

1. **تست‌های آفلاین (بدونِ مدل/API):** `tests/test_retrieval_quality.py` +
   `tests/test_retrieval_integration.py` — صحتِ انتخاب/ترتیبِ همسایه‌ها، گاردهای
   نشت (exclude_keys / drop_self_sim)، کفِ شباهت، ریاضیِ purity، دلایلِ کناره‌گیری
   (`explain`)، و این‌که payloadِ trace دقیقاً همان چیزی است که به prompt تزریق شد.

2. **ممیزی روی ایندکس و دادهٔ واقعی:**
   ```bash
   python -m scripts.eval_retrieval tests/Ticketing_DB.jsonl --frac 0.2 --seed 42 \
       --out retrieval_audit.json --review-out retrieval_worst.jsonl
   ```
   خروجی: neighbor agreement@k و kNN vote accuracy per-layer، **کالیبراسیونِ purity**
   (اعتبارسنجیِ آستانهٔ `KNN_DISAGREE_PURITY=0.80`)، نرخ/دلایلِ کناره‌گیری، و
   worst offenderها (شباهتِ بالا + همسایگیِ غلط = کاندیدِ نویزِ برچسب) که با
   `--enqueue` مستقیم واردِ صفِ بازبینی می‌شوند.

مکمل: `src/retrieval/bench.py` برای *مقایسهٔ مدل‌های embedding* (leave-one-out)
سرِ جای خود باقی است؛ `eval_retrieval` مسیرِ *production* را با پارامترهای واقعی می‌سنجد.

## ۷) متغیرهای محیطی (خلاصه)

| متغیر | پیش‌فرض | نقش |
|---|---|---|
| `OBSERVABILITY_ENABLED` | `true` | کلیدِ کلیِ ردیابی (بدونِ کلیدها عملاً خاموش) |
| `LANGFUSE_HOST` | `http://localhost:3000` | آدرسِ Langfuse self-hosted |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | خالی | کلیدهای پروژه |
| `OBSERVABILITY_ENVIRONMENT` | `production` | جداسازیِ production/evaluation/experiment |
| `LANGFUSE_SAMPLE_RATE` | `1.0` | نمونه‌برداری (حجمِ فعلی نیازی به کاهش ندارد) |
| `DEEPSEEK_PRICE_INPUT_PER_1M` … | `0.14 / 0.0028 / 0.28` | قیمت‌گذاریِ هزینه |
| `REVIEW_QUEUE_ENABLED` | `true` | صفِ بازبینی |
| `REVIEW_DB_PATH` | `logs/review.db` | مخزنِ صف (PII) |

## ۸) عیب‌یابی

- **trace نمی‌آید:** کلیدها/HOST را چک کنید؛ لاگِ `observability` دلیلِ خاموشی را
  می‌گوید. `python -c "from src import observability as o; print(o.enabled())"`.
- **هزینه صفر است:** usage از DeepSeek نیامده (فیلدهای `prompt_cache_*`)؛ تفکیک
  نباشد کلِ prompt به‌صورتِ محافظه‌کارانه cache-miss حساب می‌شود.
- **اسکریپت زود تمام می‌شود و trace ناقص است:** `obs.flush()` در انتهای اسکریپت‌ها
  هست؛ اسکریپتِ جدید نوشتید؟ flush یادتان نرود.
- **UIِ بازبینی خالی است:** آیتم فقط از fallbackِ production، import خطاهای eval،
  یا `--enqueue` ممیزی ساخته می‌شود.
