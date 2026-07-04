# چت‌بات مسیریابی خودکار تیکت (Ticket Triage Chatbot)

کارمند فقط مشکلش را توضیح می‌دهد؛ چت‌بات با کمک **DeepSeek** آن را در دو لایه
دسته‌بندی می‌کند و در صورت ابهام، **حداکثر ۲ سوال تکمیلی** می‌پرسد.

- **Layer 1 (Intent):** `Incident` یا `Service Request`
- **Layer 2 (Domain):** `ERP` یا `Staff`

> دسته‌ها در `config/taxonomy.yaml` تعریف شده‌اند. **افزودن دستهٔ جدید = افزودن یک بلوک
> به همان فایل**، بدون هیچ تغییری در کد.

---

## معماری

```
کاربر ──▶ ConversationManager ──▶ Classifier ──▶ DeepSeekClient ──▶ DeepSeek API
              │  (سقف ۲ سوال)         │  (یک دور)        │  (JSON mode + retry)
              ▼                       ▼
          Decision  ◀────────  Ambiguity-Driven
       (done / ask / fallback)   (مبتنی بر شواهد، نه عدد confidence)
```

### چرا Ambiguity-Driven و نه عدد confidence؟
عددِ confidence که خود LLM گزارش می‌کند کالیبره نیست و بیش‌اعتماد است. به‌جای آن،
از مدل می‌خواهیم برای هر برچسب **شواهد عینی (کلمات/عبارات متن)** بیاورد. یک لایه
«مبهم» است اگر مدل خودش اعلام کند یا کاندیدای برترش **هیچ شاهدی** نداشته باشد.
سوال تکمیلی فقط وقتی پرسیده می‌شود که اطلاعات واقعاً در متن **نباشد**.

---

## ساختار فایل‌ها

```
ChatBot/
├── config/
│   ├── settings.py          # env، مدل، آستانه‌ها، سقف سوال
│   └── taxonomy.yaml        # ★ تعریف دسته‌ها (تنها منبع حقیقت)
├── data/
│   └── examples.jsonl       # مثال‌های برچسب‌خورده (few-shot)
├── src/
│   ├── taxonomy.py          # بارگذاری و wrapper تایپ‌دار روی YAML
│   ├── llm/{client,prompts}.py
│   ├── classifier/
│   │   ├── schema.py        # فقط مدل‌های دادهٔ Pydantic
│   │   ├── output_parser.py # ★ تبدیل/ترمیم/اعتبارسنجی خروجی خام LLM
│   │   ├── few_shot.py      # ساخت مثال‌های متوازن
│   │   ├── classifier.py    # یک دور دسته‌بندی
│   │   └── decision.py      # ★ منطق Ambiguity-Driven
│   ├── conversation/{state,manager}.py   # ★ Orchestrator + لاگ
│   ├── api/app.py           # FastAPI
│   └── utils/{normalize,logging,interaction_log}.py
├── logs/interactions.jsonl  # لاگ ماندگار تعاملات (gitignore، حاوی PII)
├── scripts/evaluate.py      # سنجش دقت روی Gold Set
├── tests/test_classifier.py # تست‌های آفلاین
└── cli.py                   # تست تعاملی دستی
```

## لاگِ تعاملات
هر دور دسته‌بندی و هر جلسهٔ کامل به‌صورت یک خط JSON در `logs/interactions.jsonl`
ذخیره می‌شود: تیکت، **سوال‌های تکمیلی + پاسخ کاربر**، خروجی خام مدل (کاندیدا + شواهد)،
تصمیم نهایی، و متادیتای LLM (مدل، latency، مصرف توکن). این داده‌ها برای تحلیل دقت،
ساخت Gold Set و تیون prompt حیاتی‌اند. با `INTERACTION_LOG_ENABLED=false` خاموش می‌شود.

---

## راه‌اندازی (PowerShell / Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# سپس DEEPSEEK_API_KEY را داخل .env بگذارید
```

## اجرا

```powershell
# ۱) رابطِ وبِ کارمندان (SPA در پوشهٔ web/ + همان API) — مسیرِ production
uvicorn src.api.app:app --reload      # یا در PyCharm:  python run_web.py
#    رابط کاربری: http://127.0.0.1:8000/        مستندات: http://127.0.0.1:8000/docs

# ۲) نسخهٔ Gradio (برای تستِ سریع/Kaggle، همان تجربهٔ کاربری)
#    ★ نیازمند Gradio نسخهٔ ۴/۵ (نه ۶):  pip install "gradio>=4,<6"
python app_gradio.py                  # یا در PyCharm/ویندوز:  python app_gradio_windows.py

# ۳) تست تعاملی دستی
python cli.py

# ۴) تست‌های آفلاین (بدون API)
python -m pytest -q

# ۵) ارزیابی دقت روی Gold Set
python -m scripts.evaluate data/gold.jsonl
```

> **نکتهٔ Gradio:** رابطِ Gradio با API نسخهٔ ۴/۵ نوشته شده و روی **Gradio ۶ کار نمی‌کند**
> (`type="messages"` و css/theme در Blocks حذف شده‌اند). Gradio عمداً در `requirements.txt`
> نیست تا سرورِ production سبک بماند؛ فقط برای تستِ محلی `pip install "gradio>=4,<6"`.

### رابطِ وبِ کارمندان (production UI)

سفرِ کاربر در ۳ گام: **شناسایی** (کد پرسنلی + نام، یک‌بار در هر دستگاه) →
**جستجوی FAQ یا توضیحِ آزاد** → **تایید و ثبت** با شمارهٔ پیگیری `TKT-YYYY-NNNNN`.

- **FAQ (قالب‌های آماده):** ۲۰ درخواستِ پرتکرار در `data/faq.json` — ویرایش بدون تغییرِ کد.
  جستجوی لحظه‌ای با نرمال‌سازیِ فارسی/عربی (ي→ی، ك→ک، ارقام) در `src/faq.py` و همان منطق
  سمتِ کلاینت.
- **ثبتِ نهایی:** `POST /api/tickets` → افزودن به `logs/tickets.jsonl` (append-only، حاویِ
  PII، در gitignore) با شمارهٔ پیگیریِ ترتیبی — آمادهٔ اتصال به ITSM واقعی.
- **اندپوینت‌های جدید:** `GET /api/faq`، `POST /api/tickets`، `GET /api/logo`؛ مسیرهای
  `classify/*` بدونِ تغییر.
- فایل‌های UI: `web/index.html`، `web/styles.css`، `web/app.js` — بدونِ build step؛ تمِ
  تیره/روشن، ریسپانسیو، کیبورد/ARIA، ورودی‌های `dir=auto` برای متنِ فارسی.

### نمونهٔ فراخوانی API

```bash
curl -X POST http://127.0.0.1:8000/classify/start \
  -H "Content-Type: application/json" \
  -d '{"summary":"خطا در ثبت پانچ","description":"ورود و خروج امروز ثبت نشد"}'
```

اگر `status` برابر `need_info` بود، با `session_id` و پاسخ کاربر به
`/classify/answer` بزنید.

---

## گزارش‌های ارائه‌محور (برای مدیریت)

همهٔ خروجی‌ها از **دادهٔ واقعی** و از موتورهای واحد تغذیه می‌شوند تا اعداد هیچ‌وقت با
هم اختلاف نداشته باشند: `src/reporting/cost.py` (هزینه)، `src/reporting/metrics.py`
(دقت)، و دیزاین‌سیستمِ مشترکِ HTML در `src/reporting/html_ui.py`.

دقت و هزینه **جدا از هم** خروجی می‌گیرند — دو HTML و دو تصویر.

**۱) گزارشِ دقت** (`scripts/perf_report.py`) — یک HTMLِ خودبسنده با ۴ بخش: خلاصهٔ
مدیریتی، آمادگیِ عملیاتی (auto در برابر needs-review)، Precision/Recall/F1 هر کلاس،
و ماتریسِ درهم‌ریختگیِ بصری. (نیازمندِ DEEPSEEK_API_KEY)

```bash
python -m scripts.perf_report tests/Ticketing_DB.jsonl --frac 0.2 --workers 6 \
  --out accuracy_report.html --errors errors.jsonl
```

**۲) گزارشِ هزینه/توکن** (`scripts/cost_report.py`) — یک HTMLِ مستقل. مزیتِ مهم:
از لاگِ تولید **بدونِ نیاز به API** هم ساخته می‌شود:

```bash
python -m scripts.cost_report --from-log logs/interactions.jsonl --out cost_report.html
```

نرخ‌ها قابلِ تنظیم‌اند و به‌عنوان «مفروضات» در پاورقی برچسب می‌خورند
(`--price-in`, `--price-cache`, `--price-out`؛ دلار به‌ازای هر ۱M توکن).

**یک اجرا → همهٔ خروجی‌ها** (Kaggle/Notebook). `evaluate_and_report` با یک اجرای
واقعی، **دو HTML و دو تصویرِ جدا** (دقت و هزینه) + خطاها (JSON و Excel) را می‌سازد
(بدونِ مصرفِ دوبارهٔ API). هر دو داشبوردِ تصویری **همیشه inline نمایش داده می‌شوند**
(مثلِ همیشه در Kaggle)، مستقل از اینکه فایلی ذخیره بخواهی:

```python
from scripts.report import evaluate_and_report
res, figs = evaluate_and_report(
    "/kaggle/working/ChatBot-v2/tests/Ticketing_DB.jsonl", frac=0.2, workers=6,
    accuracy_html="/kaggle/working/accuracy_report.html",  # گزارشِ HTMLِ دقت (تمِ تیره)
    cost_html="/kaggle/working/cost_report.html",          # گزارشِ HTMLِ هزینه/توکن (تمِ تیره)
    accuracy_png="/kaggle/working/accuracy_report.png",    # تصویرِ دقت
    cost_png="/kaggle/working/cost_report.png",            # تصویرِ هزینه/توکن
    errors_out="/kaggle/working/errors.jsonl",             # تیکت‌های اشتباه (JSON)
    errors_xlsx="/kaggle/working/errors.xlsx",             # تیکت‌های اشتباه (Excel)
)
```

> بخشِ **Operational readiness** فعلاً خاموش است؛ با `SHOW_OPERATIONAL_READINESS = True`
> در `scripts/perf_report.py` برمی‌گردد.

---

## گام‌های بعدی (پیشنهاد مهندسی)
1. **Gold Set:** ۱۵۰–۲۰۰ تیکت دستی‌تأییدشده بسازید (برچسب خام Key/Application نویزی است)
   و با `scripts/evaluate.py` دقت واقعی را اندازه بگیرید.
2. **few-shot را غنی کنید:** نمونه‌های بیشتر در `data/examples.jsonl` (متوازن بین ۴ ترکیب).
3. **session store:** برای تولید، `_sessions` درون‌حافظه‌ای را با Redis جایگزین کنید.
4. **لاگ:** ورودی/خروجی/شواهد را ذخیره کنید؛ سرمایهٔ آیندهٔ fine-tune یا RAG.
```
