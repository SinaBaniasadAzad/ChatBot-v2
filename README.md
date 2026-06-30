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
# ۱) تست تعاملی دستی
python cli.py

# ۲) سرویس API
uvicorn src.api.app:app --reload
#    مستندات: http://127.0.0.1:8000/docs

# ۳) تست‌های آفلاین (بدون API)
python -m pytest -q

# ۴) ارزیابی دقت روی Gold Set
python -m scripts.evaluate data/gold.jsonl
```

### نمونهٔ فراخوانی API

```bash
curl -X POST http://127.0.0.1:8000/classify/start \
  -H "Content-Type: application/json" \
  -d '{"summary":"خطا در ثبت پانچ","description":"ورود و خروج امروز ثبت نشد"}'
```

اگر `status` برابر `need_info` بود، با `session_id` و پاسخ کاربر به
`/classify/answer` بزنید.

---

## گام‌های بعدی (پیشنهاد مهندسی)
1. **Gold Set:** ۱۵۰–۲۰۰ تیکت دستی‌تأییدشده بسازید (برچسب خام Key/Application نویزی است)
   و با `scripts/evaluate.py` دقت واقعی را اندازه بگیرید.
2. **few-shot را غنی کنید:** نمونه‌های بیشتر در `data/examples.jsonl` (متوازن بین ۴ ترکیب).
3. **session store:** برای تولید، `_sessions` درون‌حافظه‌ای را با Redis جایگزین کنید.
4. **لاگ:** ورودی/خروجی/شواهد را ذخیره کنید؛ سرمایهٔ آیندهٔ fine-tune یا RAG.
```
