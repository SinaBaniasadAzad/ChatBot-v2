# Retrieval Integration — Kaggle A/B Evaluation (cell by cell)

سنجشِ اثرِ **retrieval-augmented classification** روی دقتِ واقعی: همان نمونهٔ
ثابت (frac=0.2, seed=42) یک‌بار بدونِ retrieval و یک‌بار با آن اجرا و به‌صورتِ
جفتی (McNemar) مقایسه می‌شود.

**پیش‌نیاز Kaggle:** Accelerator = GPU (15GB کافی است؛ BGE-M3 حدود ~2GB می‌گیرد)،
Internet = ON، و Secret به نامِ `DEEPSEEK_API_KEY`.

**هزینهٔ تقریبی:** دو اجرای ~۳۲۰ تیکتی → در حدِ چند ده سنت (کش prompt فعال است).

---

## Cell 1 — کد + وابستگی‌ها + کلید (~۳ دقیقه)

```python
!git clone https://github.com/<YOUR_USER>/<YOUR_REPO>.git
%cd <YOUR_REPO>
!pip -q install -r requirements.txt -r requirements-retrieval.txt

from kaggle_secrets import UserSecretsClient
import os
os.environ["DEEPSEEK_API_KEY"] = UserSecretsClient().get_secret("DEEPSEEK_API_KEY")
```

## Cell 2 — ساختِ ایندکسِ retrieval با BGE-M3 (~۱–۲ دقیقه روی GPU)

```python
!python -m scripts.build_retrieval_index
```

خروجی: `data/retrieval/index.npz` — بردارهای ۱۵۵۳ تیکتِ پاک‌شده + نامِ مدل داخلِ فایل.

## Cell 3 — تستِ دود (اختیاری، ~۱ دقیقه، ۳۰ تیکت)

```python
!python -m scripts.eval_incdb tests/Ticketing_DB.jsonl --limit 30 --workers 4
```

اگر بالای خروجی `retrieval: ON` دیدید، ادغام فعال است.

## Cell 4 — اجرای پایه: بدونِ retrieval (~۵–۱۰ دقیقه)

```python
!python -m scripts.eval_incdb tests/Ticketing_DB.jsonl --frac 0.2 --seed 42 \
    --workers 6 --no-retrieval \
    --out preds_base.jsonl --errors errors_base.jsonl
```

## Cell 5 — اجرای جدید: با retrieval + گِیتِ اطمینان (~۵–۱۰ دقیقه)

```python
!python -m scripts.eval_incdb tests/Ticketing_DB.jsonl --frac 0.2 --seed 42 \
    --workers 6 \
    --out preds_ret.jsonl --errors errors_ret.jsonl
```

## Cell 6 — مقایسهٔ جفتی (چند ثانیه)

```python
!python -m scripts.compare_eval_runs preds_base.jsonl preds_ret.jsonl
```

---

## تفسیرِ نتایج

**۱) جدولِ مقایسهٔ جفتی (Cell 6):** `fixed` = تیکت‌هایی که retrieval درست کرد؛
`broken` = تیکت‌هایی که خراب کرد. p ≤ 0.05 یعنی بهبود واقعی است، نه نویز.
(مقایسهٔ دو عددِ کلی گمراه‌کننده است؛ همیشه fixed/broken را بخوانید.)

**۲) دو بلوکِ confidence در گزارشِ هر اجرا (Cell 4/5):**

- *legacy gate* — رفتارِ قبلی (خوداظهاری + وجودِ شاهد).
- *NEW confidence gate* — رفتارِ جدیدِ production: شاهدِ راستی‌آزمایی‌شده + مخالفتِ
  سابقهٔ kNN. این همان درمانِ «مسیریابیِ مطمئن در عینِ ندانستن» است:
  - `Auto-routable` باید دقتِ بالاتری از دقتِ کل داشته باشد (هدف: ≥95٪).
  - `Would ask/flag` همان تیکت‌هایی‌اند که در production سوال می‌گیرند
    (حداکثر ۲) یا پرچمِ بازبینی می‌خورند.

**۳) تنظیمِ آستانه‌ها** (بدونِ تغییرِ کد، با env در همان سلول):

```python
# سخت‌گیرترِ گِیت: سوالِ بیشتر، auto-دقیق‌تر     | نرم‌تر: سوالِ کمتر
!KNN_DISAGREE_PURITY=0.75 python -m scripts.eval_incdb ...   # پیش‌فرض 0.80
!RETRIEVAL_SIM_FLOOR=0.50  python -m scripts.eval_incdb ...  # پیش‌فرض 0.40
!EVIDENCE_VERIFICATION=false python -m scripts.eval_incdb ...  # جداسازیِ اثرِ هر جزء
```

## چه چیزی برای تحلیل برگردانید

1. خروجی کاملِ Cell 6 (fixed/broken + p).
2. بلوکِ «NEW confidence gate» هر دو اجرا (coverage و accuracy).
3. فایل‌های `errors_ret.jsonl` و `preds_*.jsonl` (برای تحلیلِ الگوی خطاهای باقیمانده).

## استقرار روی سرور

`data/retrieval/index.npz` و `tickets_clean.jsonl` را کنارِ کد بگذارید و
`pip install -r requirements-retrieval.txt` کنید؛ روی CPU هم کار می‌کند
(انکدِ هر کوئری ~۵۰–۱۵۰ms). اگر ایندکس/وابستگی‌ها نباشند سیستم خودکار بدونِ
retrieval بالا می‌آید (فقط لاگِ هشدار).
