# ایمیجِ production — فقط runtime (بدونِ ابزار تست/گزارش/بنچمارک).
# بیلد:   docker build -t chatbot-v2:local --build-arg APP_VERSION=$(git rev-parse --short HEAD) .
# اجرا:   docker run --env-file .env -p 8000:8000 -v chatbot-data:/data chatbot-v2:local
FROM python:3.12-slim

ARG APP_VERSION=dev
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_VERSION=${APP_VERSION} \
    # پیش‌فرض‌های production؛ با env-file قابلِ‌بازنویسی
    APP_DB_PATH=/data/app.db \
    LOG_FORMAT=json

WORKDIR /app

# لایهٔ وابستگی‌ها جدا از کد تا cache بیلد حفظ شود
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# فقط چیزهایی که سرویس لازم دارد (نه تست/گزارش/آرتیفکت‌های بنچمارک)
COPY config/ config/
COPY migrations/ migrations/
COPY src/ src/
COPY web/ web/
COPY scripts/backup_db.py scripts/db_maintenance.py scripts/export_interactions.py \
     scripts/build_retrieval_index.py scripts/
COPY data/faq.json data/examples.jsonl data/logo.png data/
# سابقهٔ برچسب‌خورده برای retrieval (ایندکسِ npz جدا ساخته/mount می‌شود)
COPY data/retrieval/tickets_clean.jsonl data/retrieval/

# اجرای غیرریشه + volume دادهٔ ماندگار (DB + بکاپ‌ها)
RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /data && chown appuser:appuser /data
USER appuser
VOLUME /data

EXPOSE 8000

# liveness در سطحِ کانتینر (readiness کامل‌تر: GET /ready)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); sys.exit(0 if r.status==200 else 1)"]

# دقیقاً یک worker: جلسه‌های گفتگو درون‌حافظه‌ای‌اند (deploy.md).
# خاموشیِ تمیز: uvicorn با SIGTERM درخواست‌های باز را تا 20s تمام می‌کند؛ lifespan اتصال‌های DB را می‌بندد.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-graceful-shutdown", "20", "--no-server-header"]
