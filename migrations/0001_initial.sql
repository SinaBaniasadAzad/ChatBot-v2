-- Schema v1: تیکت‌ها + شمارندهٔ پیگیری + جستجوی تمام‌متنی + لاگ تعاملات.
-- SQLite متن را ذاتاً UTF-8 ذخیره می‌کند؛ مقایسه/جستجوی فارسی روی ستونِ نرمال‌شده
-- (ي→ی، ك→ک، ارقام فارسی/عربی→لاتین، نیم‌فاصله→فاصله) انجام می‌شود، نه متنِ خام.
-- همهٔ دستورها idempotent هستند (IF NOT EXISTS) تا اجرای مجددِ migration امن باشد.

CREATE TABLE IF NOT EXISTS tickets (
    id            INTEGER PRIMARY KEY,
    reference     TEXT NOT NULL UNIQUE,          -- TKT-YYYY-NNNNN
    submitted_at  TEXT NOT NULL,                 -- UTC ISO-8601
    employee_id   TEXT,                          -- PII؛ با retention ناشناس می‌شود (NULL)
    first_name    TEXT,                          -- PII
    last_name     TEXT,                          -- PII
    summary       TEXT NOT NULL DEFAULT '',
    description   TEXT NOT NULL DEFAULT '',
    labels        TEXT NOT NULL DEFAULT '{}',    -- JSON: {"layer1": ..., "layer2": ...}
    needs_review  INTEGER NOT NULL DEFAULT 0,
    session_id    TEXT,
    anonymized_at TEXT                            -- زمانِ ناشناس‌سازی (NULL = هنوز)
);

CREATE INDEX IF NOT EXISTS idx_tickets_submitted_at ON tickets(submitted_at);
CREATE INDEX IF NOT EXISTS idx_tickets_needs_review ON tickets(needs_review) WHERE needs_review = 1;
CREATE INDEX IF NOT EXISTS idx_tickets_layer1 ON tickets(json_extract(labels, '$.layer1'));
CREATE INDEX IF NOT EXISTS idx_tickets_layer2 ON tickets(json_extract(labels, '$.layer2'));

-- شمارهٔ پیگیریِ ترتیبی per-year؛ تراکنشی (BEGIN IMMEDIATE) — جایگزینِ اسکنِ فایلِ JSONL
CREATE TABLE IF NOT EXISTS ticket_counters (
    year     INTEGER PRIMARY KEY,
    last_seq INTEGER NOT NULL
);

-- FTS5 مستقل (متنِ نرمال‌شده را خودش نگه می‌دارد؛ rowid = tickets.id).
-- عمداً external-content نیست: متنِ ایندکس‌شده «نرمال‌شده» است و با ستونِ خام فرق دارد.
CREATE VIRTUAL TABLE IF NOT EXISTS tickets_fts USING fts5(
    content_norm,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- لاگ تحلیلیِ تعاملات (حاوی PII — سیاستِ retention روزانه اعمال می‌شود).
-- payload = همان رکوردِ JSON که قبلاً هر خطِ logs/interactions.jsonl بود
-- (scripts/export_interactions.py همان فایل را برای ابزارهای تحلیل بازتولید می‌کند).
CREATE TABLE IF NOT EXISTS interactions (
    id         INTEGER PRIMARY KEY,
    ts         TEXT NOT NULL,                    -- UTC ISO-8601
    event      TEXT NOT NULL,                    -- round | session_final
    session_id TEXT NOT NULL,
    payload    TEXT NOT NULL                     -- JSON کامل رویداد
);

CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(ts);
CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id);
