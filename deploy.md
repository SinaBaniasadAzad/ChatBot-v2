# deploy.md — Ticket Triage Chatbot, application deployment runbook

Everything in this document is implemented and scripted in this repository — every command is
copy-paste runnable. Scope: **the application only**. Host security, firewall, edge TLS/reverse
proxy and host-level backups belong to the infra team (see the handoff checklist at the end).

- App: FastAPI + employee SPA, classification via the hosted **DeepSeek API** (outbound HTTPS).
- State: **one SQLite file** (WAL) — tickets, tracking counter, interaction analytics, FTS search.
- Artifact: **Docker image** `ghcr.io/sinabaniasadazad/chatbot-v2` (built by CI) — or the bare-venv
  recipe below (§ 3.3).

---

## 1) Load assumptions and sizing (override via `.env` if reality differs)

| Assumption | Value |
|---|---|
| Employees / adoption | 5,000; chatbot is the primary intake channel |
| Volume | ≤ 1 ticket/employee/month → **≈ 5,000/month ≈ 250/working day** |
| Active window / peak | 08:00–17:00; peak hour ≈ 20% of daily volume → ≈ 50 tickets/h; 5× burst → ≈ 4/min |
| LLM calls per ticket | 1 classify + ≤ 2 clarifications ≈ 1.7 calls; p50 ≈ 5 s, p95 ≈ 15 s |
| → Peak concurrent LLM calls | 4/min × 1.7 × 5 s ÷ 60 ≈ **0.6**; designed for **8–10** (>10× headroom) |
| → DB writes | ≤ ~10/min peak; DB growth < ~1 GB/yr *before* retention (≈ 300 MB steady-state with defaults) |

Derived settings (already the defaults):
**1 uvicorn worker** (sessions are in-memory — this is a hard requirement, not a tuning knob),
`LLM_MAX_CONNECTIONS=16` HTTP pool to DeepSeek, SQLite WAL + `busy_timeout=5000`.
Host sizing: **2 vCPU / 2 GB RAM / 10 GB disk** is comfortable
(**+4 GB RAM** only if the optional retrieval feature is enabled, § 7).

## 2) Configure

All configuration is environment-only; nothing is hard-coded.

```bash
cp .env.example .env
# then set at minimum:
#   DEEPSEEK_API_KEY=...      ← the only required secret
```

Every variable is documented inline in [.env.example](.env.example). Notable defaults:
`REQUEST_TIMEOUT=30`, `MAX_RETRIES=3` (exponential backoff + jitter, transient errors only),
`CB_FAILURE_THRESHOLD=3`/`CB_COOLDOWN_SECONDS=30` (circuit breaker),
`INTERACTION_RETENTION_DAYS=90`, `TICKET_ANONYMIZE_DAYS=365`, `LOG_FORMAT=json` (in the image).

**Secrets & git history:** scanned all history (`git rev-list --all` + pattern grep).
A `.env` file *was* committed in the past (removed in `877e141`), but every historical version
contains only placeholder values (`sk-xxxx…`, `sk-or-v1-xxxx…`). **No real key ever leaked —
no rotation required.** `.env` is gitignored; keep real keys only in the server's `.env`.

## 3) Build & run

### 3.1 Docker (recommended path)

```bash
# build locally (CI does the same; § 6)
docker build -t ghcr.io/sinabaniasadazad/chatbot-v2:local \
  --build-arg APP_VERSION=$(git rev-parse --short HEAD) .

# run — one command, includes the persistent /data volume
docker compose up -d
docker compose logs -f chatbot     # JSON logs to stdout (host handles rotation)
```

Or without compose:

```bash
docker volume create chatbot-data
docker run -d --name chatbot --restart unless-stopped \
  --env-file .env -p 8000:8000 -v chatbot-data:/data \
  ghcr.io/sinabaniasadazad/chatbot-v2:local
```

Startup runs migrations automatically (§ 4). Shutdown is clean: SIGTERM → uvicorn finishes
in-flight requests (≤ 20 s) → DB WAL checkpoint + connection close (`stop_grace_period: 30s`).

### 3.2 Verify it is up (health check)

```bash
curl -fsS http://localhost:8000/health    # {"status":"ok","version":"..."}  (liveness)
curl -fsS http://localhost:8000/ready     # component checks: db / llm / retrieval (readiness)
curl -fsS http://localhost:8000/metrics   # counters + latency summaries (JSON)
```

`/ready` returns **503 only if the DB is unusable**. `llm: unconfigured/circuit=open` does *not*
fail readiness by design — ticket submission works without the LLM (degraded mode, § 8).

### 3.3 Manual fallback (no Docker)

Requires Python **3.12** on the host:

```bash
python3.12 -m venv .venv && . .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt                      # exact pins
cp .env.example .env                                 # set DEEPSEEK_API_KEY, optionally APP_DB_PATH
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-graceful-shutdown 20
```

Run it under the host's process supervisor (systemd et al. — infra's choice); the app needs
nothing beyond "restart on failure" and SIGTERM for shutdown.

## 4) Migrations

Versioned SQL files in [migrations/](migrations) tracked in a `schema_migrations` table.
Applied **automatically at startup**; each file runs at most once. Manual/explicit run:

```bash
docker compose exec chatbot python -c "from src.db.database import get_database; print(get_database().migrate() or 'up to date')"
```

Adding a schema change later = add `migrations/0002_<name>.sql` (idempotent statements), deploy.

## 5) Backup & restore

**What to back up: one artifact — the SQLite snapshot file.** The script uses `VACUUM INTO`
(consistent while the service runs; read-only source connection) and prunes old snapshots:

```bash
# inside the container → lands on the persistent volume /data/backups/app-YYYYmmdd-HHMMSS.db
docker compose exec chatbot python -m scripts.backup_db --out-dir /data/backups --keep 14
```

Restore (also the **rollback** path for bad data):

```bash
docker compose stop chatbot
docker compose run --rm --entrypoint sh chatbot -c \
  'cp /data/backups/app-<STAMP>.db /data/app.db && rm -f /data/app.db-wal /data/app.db-shm'
docker compose start chatbot && curl -fsS http://localhost:8000/ready
```

**Application rollback** (bad release): images are immutable and tagged —
`docker compose down && APP_VERSION=<previous-tag> docker compose up -d`.
Schema note: migration `0001` is additive-only; future migrations must stay
backwards-compatible one release back so image rollback never requires a schema downgrade.

## 6) CI/CD

[.github/workflows/ci.yml](.github/workflows/ci.yml):
- every push/PR → offline test suite (79 tests, no API key needed);
- push to `main` / tag `v*` → build + publish image to GHCR
  (`type=sha`, branch, version tag, `latest` on main).

Manual fallback if CI is down: § 3.1 build command + `docker save chatbot-v2 | gzip > chatbot.tgz`
→ copy to the server → `docker load < chatbot.tgz`.

## 7) Data layer & PII (what infra should know exists)

- Single file at `APP_DB_PATH` (container: `/data/app.db`), UTF-8 native, WAL mode.
  Persian search works via FTS5 + app-side normalization (ي→ی، ك→ک، digits, ZWNJ).
- **PII:** employee id + name in `tickets`; free-text may contain incidental PII;
  `interactions` holds ticket text for accuracy analysis.
- **Retention runs daily inside the app** (and manually: `python -m scripts.db_maintenance`):
  interactions deleted after 90 d; ticket identity anonymized after 365 d (both env-tunable).
- Analytics export (for the existing cost/accuracy tooling):
  `python -m scripts.export_interactions --out interactions.jsonl`.
- Optional retrieval feature: needs `requirements-retrieval.txt` + a built index
  (`python -m scripts.build_retrieval_index`) and ~4 GB extra RAM; **auto-disables** when absent.

## 8) Behavior when DeepSeek is down (degradation contract)

Timeout 30 s → up to 3 attempts (backoff + jitter, transient errors only) → after 3 consecutive
failed cycles the circuit breaker opens for 30 s and classification fails **fast**.
The API returns `503 {"detail":{"code":"llm_unavailable"}}` + `Retry-After`; the SPA then offers
**submit without classification** — the ticket is stored with `needs_review=true` for manual
routing. Intake never stops; verify with `/metrics` (`classify_unavailable_total`) and `/ready`.

---

## 9) Handoff to infra team — checklist

**Runtime**
- [ ] Docker Engine (or Python **3.12** for the bare recipe in § 3.3); Linux or Windows host both fine
- [ ] Run: `docker compose up -d` from a directory containing `compose.yaml` + the filled `.env`
- [ ] Exactly **1 replica / 1 worker** (in-memory conversation sessions)

**Network**
- [ ] Inbound: app listens on **:8000** (HTTP). Put your edge TLS/reverse proxy in front; no app changes needed
- [ ] Outbound: HTTPS :443 to **`https://api.deepseek.com`** only (value of `DEEPSEEK_BASE_URL`)
- [ ] No other inbound/outbound requirements; no inbound access needed from the internet

**Configuration (env)**
- [ ] `.env` from [.env.example](.env.example); required: **`DEEPSEEK_API_KEY`** (secret — store per your practice)
- [ ] Optional overrides documented inline; defaults are production-safe

**Datastore**
- [ ] SQLite file on the **`/data` volume** (or `APP_DB_PATH`) — persistent, ~1 GB/yr worst case
- [ ] No external database service required

**Health / monitoring**
- [ ] Liveness: `GET /health` = 200; Readiness: `GET /ready` = 200 (503 ⇒ DB problem)
- [ ] Logs: JSON lines on stdout/stderr — attach to your log collection/rotation
- [ ] Optional: scrape `GET /metrics` (JSON counters/latencies)

**Backups (please schedule + offload)**
- [ ] Daily: `docker compose exec chatbot python -m scripts.backup_db --out-dir /data/backups --keep 14`
- [ ] Offload `/data/backups/app-*.db` to your backup storage (single-file logical backups)
- [ ] Restore procedure: § 5 (app-level; we own it — call us)

**Sizing**
- [ ] 2 vCPU / 2 GB RAM / 10 GB disk (≥ 6 GB RAM only if we later enable the retrieval feature)
