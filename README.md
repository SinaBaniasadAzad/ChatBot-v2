# Ticket Triage Chatbot

The employee only describes their issue; the chatbot uses **DeepSeek** to classify it
into two layers and, if necessary, asks **up to 2 follow-up questions**.

- **Layer 1 (Intent):** `Incident` or `Service Request`
- **Layer 2 (Domain):** `ERP` or `Staff`

> Categories are defined in `config/taxonomy.yaml`. **Adding a new category = adding a new block
> to the same file**, with **no code changes required**.

---

## Architecture

```
User ──▶ ConversationManager ──▶ Classifier ──▶ DeepSeekClient ──▶ DeepSeek API
            │  (max 2 questions)      │  (single pass)      │  (JSON mode + retry)
            ▼                         ▼
        Decision  ◀──────────  Ambiguity-Driven
     (done / ask / fallback)   (evidence-based, not confidence-based)
```

### Why Ambiguity-Driven instead of confidence scores?
The confidence score reported by an LLM is not calibrated and is often overconfident.
Instead, the model is required to provide **objective evidence (words/phrases from the text)**
for each predicted label. A layer is considered **ambiguous** if the model explicitly marks it
as ambiguous or if its top candidate has **no supporting evidence**. A follow-up question is
asked only when the required information is genuinely **missing from the user's text**.

---

## Project Structure

```
ChatBot/
├── config/
│   ├── settings.py          # Environment variables, model, thresholds, question limit
│   └── taxonomy.yaml        # ★ Category definitions (single source of truth)
├── data/
│   └── examples.jsonl       # Labeled few-shot examples
├── src/
│   ├── taxonomy.py          # YAML loader + typed wrapper
│   ├── llm/{client,prompts}.py
│   ├── classifier/
│   │   ├── schema.py        # Pydantic data models only
│   │   ├── output_parser.py # ★ Parse/repair/validate raw LLM output
│   │   ├── few_shot.py      # Balanced few-shot example builder
│   │   ├── classifier.py    # Single-pass classification
│   │   └── decision.py      # ★ Ambiguity-driven decision logic
│   ├── conversation/{state,manager}.py   # ★ Orchestrator + logging
│   ├── api/app.py           # FastAPI
│   └── utils/{normalize,logging,interaction_log}.py
├── logs/interactions.jsonl  # Persistent interaction log (gitignored, contains PII)
├── scripts/evaluate.py      # Accuracy evaluation on a Gold Set
├── tests/test_classifier.py # Offline tests
└── cli.py                   # Manual interactive testing
```

## Interaction Logging

Each classification round and every complete conversation session is stored as a
single JSON line in `logs/interactions.jsonl`: the ticket, **follow-up questions + user
answers**, raw model output (candidates + evidence), final decision, and LLM metadata
(model, latency, token usage). These logs are essential for accuracy analysis,
building a Gold Set, and prompt tuning. Logging can be disabled with
`INTERACTION_LOG_ENABLED=false`.

---

## Setup (PowerShell / Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# Then add your DEEPSEEK_API_KEY to the .env file
```

## Run

```powershell
# 1) Employee web UI (SPA in web/ + same API) — production path
uvicorn src.api.app:app --reload      # or in PyCharm: python run_web.py
#    UI: http://127.0.0.1:8000/        Docs: http://127.0.0.1:8000/docs

# 2) Gradio version (for quick testing/Kaggle, same user experience)
#    ★ Requires Gradio 4/5 (not 6): pip install "gradio>=4,<6"
python app_gradio.py                  # or on Windows/PyCharm: python app_gradio_windows.py

# 3) Manual interactive test
python cli.py

# 4) Offline tests (no API required)
python -m pytest -q

# 5) Evaluate accuracy on the Gold Set
python -m scripts.evaluate data/gold.jsonl
```

> **Gradio note:** The Gradio UI was built against the Gradio 4/5 API and is **not compatible with Gradio 6**
> (`type="messages"` and some Blocks css/theme features were removed). Gradio is intentionally excluded
> from `requirements.txt` to keep the production server lightweight; install it locally with
> `pip install "gradio>=4,<6"` when needed.

### Employee Web UI (Production)

The user journey consists of three steps:
**Identification** (employee ID + name, once per device) →
**FAQ search or free-text description** →
**Confirmation and submission** with a tracking number in the format `TKT-YYYY-NNNNN`.

- **FAQ (predefined templates):** 20 common requests stored in `data/faq.json` — editable without code changes.
  Instant search uses Persian/Arabic normalization (ي→ی, ك→ک, digits) in `src/faq.py`
  with the same logic implemented on the client side.
- **Final submission:** `POST /api/tickets` → appends to `logs/tickets.jsonl`
  (append-only, contains PII, gitignored) with a sequential tracking number — ready for
  integration with a real ITSM system.
- **New endpoints:** `GET /api/faq`, `POST /api/tickets`, `GET /api/logo`;
  existing `classify/*` endpoints remain unchanged.
- **UI files:** `web/index.html`, `web/styles.css`, `web/app.js` — no build step;
  dark/light theme, responsive design, keyboard/ARIA support, and `dir=auto`
  inputs for Persian text.

### API Example

```bash
curl -X POST http://127.0.0.1:8000/classify/start \
  -H "Content-Type: application/json" \
  -d '{"summary":"Punch registration error","description":"Today's check-in/check-out was not recorded"}'
```

If `status` is `need_info`, send the user's answer together with the `session_id`
to `/classify/answer`.

---

## Management Reports

All outputs are generated from **real execution data** and share the same underlying
engines, ensuring that reported numbers are always consistent:
`src/reporting/cost.py` (cost), `src/reporting/metrics.py` (accuracy), and the shared
HTML design system in `src/reporting/html_ui.py`.

Accuracy and cost are generated **independently** — producing two HTML reports and two images.

**1) Accuracy Report** (`scripts/perf_report.py`) — a self-contained HTML report with four sections:
executive summary, operational readiness (auto vs needs-review), per-class
Precision/Recall/F1, and a visual confusion matrix.
(**Requires DEEPSEEK_API_KEY**)

```bash
python -m scripts.perf_report tests/Ticketing_DB.jsonl --frac 0.2 --workers 6 \
  --out accuracy_report.html --errors errors.jsonl
```

**2) Cost/Token Report** (`scripts/cost_report.py`) — a standalone HTML report.
A key advantage is that it can be generated directly from production logs
**without calling the API**:

```bash
python -m scripts.cost_report --from-log logs/interactions.jsonl --out cost_report.html
```

Pricing is configurable and clearly labeled as **assumptions** in the report footer
(`--price-in`, `--price-cache`, `--price-out`; USD per 1M tokens).

**One run → all outputs** (Kaggle/Notebook).
`evaluate_and_report` performs a single evaluation run and produces **two HTML reports,
two separate images** (accuracy and cost), plus error files (JSON and Excel),
**without consuming additional API calls**. Both visual dashboards are **always displayed inline**
(as in Kaggle), regardless of whether output files are saved.

```python
from scripts.report import evaluate_and_report
res, figs = evaluate_and_report(
    "/kaggle/working/ChatBot-v2/tests/Ticketing_DB.jsonl", frac=0.2, workers=6,
    accuracy_html="/kaggle/working/accuracy_report.html",  # Accuracy HTML report (dark theme)
    cost_html="/kaggle/working/cost_report.html",          # Cost/token HTML report (dark theme)
    accuracy_png="/kaggle/working/accuracy_report.png",    # Accuracy image
    cost_png="/kaggle/working/cost_report.png",            # Cost/token image
    errors_out="/kaggle/working/errors.jsonl",             # Misclassified tickets (JSON)
    errors_xlsx="/kaggle/working/errors.xlsx",             # Misclassified tickets (Excel)
)
```

> The **Operational readiness** section is currently disabled. Re-enable it by setting
> `SHOW_OPERATIONAL_READINESS = True` in `scripts/perf_report.py`.

---

## Next Steps (Engineering Suggestions)

1. **Build a Gold Set:** Create 150–200 manually verified tickets (raw Key/Application labels are noisy)
   and measure real accuracy with `scripts/evaluate.py`.
2. **Expand the few-shot dataset:** Add more balanced examples to `data/examples.jsonl`
   across all four label combinations.
3. **Session store:** Replace the in-memory `_sessions` with Redis for production deployment.
4. **Logging:** Store inputs, outputs, and supporting evidence as future assets for fine-tuning or RAG.
