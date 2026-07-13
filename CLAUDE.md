# CLAUDE.md

Guidelines for working with this repository using Claude. Before making any changes, read this file and then `README.md`.

---

## Architecture at a Glance

```
User ─▶ ConversationManager ─▶ Classifier ─▶ DeepSeekClient ─▶ DeepSeek API
            (max 2 questions)      (single pass)      (JSON mode + retry)
                                    │
                           Decision (done / ask / fallback)  ← Ambiguity-Driven
```

- **Two classification layers** (defined in `config/taxonomy.yaml`):
  - `layer1` = **Type**: `incident` or `service_request`
  - `layer2` = **Domain**: `erp` or `staff`
- **Single source of truth = `config/taxonomy.yaml`.** Add or modify categories and rules only there.
  No category names are hard-coded in the source code (`src/llm/prompts.py` builds everything dynamically).
- **Few-shot examples** are loaded from `data/examples.jsonl`; `src/classifier/few_shot.py`
  selects a balanced number of examples for each label combination (default: 4 each).
- **Ambiguity-Driven** (not confidence-based): a layer is considered ambiguous if the model
  returns `needs_clarification` or if the top candidate has no textual evidence. In that case,
  the chatbot asks up to **2 follow-up questions**.

### Key Files

| Path | Purpose |
|---|---|
| `config/taxonomy.yaml` | ★ Category definitions, cues, and signal/rule mapping. Primary accuracy lever. |
| `data/examples.jsonl` | Balanced few-shot examples without data leakage. |
| `src/llm/prompts.py` | Builds system/user prompts dynamically from taxonomy + few-shot examples. |
| `src/classifier/{classifier,decision,output_parser,few_shot}.py` | Core classification and decision logic. |
| `scripts/eval_incdb.py` | Accuracy evaluation on the raw dataset (single-shot). |
| `scripts/report.py` | Visual reporting dashboard. |

---

## Organization Labeling Rules (★ Most Important Accuracy Knowledge)

These rules were derived from **statistical analysis of the entire dataset** (1,633 tickets)
and may differ from standard ITIL intuition. When in doubt, **trust the data, not intuition**.

### Type (layer1) — Incident vs Service Request

> A previous attempt to make this layer more Service Request–leaning reduced overall
> accuracy from **89% to 84%** by misclassifying real Incidents. Therefore, the
> **"blocked/broken = Incident"** rule should be preserved.

- **Incident** (default for problems, errors, and blockers): the user is blocked by the system,
  encounters an error, incorrect behavior, failed calculations, synchronization issues,
  rejected actions, disabled options, missing records, or workflows that cannot proceed.
  Even if the user phrases the request as "create/delete/activate", treat those verbs as
  proposed solutions rather than ticket intent.
- **Service Request** only when **no malfunction exists** and the request is a normal
  administrative task, such as:
  - Creating user accounts or access permissions.
  - Changing job position, title, contract type, or approvers.
  - Providing reports, payroll slips, or increasing limits.
  - Enabling self-evaluation access.
  - Reverting an incorrect approval step.
  - Cancelling an accidental action performed by the user.

> **Important Loan exception:** If the entire Loan module is unavailable or not enabled,
> classify it as **Service Request**. If a specific Loan workflow step fails (approval,
> rejection, document upload, HR approval, etc.), classify it as **Incident**.

- Improve layer1 only through **high-confidence narrow exceptions** supported by dataset
  statistics and balanced few-shot examples.
- Some cases (e.g., vague Loan workflow issues or leave balance questions) are inherently
  noisy. Avoid overfitting.

### Domain (layer2) — ERP vs Staff

- **ERP**: attendance, check-in/out, timesheets, leave, missions, payroll,
  job positions, contracts, cost centers, checkout, personnel profile updates,
  compensation periods.
- **Staff**: Staff platform, evaluations, goals, self-evaluation, rewards,
  loans, guarantees, module activation, IT infrastructure requests
  (Staff account, Jabber, VPN, laptop, backup, Nakccess).
- Known taxonomy pitfalls:
  - Job Position → ERP
  - Contract → ERP
  - Checkout → ERP

---

## What Actually Improved Accuracy

- ✅ The most reliable improvements came from **layer2 (Domain)** by correctly moving
  Position, Contract, Checkout, Cost Center, and Compensation to **ERP**, while clearly
  defining Staff as including IT services.
- ⚠️ Directly changing **layer1** proved harmful and significantly increased regressions.
  Keep it Incident-leaning.
- Compare model versions using the **same evaluation subset** (`frac=0.2`, `seed=42`)
  to measure fixes and regressions consistently.

---

## Evaluating Accuracy

```bash
# Evaluate using the reproducible 20% subset
python -m scripts.eval_incdb tests/Ticketing_DB.jsonl --frac 0.2 --seed 42 \
  --workers 6 --out preds.jsonl --errors errors.jsonl

# Visual dashboard
python -m scripts.report tests/Ticketing_DB.jsonl --frac 0.2 --seed 42

# Offline tests (no API required)
python -m pytest -q
```

**Recommended improvement workflow**

1. Review `errors.jsonl`.
2. Verify the error pattern across the **entire dataset**.
3. Update `taxonomy.yaml` and `data/examples.jsonl`.
4. Re-evaluate using the same seed and check for regressions.

### Engineering Principles

- Do **not** overfit to a small set of errors.
- Validate every new rule against the full dataset.
- Prevent regressions before accepting changes.
- Avoid data leakage between evaluation data and few-shot examples.
- Accept that some label noise is unavoidable.

---

## Runtime & Environment

- Production execution requires `DEEPSEEK_API_KEY` in `.env`.
  Taxonomy, prompt generation, and few-shot logic can still be tested without it.
- Model configuration and thresholds are defined in `config/settings.py` and `.env`.
- **Never** commit `.env` or `logs/interactions.jsonl` (contains PII).
