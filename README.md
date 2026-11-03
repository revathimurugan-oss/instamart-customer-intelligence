# 🛒 Instamart Customer Intelligence & Support Copilot

> An end-to-end offline pipeline covering customer segmentation, lapse-risk modelling,
> support-ticket triage, policy retrieval, and a tool-using refund agent —
> built entirely on synthetic data with no external API keys or network calls required.

---

## Table of Contents

1. [What This Project Tackles](#1-what-this-project-tackles)
2. [Setup & Run](#2-setup--run)
3. [Repository Map](#3-repository-map)
4. [Process Flow](#4-process-flow)
5. [Part A — Data Foundation](#5-part-a--data-foundation)
6. [Part B — Customer Intelligence](#6-part-b--customer-intelligence)
7. [Part C — Support Ticket Triage](#7-part-c--support-ticket-triage)
8. [Part D — Policy Retrieval & Refund Agent](#8-part-d--policy-retrieval--refund-agent)
9. [Key Output Files](#9-key-output-files)
10. [Limitations & Next Steps](#10-limitations--next-steps)

---

## 1. What This Project Tackles

Modern grocery-delivery platforms face three recurring operational challenges:

| Challenge | This Project's Answer |
|---|---|
| **Who are my customers and who is about to leave?** | RFM segmentation + Random Forest lapse-risk model |
| **How do I route 100s of support tickets quickly?** | Offline LLM classification with redaction, retry logic, and token budgeting |
| **Can an AI agent handle a refund end-to-end?** | ReAct tool-calling agent with checkpointing and safe fallbacks |

Every component runs fully offline using a deterministic mock LLM, a local SQLite database, and Python standard libraries plus a small set of pinned dependencies. No customer PII, no API keys, no cloud costs.

---

## 2. Setup & Run

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12 recommended (3.11+ works) |
| SQLite | 3.25 or later (bundled with Python) |
| OS | Windows / macOS / Linux |

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/instamart-cis-project.git
cd instamart-cis-project
```

### Step 2 — Create a virtual environment

**Windows PowerShell**
```powershell
py -3.12 -m venv .venv312
```

**macOS / Linux**
```bash
python3.12 -m venv .venv312
```

### Step 3 — Install dependencies

**Windows PowerShell**
```powershell
.\.venv312\Scripts\python.exe -m pip install --upgrade pip
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux**
```bash
.venv312/bin/python -m pip install --upgrade pip
.venv312/bin/python -m pip install -r requirements.txt
```

> **Note:** All commands below use the Windows form. Replace `.\.venv312\Scripts\python.exe` with `.venv312/bin/python` on macOS / Linux.

### Step 4 — Run the full pipeline in order

```powershell
# 1. Generate synthetic raw data and policy documents
.\.venv312\Scripts\python.exe generate_data.py

# 2. Clean and standardise the raw CSVs
.\.venv312\Scripts\python.exe clean_data.py

# 3. Validate integrity and load into SQLite
.\.venv312\Scripts\python.exe validate_data.py

# 4. Exploratory data analysis + SQL cross-checks
.\.venv312\Scripts\python.exe part_a_eda.py

# 5. Segmentation and lapse-risk model
.\.venv312\Scripts\python.exe part_b.py

# 6. Ticket classification and triage
.\.venv312\Scripts\python.exe part_c.py

# 7. Policy retrieval evaluation
.\.venv312\Scripts\python.exe part_d.py

# 8. Refund agent scenarios (imports part_d.py)
.\.venv312\Scripts\python.exe part_d_agent_run.py
```

Each step depends on the outputs of all previous steps. **Run them in order.**

### Step 5 — Launch the interactive dashboard

```powershell
.\.venv312\Scripts\streamlit.exe run dashboard.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 3. Repository Map

```
instamart-cis-project/
│
├── 📄 generate_data.py        ← Synthetic data generator (stdlib only)
├── 📄 clean_data.py           ← Raw → clean CSVs
├── 📄 validate_data.py        ← Integrity checks + SQLite loader
├── 📄 part_a_eda.py           ← EDA and SQL cross-validation
├── 📄 part_b.py               ← Segmentation + lapse-risk model
├── 📄 part_c.py               ← Ticket triage pipeline
├── 📄 part_d.py               ← Policy retrieval + refund agent
├── 📄 part_d_agent_run.py     ← Agent scenario runner (entry point)
├── 📄 mock_llm.py             ← Deterministic offline LLM stand-in
├── 📄 dashboard.py            ← Streamlit interactive dashboard
├── 📄 requirements.txt        ← Pinned Python dependencies
│
├── 📁 data/
│   ├── raw/                   ← Generated CSVs (tickets, orders, …)
│   ├── clean/                 ← Cleaned CSVs (customers, orders, products)
│   ├── policies/              ← Five plain-text policy documents
│   └── instamart.db           ← SQLite database (created by validate_data.py)
│
├── 📁 outputs/                ← All artefacts produced by Part A–D scripts
├── 📁 state/                  ← Agent JSON checkpoint files
├── 📁 sql/
│   └── queries.sql            ← Reference SQL for Part A analysis
└── 📁 docs/
    └── screenshots/           ← Dashboard screenshots
```

---

## 4. Process Flow

```
generate_data.py
      │
      ▼
  data/raw/                 ← 1 200 customers · 40 products
  data/policies/            ← 5 policy documents
      │
      ▼
clean_data.py  ──────────►  data/clean/   (deduped, typed, validated)
      │
      ▼
validate_data.py ────────►  data/instamart.db   (SQLite)
      │
      ├──► part_a_eda.py   ── EDA + SQL cross-checks ──► outputs/part_a_eda.txt
      │
      ├──► part_b.py       ── RFM · K-Means · RF/LR  ──► outputs/customer_segments.csv
      │                                                    outputs/lapse_model.joblib
      │
      ├──► part_c.py       ── Classify · Redact · Triage ► outputs/triage_results.csv
      │
      ├──► part_d.py       ── TF-IDF retrieval eval   ──► outputs/policy_eval_overlap*.csv
      │
      └──► part_d_agent_run.py ── ReAct agent ─────────► outputs/agent_runs.jsonl
                                                          outputs/refund_ledger.jsonl
                                                          state/*.json
```

---

## 5. Part A — Data Foundation

**Scripts:** `generate_data.py` → `clean_data.py` → `validate_data.py` → `part_a_eda.py`

### What was generated

| Table | Records |
|---|---|
| Customers | 1 200 |
| Products | 40 (5 categories) |
| Orders (raw) | ~12 400 rows including duplicates and planted errors |
| Support tickets | 164 |
| Policy documents | 5 text files |
| Policy eval questions | 14 (12 answerable + 2 unanswerable) |
| Agent scenarios | 7 |

### What was cleaned

- City names normalised (strip, title-case); invalid ages set to `NaN`
- Gender values mapped to `male` / `female`; anything else → `NaN`
- Negative delivery minutes → `NaN`; unparseable order amounts → `NaN`
- Duplicate orders resolved by keeping the latest timestamp record

### Cleaned dataset

- **1 200** unique customers · **40** products · **12 309** unique order IDs
- SQL and pandas status counts and monthly revenue cross-checks both pass ✅

---

## 6. Part B — Customer Intelligence

**Script:** `part_b.py`

### Task 1 — RFM features

Recency, Frequency, and Monetary values computed as of `2025-10-01` for all
**1 175** customers who placed at least one completed order.

### Task 2 — K-Means segmentation (K = 2 … 8)

Silhouette scores evaluated across seven values of K; results saved to
`outputs/kmeans_metrics.csv`. **K = 4** selected as the best balance of
separation and marketing interpretability.

| Segment | Description |
|---|---|
| Premium High-Value | High spend, recent, infrequent |
| Loyal Frequent | Moderate spend, very frequent |
| Cooling / At-Risk | Long since last order |
| Steady Customers | Consistent but low monetary |

### Task 3 — PCA visualisation

Two-component PCA applied to scaled RFM data; cluster map rendered in the
dashboard under **Part B**.

### Tasks 4–8 — Lapse-risk model

- **Cohort:** 1 156 customers with at least one completed order before `2025-09-01`.
  Label = `1` if no completed order in September.
- **Features:** recency, frequency, monetary, discount rate, cancellation rate,
  avg delivery time, age, tenure, city, gender, favourite category.
- **Models:** Logistic Regression and Random Forest, each tuned with 5-fold
  `GridSearchCV`; best model persisted to `outputs/lapse_model.joblib`.
- **Oversampling:** minority class oversampled on training rows only; test set
  never touched.
- **Top features** saved to `outputs/rf_top_features.csv`.

---

## 7. Part C — Support Ticket Triage

**Script:** `part_c.py`

### Pipeline overview

```
Load 164 tickets
      │
      ├── 4 empty  ──────────────────────────► skipped_empty
      ├── 4 long (> 400 tokens) ─────────────► flagged_long
      └── 156 eligible
            │
            ▼
      Redact PII (emails, phones, order IDs)
            │
            ▼
      Call mock LLM  ◄─── retry on TransientLLMError (max 3, exp. backoff)
            │
            ├── Valid JSON + allowed category ────► classified_json
            ├── Plain-text with single keyword ──► classified_keyword
            └── Anything else ───────────────────► flagged_invalid_output
                        │
                        ▼
               Priority + routing
               ┌─────────────────────────────────────┐
               │ fraud          → HIGH · security_team│
               │ premium cust.  → HIGH · premium_desk │
               │ human statuses → NORMAL · human_review│
               │ others         → NORMAL · support_queue│
               └─────────────────────────────────────┘
```

### Results

| Metric | Zero-shot | Few-shot |
|---|---|---|
| Classified tickets | 147 / 164 | 147 / 164 |
| Accuracy | **85.7 %** | **100.0 %** |
| Total tokens used | ~3 200 | ~4 100 |
| Recovered retries | 19 | 19 |

- **Redactions:** 55 emails · 91 phone numbers · 132 order IDs = **278 total**
- **Budget demo:** pipeline halts mid-run when the token ceiling is reached;
  remaining tickets are marked `budget_exhausted`.

---

## 8. Part D — Policy Retrieval & Refund Agent

**Scripts:** `part_d.py` · `part_d_agent_run.py`

### Policy retrieval (TF-IDF RAG)

Each of the five policy documents is split into **40-word chunks** with a
configurable word overlap.

| Config | Chunks | Doc retrieval (12 Q) | Fact match (12 Q) | Fallback (2 Q) |
|---|---|---|---|---|
| Overlap = 10 | 15 | 12 / 12 ✅ | 12 / 12 ✅ | 2 / 2 ✅ |
| Overlap = 0 | 11 | 12 / 12 ✅ | 10 / 12 ⚠️ | 2 / 2 ✅ |

Overlap = 0 misses two expected facts because sentence evidence is split across
chunk boundaries — demonstrating why overlapping chunks matter.

### Refund agent tools

| Tool | Purpose |
|---|---|
| `get_order` | Fetch order status, amount, delivery time from SQLite |
| `search_policy` | Retrieve top-3 relevant policy chunks (TF-IDF) |
| `calculate_refund` | Apply the 45-minute rule to compute eligible amount |
| `issue_refund` | Append auto-approved refund to `refund_ledger.jsonl` |
| `escalate_to_human` | Record refunds > INR 1 000 for human review |

### Safety limits

| Limit | Default |
|---|---|
| Max LLM calls per run | 6 |
| Token budget | 6 000 |
| Time limit | 10 s |
| Fallback if exceeded | `FALLBACK_MAX_ITERATIONS / TOKEN_LIMIT / TIMEOUT` |

### Scenario outcomes

| Scenario | Expected | Result |
|---|---|---|
| S1, S2 — late small refund (≤ INR 1 000) | `REFUND` | ✅ |
| S3 — late large refund (> INR 1 000) | `ESCALATE` | ✅ |
| S4 — on-time delivery | `DENIED` | ✅ |
| S5 — cancelled order | `DENIED` | ✅ |
| S6 — unknown order ID | `NOT_FOUND` | ✅ |
| S7 — infinite-loop trigger | `FALLBACK` | ✅ |

### Checkpointing & crash recovery

A JSON checkpoint is written to `state/` **after every tool observation**.
The crash-recovery demo:
1. Crashes after the 2nd tool observation.
2. Resumes from the saved checkpoint.
3. Completes `get_order → search_policy → calculate_refund → issue_refund`
   without repeating any already-executed step. ✅

---

## 9. Key Output Files

| File | Generated by | Description |
|---|---|---|
| `data/instamart.db` | `validate_data.py` | SQLite database |
| `outputs/part_a_eda.txt` | `part_a_eda.py` | EDA summary report |
| `outputs/customer_segments.csv` | `part_b.py` | RFM + cluster labels for all customers |
| `outputs/segment_profiles.csv` | `part_b.py` | Mean RFM stats per segment |
| `outputs/kmeans_metrics.csv` | `part_b.py` | Inertia + silhouette for K = 2…8 |
| `outputs/lapse_model.joblib` | `part_b.py` | Serialised best classifier |
| `outputs/rf_top_features.csv` | `part_b.py` | Top-5 Random Forest feature importances |
| `outputs/retention_list.csv` | `part_b.py` | Top lapse-risk premium customers |
| `outputs/triage_results.csv` | `part_c.py` | Final few-shot ticket classifications |
| `outputs/triage_budget_demo.csv` | `part_c.py` | Budget-limited run demonstration |
| `outputs/policy_eval_overlap10.csv` | `part_d.py` | Retrieval eval — overlap 10 |
| `outputs/policy_eval_overlap0.csv` | `part_d.py` | Retrieval eval — overlap 0 |
| `outputs/agent_runs.jsonl` | `part_d_agent_run.py` | Per-scenario agent run records |
| `outputs/refund_ledger.jsonl` | `part_d_agent_run.py` | Refund + escalation audit log |
| `state/*.json` | `part_d_agent_run.py` | Per-step agent checkpoints |

---

## 10. Limitations & Next Steps

### Current limitations

- All data is **synthetic**; metrics do not represent real-world production performance.
- The LLM is an **offline mock**; production behaviour with a real model will differ.
- The refund ledger is a **local JSONL log**, not a payment-system integration.
- TF-IDF retrieval is **lexical**; semantic search (embeddings) would handle paraphrasing better.
- Token counts are estimates from a word-count heuristic, not a real tokeniser.

### Suggested next steps

| Area | Suggestion |
|---|---|
| Retrieval | Replace TF-IDF with sentence-transformer embeddings + vector store |
| Classification | Fine-tune a small model on labelled ticket data |
| Agent | Compare ReAct vs plan-and-execute on the same seven scenarios |
| Observability | Add structured logging, latency tracking, and cost dashboards |
| Production readiness | Add auth, audit controls, DB transactions, privacy review, integration tests |
| Data | Expand the evaluation set; add multi-turn conversation scenarios |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Data processing | pandas 3.0.6 · numpy 2.5.4 |
| Machine learning | scikit-learn 1.9.1 · scipy 1.18.1 · joblib 1.6.0 |
| Dashboard | Streamlit 1.66.0 · Plotly 7.1.0 |
| Database | SQLite (stdlib `sqlite3`) |
| LLM | Offline mock (`mock_llm.py`) — no API key needed |

---

<p align="center">
  Built for the Instamart Customer Intelligence & Support Copilot capstone project.
</p>
