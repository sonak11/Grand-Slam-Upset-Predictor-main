# 🎾 Grand Slam Upset Predictor

> Predicts whether the lower-ranked player will defeat the higher-ranked player in a Grand Slam tennis match by engineering a novel **Cumulative Tournament Fatigue Index (CTFI)** — the first within-tournament fatigue feature ever used for this task.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Setup in VS Code (Step-by-Step)](#setup-in-vs-code)
4. [Running the Pipeline](#running-the-pipeline)
5. [Launching the Streamlit Dashboard](#launching-the-streamlit-dashboard)
6. [Important Notes on Runtime](#important-notes-on-runtime)
7. [Dataset Description](#dataset-description)
8. [Key Results](#key-results)

---

## Project Overview

This is a 6-phase end-to-end machine learning pipeline:

| Phase | Script | What it does |
|-------|--------|-------------|
| 1 | `data_ingestion.py` | Downloads ATP + WTA match CSVs (1990–2025) → 3NF SQLite database |
| 2 | `scraping.py` | Scrapes post-match press conference transcripts from ASAP Sports |
| 3 | `nlp.py` | 3-layer NLP: rule-based lexicon + DistilBERT + LLM zero-shot fatigue classification |
| 4 | `features.py` | Computes CTFI via SQL window function, merges NLP features → `features.csv` |
| 5 | `model.py` | 4-model ablation, SMOTE, temporal split, SHAP, McNemar's test → 9 plots |
| 6 | `clustering.py` | K-Means player archetype clustering + PCA visualization |

Plus:
- **`app.py`** — Streamlit interactive dashboard
- **`agent_service.py` + `rag_service.py`** — RAG Q&A agent (ChromaDB + LangChain + Groq)
- **`run_pipeline.py`** — Master pipeline runner

---

## File Structure

```
grand-slam-upset-predictor/
├── data_ingestion.py       # Phase 1
├── scraping.py             # Phase 2
├── nlp.py                  # Phase 3
├── features.py             # Phase 4
├── model.py                # Phase 5
├── clustering.py           # Phase 6
├── run_pipeline.py         # Orchestrator
├── app.py                  # Streamlit dashboard
├── agent_service.py        # LLM Q&A agent
├── rag_service.py          # ChromaDB RAG service
├── prediction_service.py   # Real-time prediction API
├── requirements.txt        # All dependencies
│
├── features.csv            # Final feature matrix (9,876 rows × 26 cols)
├── nlp_features.csv        # NLP output per transcript
├── real_event_ids.json     # Tournament event IDs for scraping
│
├── roc_curves_all.png
├── precision_recall_curves.png
├── confusion_matrices.png
├── calibration_curves.png
├── correlation_heatmap.png
├── ctfi_upset_by_surface.png
├── shap_importance.png
├── metric_summary.png
└── README.md
```

---

## Setup in VS Code

### Prerequisites
- Python 3.10 or 3.11
- VS Code with the **Python extension** installed
- Git (optional, for cloning)

### Step 1 — Open the project in VS Code

1. Open VS Code
2. Go to **File → Open Folder**
3. Select the project folder (wherever you unzipped or cloned it)

### Step 2 — Create a virtual environment

Open the **integrated terminal** in VS Code (`Ctrl+`` ` `` or Terminal → New Terminal) and run:

```bash
# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including pandas, scikit-learn, imbalanced-learn, SHAP, Streamlit, sentence-transformers, ChromaDB, LangChain, Groq, and more.

> **Note:** The first install may take 3–5 minutes due to large packages like `sentence-transformers` and `torch`.

### Step 4 — Select the virtual environment as your Python interpreter

1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type `Python: Select Interpreter`
3. Choose the interpreter from your `venv` folder (it will say something like `./venv/bin/python`)

### Step 5 — (Optional) Install spaCy language model

If running the NLP pipeline:
```bash
python -m spacy download en_core_web_sm
```

---

## Running the Pipeline

### Option A — Run the full pipeline (all 6 phases)

```bash
python run_pipeline.py
```

### Option B — Run from a specific phase (skip what's already done)

```bash
# Start from Phase 5 (modeling) — skip data download and NLP
python run_pipeline.py --from 5

# Start from Phase 3 (NLP) — data already ingested
python run_pipeline.py --from 3
```

### Option C — Common flags

```bash
# Skip web scraping (transcripts already in database)
python run_pipeline.py --skip-scrape

# Enable LLM zero-shot classification (requires HuggingFace API access)
python run_pipeline.py --llm

# Skip unsupervised clustering (Phase 6)
python run_pipeline.py --no-cluster

# Combine flags
python run_pipeline.py --from 4 --no-cluster
```

### Option D — Run individual phases

```bash
python data_ingestion.py   # Phase 1
python scraping.py         # Phase 2
python nlp.py              # Phase 3
python features.py         # Phase 4
python model.py            # Phase 5
python clustering.py       # Phase 6
```

---

## ⚠️ Important Notes on Runtime

| Phase | Estimated Time | Reason |
|-------|---------------|--------|
| Phase 1 — Data ingestion | 5–10 minutes | Downloading 72 CSV files from GitHub |
| **Phase 2 — Scraping** | **1–4 hours** | **Web scraping ASAP Sports is slow by design — rate-limited to avoid being blocked. This is the longest phase.** |
| Phase 3 — NLP | 15–45 minutes | DistilBERT and LLM inference on 2,279 transcripts |
| Phase 4 — Feature engineering | 2–5 minutes | SQL window functions + pandas merge |
| Phase 5 — Modeling | 10–20 minutes | Grid search CV across 4 models |
| Phase 6 — Clustering | 1–2 minutes | K-Means + PCA |

> **If you want to skip scraping entirely** and use the pre-existing data: run `python run_pipeline.py --from 3 --skip-scrape` to start from the NLP phase using whatever transcripts are already in the database.

> **If you want to skip everything and just run the model**: `features.csv` is already included in the repository. Run `python model.py` directly to train and evaluate all four models on the existing feature matrix.

---

## Launching the Streamlit Dashboard

After running at least Phase 5 (so `upset_model.pkl` exists):

```bash
streamlit run app.py
```

This opens the interactive dashboard in your browser at `http://localhost:8501`.

**Dashboard features:**
- **Upset Probability Calculator**: Enter player rankings, CTFI values, surface, and round → get real-time upset probability
- **RAG Q&A Agent**: Ask natural language questions about matches and transcripts (requires Groq API key)
- **Results Viewer**: Interactive display of all evaluation plots and clustering results

---

## Dataset Description

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `features.csv` | Final modelling matrix | 9,876 | 26 |
| `nlp_features.csv` | NLP output per transcript | 2,279 | 19 |
| `tennis_upsets.db` | SQLite 3NF database | — | 4 tables |

**Key features:**
- `ctfi_minutes` — Cumulative court minutes accumulated before the current match (novel feature)
- `ctfi_diff_minutes` — Player CTFI minus opponent CTFI (relative fatigue)
- `log_rank_diff` — Signed log of ranking gap
- `upset` — Target: 1 = lower-ranked player won

---

## Key Results

| Model | PR-AUC | ROC-AUC | F1 (upset) |
|-------|--------|---------|-----------|
| LR Baseline (rank + CTFI) | 0.363 | 0.714 | 0.342 |
| RF — No CTFI (rank only) | 0.369 | 0.705 | 0.331 |
| RF — Traditional (rank + CTFI) | 0.398 | 0.738 | 0.370 |
| **RF — Full (rank + CTFI + NLP)** | **0.412** | **0.748** | **0.384** |
| No-skill baseline | 0.182 | 0.500 | — |

- **CTFI lift**: +0.029 PR-AUC over rank-only, **statistically significant** (McNemar's, p = 0.022)
- **SHAP**: CTFI-differential ranks **3rd of 27 features**, above round number and player rank
- **Clustering**: High-ranked high-CTFI players (Cluster 3) have **2.7× the upset rate** of high-ranked rested players (Cluster 0)

---

## Academic Integrity

All code is original. External libraries are cited in `requirements.txt`. Data is sourced from publicly licensed datasets (Jeff Sackmann, MIT license) and public web archives (ASAP Sports). Built entirely by a single author.