## HSTGT Stock Predictor (Heterogeneous Spatio-Temporal Graph Transformer)

This repository contains an end-to-end attempt at predicting next-day stock returns using a **Heterogeneous Spatio-Temporal Graph Transformer (HSTGT)**.

The core idea is to represent each trading day as a **heterogeneous graph** of stocks with multiple edge types:

- **Mathematical (dynamic)**: rolling-window Pearson correlation between stock returns
- **Structural (static)**: GICS sub-industry co-membership
- **Semantic (dynamic)**: **news-based FinBERT “sentiment spillover”** turned into edges between stocks discussed in the same news headlines

Training is done with **walk-forward / rolling-window optimization** to reduce lookahead bias.

> Model architecture
<img width="827" height="302" alt="image" src="https://github.com/user-attachments/assets/fbcd6de1-d50c-4bdd-88b8-7802fd4c97ab" />


## What I tried (project history)

1. **Baseline model (no semantic/news edges)**
   - Build daily heterogeneous graphs using:
     - rolling correlation edges (`stock -> mathematical -> stock`)
     - GICS sector structure edges (`stock -> structural -> stock`)
   - Train `HSTGT` with walk-forward splits.
   - Implemented in `run_pipeline.py` (baseline) + `src/` model/training modules.

2. **Semantic/news extension (FinBERT)**
   - Download news headlines per stock ticker (using Google News RSS).
   - Run FinBERT on every headline.
   - For each date, create semantic edges between tickers that both had sufficiently “strong” sentiment signal on that date.
   - Rebuild graphs for dates where semantic edges exist.
   - Retrain the same `HSTGT` architecture and compare against baseline metrics.
   - Implemented in the repo’s `step1.py` → `step2.py` → `step3.py` → `step4.py` scripts.

3. **Alternative/older semantic pipelines (kept in code, may be redundant)**
   - There are additional scripts/modules for semantic edge generation (e.g., `src/data_pipeline/run_finbert.py` and `src/data_pipeline/semantic_edges.py`) that reflect earlier iterations.
   - For a clean GitHub repo, you likely want to keep only one canonical path in the README and archive/remove the older variants (details below).

## Repo layout (main pieces)

- `src/models/`
  - `hstgt.py`: Temporal encoder + HGTConv-based heterogeneous spatio-temporal blocks + prediction head
  - `temporal_encoder.py`: Transformer encoder with sinusoidal positional encoding and a CLS token
- `src/training/`
  - `loss.py`: `RiskAwareLoss` (MSE + directional-accuracy penalty)
  - `walk_forward.py`: rolling-window training + validation with early stopping + test metrics (RMSE, directional accuracy, Sharpe)
- `src/data_pipeline/`
  - `download_data.py`: get S&P 500 tickers + GICS mapping + OHLCV download
  - `feature_engineering.py`: build technical indicator features + next-day target return
  - `graph_builder.py`: build daily `torch_geometric.data.HeteroData` graphs with 3 edge types
- Root scripts:
  - `run_pipeline.py`: baseline pipeline (no semantic edges)
  - `step1.py`..`step4.py`: semantic pipeline (news → FinBERT spillover edges → graph rebuild → semantic retraining)
- `app/streamlit_demo.py`: Streamlit UI for predictions, model performance charts, and graph exploration

## Project structure

```text
.
├── app/
├── src/
├── data/
│   ├── raw/
│   ├── processed/
│   └── graphs/
├── results/
├── .gitignore
├── README.md
├── requirements.txt
├── run_pipeline.py
├── step1.py
├── step2.py
├── step3.py
└── step4.py
```

## Baseline training (no semantic edges)

Run:

```bash
python run_pipeline.py
```

What it does:

1. Downloads OHLCV for a subset of S&P 500 tickers via `yfinance` (`src/data_pipeline/download_data.py`)
2. Builds features + targets (`src/data_pipeline/feature_engineering.py`)
3. Builds daily graphs (`src/data_pipeline/graph_builder.py`)
   - Semantic edges are absent because no semantic scores file is passed.
4. Walk-forward trains `HSTGT` and writes:
   - `results/walk_forward_results.csv`
   - fold checkpoints in `results/`

## Semantic/news pipeline (FinBERT edges)

This is implemented as four sequential steps:

### Step 1: Download news headlines

Run:

```bash
python step1.py
```

Outputs:

- `data/raw/news/all_news_raw.parquet`

Implementation notes:

- Uses Google News RSS (no auth).
- Produces records with: `ticker`, `date`, `headline`, `source`, `url`.

### Step 2: FinBERT scoring + semantic edge generation

Run:

```bash
python step2.py
```

Inputs:

- `data/raw/news/all_news_raw.parquet`

Outputs:

- `data/processed/semantic_edges.parquet`

How semantic edges are built (high level):

- Each headline gets a FinBERT spillover score `spillover = P(positive) + P(negative)`
- For each date:
  - the best (max) spillover score per ticker is used
  - edge weight is the geometric mean across the two tickers
  - edges below a minimum threshold are dropped

### Step 3: Rebuild graphs for semantic edge dates

Run:

```bash
python step3.py
```

Outputs:

- `data/graphs/<date>.pkl` rebuilt/created for dates with semantic edges

### Step 4: Retrain with semantic edges + compare to baseline

Run:

```bash
python step4.py
```

Outputs:

- `results/semantic/walk_forward_results.csv`
- fold checkpoints in `results/semantic/`

## Streamlit demo

Run:

```bash
streamlit run app/streamlit_demo.py
```

The app expects (by default):

- Baseline: `results/walk_forward_results.csv`
- Semantic: `results/semantic/walk_forward_results.csv`
- Graphs: `data/graphs/`

It provides:

- Next-day predicted vs actual scatter plot
- Walk-forward evaluation curves
- Graph explorer showing edge counts by type and a simple visualization

## Important notes / limitations

1. **Edge weights are stored but may not be used by the model**
   - `src/data_pipeline/graph_builder.py` stores `edge_attr` for semantic edges.
   - In `src/models/hstgt.py`, message passing uses `HGTConv` with `edge_index_dict` (and does not consume `edge_attr` directly).
   - If you want weighted semantic edges, you may need to modify the model to incorporate edge weights.

2. **Semantic edges can be sparse**
   - Semantic edges depend on news coverage and the FinBERT threshold.
   - It’s normal for some dates to have `0` semantic edges.

3. **Runtime**
   - FinBERT scoring is the slowest part.
   - Walk-forward training is compute-heavy.
