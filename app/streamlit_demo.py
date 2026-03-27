"""
HSTGT Streamlit Demo — updated with semantic edge support.
Replace app/streamlit_demo.py with this file.
Run: streamlit run app/streamlit_demo.py
"""

import os, sys, pickle, json
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.hstgt import HSTGT

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "HSTGT Stock Predictor",
    page_icon  = "📈",
    layout     = "wide",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ HSTGT Configuration")
st.sidebar.markdown("---")

model_path = st.sidebar.text_input(
    "Model Checkpoint Path",
    value = "results/fold_1_model.pt",
    help  = "e.g. results/fold_31_model.pt or results/semantic/fold_1_model.pt"
)
graph_dir = st.sidebar.text_input(
    "Cached Graphs Directory",
    value = "data/graphs/",
)
d_model    = st.sidebar.slider("d_model (must match training)", 32, 256, 64, step=32)
num_blocks = st.sidebar.slider("HSTGT Blocks (must match training)", 1, 4, 2)

# ── Helpers ────────────────────────────────────────────────────────────────────

DEFAULT_BROWSE_DAYS = 30


@st.cache_data
def load_all_graph_dates(graph_dir: str) -> list[str]:
    if not os.path.exists(graph_dir):
        return []
    return sorted([
        f.replace(".pkl", "")
        for f in os.listdir(graph_dir)
        if f.endswith(".pkl")
    ])


def parse_fold_num_from_path(checkpoint_path: str) -> int | None:
    """Best-effort fold number parsing from `.../fold_<n>_model.pt`."""
    try:
        return int(checkpoint_path.split("fold_")[1].split("_")[0])
    except Exception:
        return None


def get_default_date_options(all_dates: list[str], n: int = DEFAULT_BROWSE_DAYS) -> list[str]:
    if not all_dates:
        return []
    return all_dates[-min(n, len(all_dates)) :]


@st.cache_resource
def load_model(path: str, d_model: int, num_blocks: int):
    try:
        state = torch.load(path, map_location="cpu")
        model = HSTGT(
            input_dim      = 8,
            d_model        = d_model,
            seq_len        = 30,
            nhead_temporal = 4,
            nhead_spatial  = 4,
            num_blocks     = num_blocks,
            dropout        = 0.0,   # eval mode — no dropout
        )
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None


@st.cache_data
def load_graph_cached(graph_dir: str, date_str: str):
    path = os.path.join(graph_dir, f"{date_str}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def run_inference(model, graph):
    with torch.no_grad():
        preds = model(graph)
    return preds.numpy()


def get_edge_counts(graph) -> dict:
    counts = {"mathematical": 0, "structural": 0, "semantic": 0}
    for store in graph.edge_stores:
        if hasattr(store, "_key") and store._key:
            et = store._key[1]
            if et in counts and hasattr(store, "edge_index"):
                counts[et] = int(store.edge_index.shape[1])
    return counts


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

st.title("📈 HSTGT — Heterogeneous Spatio-Temporal Graph Transformer")
st.markdown(
    "**Graph-Aware Stock Return Prediction** · "
    "Multi-relational dynamic graphs · "
    "Interleaved Temporal + Spatial Transformer attention"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Predictions",
    "📊 Model Performance",
    "🕸️ Graph Explorer",
    "ℹ️ Architecture",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Next-Day Return Predictions")

    all_dates = load_all_graph_dates(graph_dir)

    if not all_dates:
        st.error(f"No graphs found in `{graph_dir}`. Run the pipeline first.")
        st.stop()

    fold_num = parse_fold_num_from_path(model_path)

    col_left, col_right = st.columns([3, 1])
    with col_right:
        show_all = st.checkbox("Browse all dates", value=False)

    with col_left:
        if show_all:
            date_options = all_dates
            st.warning(
                "Showing all cached graph dates. "
                "If you evaluate on dates outside the checkpoint’s test window, "
                "treat the numbers as exploratory (not a fair backtest)."
            )
        else:
            date_options = get_default_date_options(all_dates)
            if fold_num is not None and date_options:
                st.caption(f"Checkpoint fold parsed from path: **{fold_num}** (date filtering is disabled by default).")

    selected_date = st.selectbox("Select Trading Date", date_options, index=len(date_options) - 1)

    # ── Load graph and show edge preview ──────────────────────────────────
    graph = load_graph_cached(graph_dir, selected_date)

    if graph is None:
        st.error(f"Graph for {selected_date} not found.")
        st.stop()

    edge_counts = get_edge_counts(graph)
    n_stocks    = int(graph["stock"].x.shape[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks",              n_stocks)
    c2.metric("Math Edges",          edge_counts["mathematical"])
    c3.metric("Structural Edges",    edge_counts["structural"])

    sem_count = edge_counts["semantic"]
    sem_delta = "✅ FinBERT active" if sem_count > 0 else "⚠️ Run step1-3 to enable"
    c4.metric("Semantic Edges",      sem_count, delta=sem_delta,
              delta_color="normal" if sem_count > 0 else "off")

    st.markdown("---")

    # ── Run inference ──────────────────────────────────────────────────────
    if st.button("🔮 Run Inference", type="primary"):
        model = load_model(model_path, d_model, num_blocks)

        if model is None:
            st.stop()

        with st.spinner("Running HSTGT forward pass..."):
            preds   = run_inference(model, graph)
            actuals = graph["stock"].y.numpy()
            tickers = graph["stock"].tickers

        da   = np.mean(np.sign(preds) == np.sign(actuals))
        rmse = np.sqrt(np.mean((preds - actuals) ** 2))

        # ── Top metrics ────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Directional Accuracy", f"{da:.1%}",
                  delta=f"{(da-0.5)*100:+.1f}pp vs random")
        m2.metric("Stocks Predicted", str(len(preds)))
        m3.metric("RMSE", f"{rmse:.4f}")

        correct_dir = int(np.sum(np.sign(preds) == np.sign(actuals)))
        m4.metric("Correct Direction", f"{correct_dir} / {len(preds)}")

        # ── Scatter plot ────────────────────────────────────────────────
        colours = ["#22c55e" if p*a > 0 else "#ef4444"
                   for p, a in zip(preds, actuals)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actuals, y=preds,
            mode="markers+text",
            text=tickers[:len(preds)],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(color=colours, size=9, opacity=0.8,
                        line=dict(width=1, color="white")),
            name="Stocks",
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Actual: %{x:.3%}<br>"
                "Predicted: %{y:.3%}<extra></extra>"
            ),
        ))
        lim = max(float(np.abs(actuals).max()), float(np.abs(preds).max())) * 1.2
        fig.add_trace(go.Scatter(
            x=[-lim, lim], y=[-lim, lim],
            mode="lines",
            line=dict(dash="dash", color="rgba(255,255,255,0.3)", width=1),
            name="Perfect Prediction",
            hoverinfo="skip",
        ))

        fig.add_shape(type="rect", x0=0, y0=0, x1=lim, y1=lim,
                      fillcolor="rgba(34,197,94,0.05)", line_width=0)
        fig.add_shape(type="rect", x0=-lim, y0=-lim, x1=0, y1=0,
                      fillcolor="rgba(34,197,94,0.05)", line_width=0)

        fig.update_layout(
            title=f"Predicted vs Actual Returns — {selected_date}"
                  f"{'  |  FinBERT Semantic Edges Active 🟢' if sem_count > 0 else ''}",
            xaxis_title="Actual Log Return",
            yaxis_title="Predicted Log Return",
            xaxis=dict(tickformat=".1%"),
            yaxis=dict(tickformat=".1%"),
            height=520,
            legend=dict(x=0.01, y=0.99),
        )
        st.plotly_chart(fig, use_container_width=True)


        st.subheader("Signal Table")
        pred_df = pd.DataFrame({
            "Ticker":           tickers[:len(preds)],
            "Predicted Return": preds,
            "Actual Return":    actuals[:len(preds)],
            "Signal":           ["🟢 LONG" if p > 0 else "🔴 SHORT" for p in preds],
            "Correct":          ["✅" if p*a > 0 else "❌"
                                 for p, a in zip(preds, actuals[:len(preds)])],
        }).sort_values("Predicted Return", ascending=False)

        pred_df["Predicted Return"] = pred_df["Predicted Return"].map("{:.3%}".format)
        pred_df["Actual Return"]    = pred_df["Actual Return"].map("{:.3%}".format)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Top 10 LONG signals**")
            st.dataframe(pred_df.head(10), use_container_width=True, hide_index=True)
        with col_r:
            st.markdown("**Top 10 SHORT signals**")
            st.dataframe(pred_df.tail(10).iloc[::-1], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Walk-Forward Evaluation Results")

    orig_path = "results/walk_forward_results.csv"
    sem_path  = "results/semantic/walk_forward_results.csv"

    has_orig = os.path.exists(orig_path)
    has_sem  = os.path.exists(sem_path)

    if not has_orig and not has_sem:
        st.info("No results found. Run the training pipeline first.")
        st.stop()

    # ── Summary metrics ────────────────────────────────────────────────────
    if has_orig:
        orig_df = pd.read_csv(orig_path)
        orig_df["model"] = "Baseline (no semantic)"

    if has_sem:
        sem_df = pd.read_csv(sem_path)
        sem_df["model"] = "With FinBERT Semantic Edges"

    if has_orig and has_sem:
        st.subheader("Baseline vs FinBERT Semantic Edges")

        c1, c2, c3 = st.columns(3)
        da_delta = (sem_df["da"].mean() - orig_df["da"].mean()) * 100
        sh_delta = sem_df["sharpe"].mean() - orig_df["sharpe"].mean()
        rm_delta = sem_df["rmse"].mean() - orig_df["rmse"].mean()

        c1.metric("Semantic Mean DA",     f"{sem_df['da'].mean():.1%}",
                  delta=f"{da_delta:+.1f}pp vs baseline")
        c2.metric("Semantic Mean Sharpe", f"{sem_df['sharpe'].mean():.2f}",
                  delta=f"{sh_delta:+.2f} vs baseline")
        c3.metric("Semantic Mean RMSE",   f"{sem_df['rmse'].mean():.4f}",
                  delta=f"{rm_delta:+.4f} vs baseline", delta_color="inverse")

        combined = pd.concat([orig_df, sem_df])

        # DA comparison
        fig_da = go.Figure()
        colours = {"Baseline (no semantic)": "#6b7280",
                   "With FinBERT Semantic Edges": "#22c55e"}
        for name, grp in combined.groupby("model"):
            fig_da.add_trace(go.Scatter(
                x=grp["fold"], y=grp["da"],
                name=name, mode="lines+markers",
                line=dict(color=colours.get(name, "steelblue"), width=2),
                marker=dict(size=7),
            ))
        fig_da.add_hline(y=0.5, line_dash="dash", line_color="red",
                         annotation_text="50% random baseline")
        fig_da.update_layout(
            title="Directional Accuracy: Baseline vs FinBERT",
            yaxis=dict(tickformat=".0%", title="Directional Accuracy"),
            xaxis_title="Fold",
            height=380,
        )
        st.plotly_chart(fig_da, use_container_width=True)

        # Sharpe comparison
        fig_sh = go.Figure()
        for name, grp in combined.groupby("model"):
            fig_sh.add_trace(go.Bar(
                x=grp["fold"], y=grp["sharpe"],
                name=name,
                marker_color=colours.get(name, "steelblue"),
                opacity=0.8,
            ))
        fig_sh.add_hline(y=0, line_dash="dash", line_color="white", line_width=1)
        fig_sh.add_hline(y=1, line_dash="dot", line_color="green",
                         annotation_text="Sharpe=1.0")
        fig_sh.update_layout(
            title="Sharpe Ratio per Fold",
            barmode="group", height=380,
        )
        st.plotly_chart(fig_sh, use_container_width=True)

        with st.expander("Full results table"):
            st.dataframe(combined.round(4), use_container_width=True)

    elif has_orig:
        # Only baseline results
        st.subheader("Baseline Results (no semantic edges yet)")
        st.info("Complete steps 1-4 to see semantic edge comparison here.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Directional Accuracy", f"{orig_df['da'].mean():.1%}")
        c2.metric("Mean Sharpe Ratio",          f"{orig_df['sharpe'].mean():.2f}")
        c3.metric("Mean RMSE",                  f"{orig_df['rmse'].mean():.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=orig_df["fold"], y=orig_df["da"],
            mode="lines+markers", name="Directional Accuracy",
            line=dict(color="steelblue", width=2),
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                      annotation_text="50% random")
        fig.update_layout(
            title="Directional Accuracy Across Folds",
            yaxis=dict(tickformat=".0%"), height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(orig_df, x="fold", y="rmse",
                      title="RMSE per Fold (lower = better)",
                      color="rmse", color_continuous_scale=["green","orange","red"])
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(orig_df.round(4), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GRAPH EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Heterogeneous Graph Explorer")
    st.markdown("Select any date to inspect the live graph structure for that day.")

    all_dates_g = load_all_graph_dates(graph_dir)
    if not all_dates_g:
        st.error("No graphs found.")
        st.stop()

    explore_date = st.selectbox("Select Date", all_dates_g,
                                 index=len(all_dates_g)-1, key="explore_date")

    g = load_graph_cached(graph_dir, explore_date)
    if g is None:
        st.error("Graph not found.")
        st.stop()

    ec       = get_edge_counts(g)
    n_stocks = int(g["stock"].x.shape[0])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks (Nodes)",   n_stocks)
    c2.metric("Math Edges",       ec["mathematical"],
              help="Rolling 30-day Pearson correlation > 0.6")
    c3.metric("Structural Edges", ec["structural"],
              help="Same GICS sub-industry")
    c4.metric("Semantic Edges",   ec["semantic"],
              delta="FinBERT active ✅" if ec["semantic"] > 0 else "Run steps 1-3 to enable",
              delta_color="normal" if ec["semantic"] > 0 else "off")

    if ec["semantic"] == 0:
        st.info(
            "💡 Semantic edges are 0 for this date. "
            "After running steps 1-3, dates within the news coverage window "
            "will show non-zero semantic edges."
        )

    # ── Graph visualisation ────────────────────────────────────────────────
    st.subheader(f"Graph Structure — {explore_date}")

    tickers_vis = g["stock"].tickers[:30]
    n = len(tickers_vis)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos_x  = np.cos(angles)
    pos_y  = np.sin(angles)

    fig_g = go.Figure()

    edge_styles = {
        "mathematical": dict(color="#3b82f6", dash="solid",  width=1.5),
        "structural":   dict(color="#22c55e", dash="dot",    width=1.5),
        "semantic":     dict(color="#f97316", dash="dashdot", width=2.0),
    }

    for store in g.edge_stores:
        if not (hasattr(store, "_key") and store._key):
            continue
        et = store._key[1]
        if et not in edge_styles:
            continue
        if not hasattr(store, "edge_index") or store.edge_index.shape[1] == 0:
            continue

        ei    = store.edge_index
        style = edge_styles[et]
        ex, ey = [], []
        shown = 0
        for k in range(ei.shape[1]):
            s, d = int(ei[0, k]), int(ei[1, k])
            if s < n and d < n:
                ex += [pos_x[s], pos_x[d], None]
                ey += [pos_y[s], pos_y[d], None]
                shown += 1
            if shown >= 80:   # cap for readability
                break

        fig_g.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color=style["color"], width=style["width"],
                      dash=style["dash"]),
            opacity=0.5, name=f"{et} ({ec[et]} edges)",
        ))

    # Nodes
    fig_g.add_trace(go.Scatter(
        x=pos_x, y=pos_y,
        mode="markers+text",
        text=tickers_vis,
        textposition="top center",
        textfont=dict(size=9, color="white"),
        marker=dict(size=10, color="#6366f1",
                    line=dict(width=1, color="white")),
        name="Stocks",
    ))

    fig_g.update_layout(
        height=580,
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
        ),
    )
    st.plotly_chart(fig_g, use_container_width=True)

    # ── Edge density over time ─────────────────────────────────────────────
    st.subheader("Edge Density Over Time")
    st.markdown("Shows how the graph changes dynamically day by day.")

    sample_dates = all_dates_g[::5][-60:]  # Last 60 sampled dates
    density_records = []

    prog = st.progress(0, text="Loading graph statistics...")
    for i, d in enumerate(sample_dates):
        gg = load_graph_cached(graph_dir, d)
        if gg:
            ec2 = get_edge_counts(gg)
            density_records.append({"date": d, **ec2})
        prog.progress((i+1)/len(sample_dates), text=f"Loading {d}...")
    prog.empty()

    if density_records:
        density_df = pd.DataFrame(density_records)
        fig_d = go.Figure()
        for et, colour in [("mathematical","#3b82f6"),
                            ("structural","#22c55e"),
                            ("semantic","#f97316")]:
            fig_d.add_trace(go.Scatter(
                x=density_df["date"], y=density_df[et],
                mode="lines", name=et,
                line=dict(color=colour, width=2),
            ))
        fig_d.update_layout(
            title="Edge Counts Over Time (sampled every 5 days)",
            xaxis_title="Date",
            yaxis_title="Number of Edges",
            height=350,
        )
        st.plotly_chart(fig_d, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Model Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### Forward Pass
```
Input: HeteroData (50 stocks × 30 days × 8 features)
            │
    ┌───────▼────────┐
    │ Temporal       │  Transformer reads each
    │ Encoder        │  stock's 30-day history
    │ (CLS token)    │  independently → (50, 64)
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │ HSTGT Block 1  │
    │  ├─ Temporal   │  Refine embeddings
    │  │  Attn       │
    │  └─ HGTConv    │  3 edge types:
    │     ├─ Math    │  correlation
    │     ├─ Struct  │  sector
    │     └─ Semantic│  news (FinBERT)
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │ HSTGT Block 2  │  2-hop propagation
    └───────┬────────┘
            │
    ┌───────▼────────┐
    │ Output MLP     │  (50,64)→(50,32)
    │ Linear→GELU    │  →(50,1)
    └───────┬────────┘
            │
    Predicted returns (50,)
```
        """)

    with col2:
        st.markdown("""
### Loss Function
```
L_total = λ₁·L_MSE + λ₂·L_Dir

L_MSE = mean((y - ŷ)²)
  → penalises magnitude errors

L_Dir = mean(max(0, −y·ŷ))
  → fires ONLY when sign is wrong
  → getting direction wrong is
    penalised separately

Both λ₁ = λ₂ = 1.0 in this run
```

### Edge Types
| Type | Source | Dynamic? |
|------|--------|----------|
| Mathematical | 30-day Pearson corr | ✅ Daily |
| Structural | GICS sub-industry | ❌ Static |
| Semantic | FinBERT on yfinance news | ✅ Daily |

### Training Protocol
```
Walk-forward rolling windows:
  Train:  500 days
  Val:    60 days  (early stopping)
  Test:   30 days  (never touched)
  Stride: 30 days  → 38 folds total

Optimiser: AdamW (lr=1e-4, wd=1e-4)
Scheduler: CosineAnnealingLR
Grad clip: max_norm = 1.0
```
        """)

    st.markdown("---")
    st.subheader("FinBERT Pipeline")
    st.markdown("""
```
yfinance headlines
        │
        ▼
  FinBERT (ProsusAI/finbert)
  Input:  headline text (max 128 tokens)
  Output: P(positive), P(negative), P(neutral)
        │
        ▼
  spillover_score = √(score_A × score_B)
  (geometric mean of the two tickers' max daily scores)
        │
        ▼
  semantic_edges.parquet
  [date, ticker_a, ticker_b, spillover_score]
        │
        ▼
  Injected as edge_attr into
  (stock, semantic, stock) edge type
  in each day's HeteroData graph
```
    """)