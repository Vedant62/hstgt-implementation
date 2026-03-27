"""
Build NEW graphs for semantic edge dates that don't exist yet,
AND rebuild existing ones. The original pipeline ended at 2024-12-31 but
Google News gives 2025-2026 dates — so we need to create those graphs fresh.
"""

import os, sys, pickle
import pandas as pd
import yfinance as yf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEMANTIC_PATH = "data/processed/semantic_edges.parquet"
GRAPHS_DIR    = "data/graphs"
FEATURES_DIR  = "data/processed/features"
OHLCV_DIR     = "data/raw/ohlcv"


def extend_features_to_date(tickers: list, end_date: str):
    """
    Download any missing OHLCV data and recompute features
    up to end_date for each ticker.
    """
    print(f"\nExtending OHLCV + features up to {end_date}...")
    from src.data_pipeline.feature_engineering import (
        compute_log_returns, compute_rsi, compute_macd,
        compute_bollinger_bands, compute_rolling_volatility
    )
    import numpy as np

    for ticker in tqdm(tickers, desc="Extending features"):
        feat_path = os.path.join(FEATURES_DIR, f"{ticker}.parquet")
        ohlcv_path = os.path.join(OHLCV_DIR, f"{ticker}.csv")

        # Check if features already go up to end_date
        if os.path.exists(feat_path):
            existing = pd.read_parquet(feat_path)
            last_date = str(existing.index.max().date())
            if last_date >= end_date:
                continue   # already up to date

        # Re-download OHLCV with extended end date
        try:
            df = yf.download(
                ticker,
                start  = "2015-01-01",
                end    = end_date,
                progress = False,
                auto_adjust = True,
            )
            if df.empty:
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            df.to_csv(ohlcv_path)

            # Recompute features
            feat = pd.DataFrame(index=df.index)
            feat["open"]   = df["open"]
            feat["high"]   = df["high"]
            feat["low"]    = df["low"]
            feat["close"]  = df["close"]
            feat["volume"] = (
                (df["volume"] - df["volume"].rolling(30).mean()) /
                (df["volume"].rolling(30).std() + 1e-9)
            )
            feat["log_return"]     = compute_log_returns(df["close"])
            feat["volatility_30d"] = compute_rolling_volatility(feat["log_return"])
            feat["rsi_14"]         = compute_rsi(df["close"])
            macd, macd_sig         = compute_macd(df["close"])
            feat["macd"]           = macd
            feat["macd_signal"]    = macd_sig
            bb_u, _, bb_l          = compute_bollinger_bands(df["close"])
            feat["bb_upper"]       = (bb_u - df["close"]) / (df["close"] + 1e-9)
            feat["bb_lower"]       = (df["close"] - bb_l) / (df["close"] + 1e-9)
            feat["target_return"]  = feat["log_return"].shift(-1)
            feat.dropna(inplace=True)
            feat.to_parquet(feat_path)

        except Exception as e:
            print(f"  ⚠️  {ticker}: {e}")

    print("Feature extension done.")


def main():
    print("=" * 55)
    print("Step 3: Build/rebuild graphs with semantic edges")
    print("=" * 55)

    if not os.path.exists(SEMANTIC_PATH):
        print(f"❌ {SEMANTIC_PATH} not found. Run step2 first.")
        return

    semantic_df   = pd.read_parquet(SEMANTIC_PATH)
    if semantic_df.empty:
        print("⚠️  Semantic edges file is empty.")
        return

    dates_with_news = sorted(semantic_df["date"].unique().tolist())
    print(f"Semantic edge dates:  {len(dates_with_news)}")
    print(f"Date range:           {dates_with_news[0]} -> {dates_with_news[-1]}")

    # Check which of these dates already have graph PKL files
    existing_graphs = set(
        f.replace(".pkl", "")
        for f in os.listdir(GRAPHS_DIR)
        if f.endswith(".pkl")
    )
    to_build    = [d for d in dates_with_news if d not in existing_graphs]
    to_rebuild  = [d for d in dates_with_news if d in existing_graphs]

    print(f"\nGraphs to build (new):   {len(to_build)}")
    print(f"Graphs to rebuild (update): {len(to_rebuild)}")

    if not to_build and not to_rebuild:
        print("Nothing to do.")
        return

    # ── Load dependencies ─────────────────────────────────────────────────
    from src.data_pipeline.download_data import get_sp500_tickers, get_gics_mapping
    from src.data_pipeline.feature_engineering import load_features_panel
    from src.data_pipeline.graph_builder import build_daily_hetero_graph

    tickers = get_sp500_tickers()[:50]
    gics_df = get_gics_mapping()

    # ── Extend features to cover new dates ────────────────────────────────
    if to_build:
        latest_date = dates_with_news[-1]
        extend_features_to_date(tickers, latest_date)

    feature_panel = load_features_panel()
    print(f"Features loaded for {len(feature_panel)} tickers")

    # ── Build all dates (new + rebuild) ───────────────────────────────────
    all_dates = sorted(set(to_build + to_rebuild))
    built    = 0
    failed   = 0

    for date_str in tqdm(all_dates, desc="Building graphs"):
        out_path = os.path.join(GRAPHS_DIR, f"{date_str}.pkl")

        graph = build_daily_hetero_graph(
            date                 = date_str,
            tickers              = tickers,
            feature_panel        = feature_panel,
            gics_df              = gics_df,
            semantic_scores_path = SEMANTIC_PATH,
            corr_window          = 30,
            corr_threshold       = 0.6,
            seq_len              = 30,
        )

        if graph is not None:
            with open(out_path, "wb") as f:
                pickle.dump(graph, f)
            built += 1
        else:
            failed += 1

    print(f"\n{'='*55}")
    print(f"BUILD COMPLETE")
    print(f"{'='*55}")
    print(f"  Successfully built:  {built}")
    print(f"  Failed (no data):    {failed}")

    # ── Sanity check ──────────────────────────────────────────────────────
    if built > 0:
        last_date = dates_with_news[-1]
        check_path = os.path.join(GRAPHS_DIR, f"{last_date}.pkl")
        if os.path.exists(check_path):
            print(f"\nSanity check — {last_date}:")
            with open(check_path, "rb") as f:
                g = pickle.load(f)
            for store in g.edge_stores:
                if hasattr(store, "_key") and store._key:
                    et  = store._key[1]
                    cnt = store.edge_index.shape[1] if hasattr(store, "edge_index") else 0
                    icon = "✅" if cnt > 0 else "⚪"
                    print(f"  {icon} {et:15s}: {cnt} edges")

            n_stocks = g["stock"].x.shape[0]
            print(f"  Stocks in graph: {n_stocks}")

            if g.edge_stores:
                sem_cnt = next(
                    (store.edge_index.shape[1]
                     for store in g.edge_stores
                     if hasattr(store,"_key") and store._key
                     and store._key[1]=="semantic"
                     and hasattr(store,"edge_index")),
                    0
                )
                if sem_cnt > 0:
                    print(f"\n✅ Semantic edges confirmed working!")
                    print(f"   Open Streamlit Graph Explorer and select {last_date}")
                else:
                    print(f"\n⚠️  Semantic edges still 0.")
                    print(f"   Check that semantic_edges.parquet has entries for {last_date}:")
                    subset = semantic_df[semantic_df["date"] == last_date]
                    print(f"   Rows for {last_date}: {len(subset)}")
                    if len(subset):
                        print(f"   Tickers: {subset['ticker_a'].tolist()[:5]}")
                        # Check if those tickers are in the graph
                        print(f"   Graph tickers: {g['stock'].tickers[:10]}")

    print(f"\n  Next: python step4_retrain.py")


if __name__ == "__main__":
    main()