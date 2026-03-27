import os
import pandas as pd
import numpy as np

# ── Pipeline imports ───────────────────────────────────────────────────────────
from src.data_pipeline.download_data    import (
    get_sp500_tickers, get_gics_mapping, download_ohlcv, load_all_close_prices
)
from src.data_pipeline.feature_engineering import build_node_features, load_features_panel
from src.data_pipeline.graph_builder       import build_graph_dataset
from src.training.walk_forward             import walk_forward_train

# ── Config ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "tickers_limit":  50,       # Use 50 tickers for quick testing; set to 500 for full run
    "start_date":    "2018-01-01",
    "end_date":      "2024-12-31",
    "seq_len":        30,
    "corr_window":    30,
    "corr_threshold": 0.6,
    "model_config": {
        "input_dim":      8,
        "d_model":        64,
        "seq_len":        30,
        "nhead_temporal": 4,
        "nhead_spatial":  4,
        "num_blocks":     2,
        "dropout":        0.1,
    },
    "train_days": 500,   # Reduced for quick testing
    "val_days":    60,
    "test_days":   30,
    "stride":      30,
    "epochs":      15,
    "lr":          1e-4,
}

def main():
    os.makedirs("data/raw/ohlcv",         exist_ok=True)
    os.makedirs("data/processed/features", exist_ok=True)
    os.makedirs("data/graphs",             exist_ok=True)
    os.makedirs("results",                 exist_ok=True)

    # ── 1. Download data ───────────────────────────────────────────────────────
    print("\n[1/5] Downloading OHLCV data...")
    tickers  = get_sp500_tickers()[: CONFIG["tickers_limit"]]
    gics_df  = get_gics_mapping()
    download_ohlcv(tickers, CONFIG["start_date"], CONFIG["end_date"])

    # ── 2. Feature engineering ─────────────────────────────────────────────────
    print("\n[2/5] Engineering features...")
    build_node_features(lookback=CONFIG["seq_len"])
    feature_panel = load_features_panel()
    print(f"  Loaded features for {len(feature_panel)} tickers")

    # ── 3. Build daily graphs ──────────────────────────────────────────────────
    print("\n[3/5] Building heterogeneous daily graphs...")
    # Get common trading dates from the price data
    prices = load_all_close_prices()
    all_dates = [str(d.date()) for d in prices.index]

    graph_paths = build_graph_dataset(
        trading_dates     = all_dates,
        tickers           = tickers,
        feature_panel     = feature_panel,
        gics_df           = gics_df,
        corr_window       = CONFIG["corr_window"],
        corr_threshold    = CONFIG["corr_threshold"],
        seq_len           = CONFIG["seq_len"],
    )
    print(f"  Total usable daily graphs: {len(graph_paths)}")

    # ── 4. Walk-forward training ───────────────────────────────────────────────
    print("\n[4/5] Starting walk-forward training...")
    results = walk_forward_train(
        graph_paths  = graph_paths,
        model_config = CONFIG["model_config"],
        train_days   = CONFIG["train_days"],
        val_days     = CONFIG["val_days"],
        test_days    = CONFIG["test_days"],
        stride       = CONFIG["stride"],
        epochs       = CONFIG["epochs"],
        lr           = CONFIG["lr"],
    )

    # ── 5. Summary ─────────────────────────────────────────────────────────────
    print("\n[5/5] Final Results Summary:")
    print(results)
    print(f"\n  Mean Test Directional Accuracy : {results['da'].mean():.2%}")
    print(f"  Mean Test Sharpe Ratio         : {results['sharpe'].mean():.2f}")
    print(f"  Mean Test RMSE                 : {results['rmse'].mean():.4f}")


if __name__ == "__main__":
    main()