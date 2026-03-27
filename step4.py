"""
Retrain the last N folds whose test windows overlap with news dates.
Saves to results/semantic/ so original results are untouched for comparison.
"""

import os, sys, pickle
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEMANTIC_PATH = "data/processed/semantic_edges.parquet"
GRAPHS_DIR    = "data/graphs"
OUTPUT_DIR    = "results/semantic"


def main():
    print("=" * 55)
    print("Step 4: Retraining with semantic edges")
    print("=" * 55)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Find which graphs have semantic edges ─────────────────────────────
    if not os.path.exists(SEMANTIC_PATH):
        print("❌ Run step3 first.")
        return

    semantic_df   = pd.read_parquet(SEMANTIC_PATH)
    dates_w_news  = set(semantic_df["date"].unique())

    # ── Get all graph paths sorted chronologically ────────────────────────
    all_paths = sorted([
        os.path.join(GRAPHS_DIR, f)
        for f in os.listdir(GRAPHS_DIR)
        if f.endswith(".pkl")
    ])

    print(f"Total graphs available: {len(all_paths)}")

    # Find the first graph that has semantic edges — train from 500 days before that
    dates_in_graphs = [
        os.path.basename(p).replace(".pkl", "") for p in all_paths
    ]
    first_semantic_idx = next(
        (i for i, d in enumerate(dates_in_graphs) if d in dates_w_news),
        len(all_paths) - 200   # fallback: use last 200 graphs
    )

    # Start 500 training days before first semantic date
    start_idx = max(0, first_semantic_idx - 500)
    paths_for_training = all_paths[start_idx:]

    print(f"Training window starts: {dates_in_graphs[start_idx]}")
    print(f"First semantic date:    {dates_in_graphs[first_semantic_idx]}")
    print(f"Graphs for training:    {len(paths_for_training)}")

    if len(paths_for_training) < 600:
        print("⚠️  Less than 600 graphs — may only get 1-2 folds.")
        print("   This is expected with yfinance (recent news only).")

    # ── Train ─────────────────────────────────────────────────────────────
    from src.training.walk_forward import walk_forward_train

    model_config = {
        "input_dim":      8,
        "d_model":        64,
        "seq_len":        30,
        "nhead_temporal": 4,
        "nhead_spatial":  4,
        "num_blocks":     2,
        "dropout":        0.1,
    }

    print(f"\nStarting walk-forward training...")
    print(f"(Uses MPS automatically on M1)\n")

    results = walk_forward_train(
        graph_paths  = paths_for_training,
        model_config = model_config,
        train_days   = 500,
        val_days     =  60,
        test_days    =  30,
        stride       =  30,
        epochs       =  20,
        lr           = 1e-4,
        output_dir   = OUTPUT_DIR,
        loss_kwargs  = {"lambda_mse": 1.0, "lambda_dir": 1.0},
    )

    # ── Compare against original results ──────────────────────────────────
    original_path = "results/walk_forward_results.csv"
    print(f"\n{'='*55}")
    print("SEMANTIC vs BASELINE COMPARISON")
    print(f"{'='*55}")

    print(f"\nSemantic model (these folds):")
    print(results[["fold","da","sharpe","rmse"]].to_string(index=False))
    print(f"\n  Mean DA:     {results['da'].mean():.2%}")
    print(f"  Mean Sharpe: {results['sharpe'].mean():.2f}")
    print(f"  Mean RMSE:   {results['rmse'].mean():.4f}")

    if os.path.exists(original_path):
        orig = pd.read_csv(original_path)
        print(f"\nBaseline model (same fold range from original run):")
        # Show the equivalent folds from baseline for fair comparison
        overlap_folds = results["fold"].tolist()
        orig_overlap  = orig[orig["fold"].isin(overlap_folds)]
        if not orig_overlap.empty:
            print(orig_overlap[["fold","da","sharpe","rmse"]].to_string(index=False))
            print(f"\n  Baseline Mean DA:     {orig_overlap['da'].mean():.2%}")
            print(f"  Baseline Mean Sharpe: {orig_overlap['sharpe'].mean():.2f}")
            print(f"\n  DA improvement:    {(results['da'].mean() - orig_overlap['da'].mean())*100:+.2f}pp")
            print(f"  Sharpe improvement:{results['sharpe'].mean() - orig_overlap['sharpe'].mean():+.2f}")

    print(f"\n✅ Semantic results saved → {OUTPUT_DIR}/")
    print(f"\n  Now open Streamlit and check the Model Performance tab.")
    print(f"  For Live Predictions, use checkpoints from: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()