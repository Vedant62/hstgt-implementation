"""
Run FinBERT on every headline to get spillover scores.
Place in project root. Run: python step2_run_finbert.py

On M1 MPS this takes ~5-15 minutes depending on article count.
Model is downloaded once (~500MB) and cached in ~/.cache/huggingface/
"""

import os, sys
import pandas as pd
import torch
from tqdm import tqdm
from itertools import combinations
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RAW_PATH      = "data/raw/news/all_news_raw.parquet"
SEMANTIC_PATH = "data/processed/semantic_edges.parquet"
BATCH_SIZE    = 64    # Increase if you have more RAM; decrease if OOM
MIN_SCORE     = 0.3   # Only keep edges where spillover >= this threshold

os.makedirs("data/processed", exist_ok=True)


def load_finbert(device: str):
    print(f"Loading FinBERT on {device}...")
    print("(First run downloads ~500MB model — subsequent runs are instant)")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model     = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval().to(device)
    print("✅ FinBERT loaded")
    return tokenizer, model


@torch.no_grad()
def score_headlines_batched(
    headlines: list[str],
    tokenizer,
    model,
    device:     str,
    batch_size: int = 64,
) -> list[float]:
    """
    Returns a spillover score in [0,1] for each headline.
    spillover = P(positive) + P(negative)
    Neutral headlines score near 0 — they carry no market signal.
    Strong sentiment (either direction) scores near 1.
    """
    all_scores = []

    for i in tqdm(range(0, len(headlines), batch_size), desc="FinBERT scoring"):
        batch = headlines[i : i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors    = "pt",
            truncation        = True,
            max_length        = 128,
            padding           = True,
        ).to(device)

        logits = model(**inputs).logits          # (B, 3)
        probs  = torch.softmax(logits, dim=-1)   # (B, 3)
        # Columns: [positive, negative, neutral]
        spillover = (probs[:, 0] + probs[:, 1]).cpu().tolist()
        all_scores.extend(spillover)

    return all_scores


def build_semantic_edges(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each date, pairs all tickers that both had news that day.
    Uses the MAX spillover score across all articles for each ticker on that day
    as the edge weight — captures the most significant news event.
    """
    print("\nBuilding semantic edges from scored headlines...")
    records = []

    for date, group in tqdm(news_df.groupby("date"), desc="Building edges"):
        tickers_today = group["ticker"].unique().tolist()
        if len(tickers_today) < 2:
            continue

        # Best (max) spillover score per ticker for this day
        score_map = (
            group.groupby("ticker")["spillover_score"].max().to_dict()
        )
        # Representative headline for the edge (highest scoring article)
        top_headline = (
            group.sort_values("spillover_score", ascending=False)
                 .iloc[0]["headline"]
        )

        for ta, tb in combinations(tickers_today, 2):
            score_a = score_map.get(ta, 0.0)
            score_b = score_map.get(tb, 0.0)
            # Edge weight = geometric mean of the two scores
            # This requires BOTH stocks to have signal, not just one
            spillover = (score_a * score_b) ** 0.5

            if spillover < MIN_SCORE:
                continue

            records.append({
                "date":            date,
                "ticker_a":        ta,
                "ticker_b":        tb,
                "spillover_score": round(float(spillover), 4),
                "headline":        top_headline,
            })

    return pd.DataFrame(records)


def main():
    print("=" * 55)
    print("Step 2: FinBERT Semantic Edge Scoring")
    print("=" * 55)

    # ── Check input exists ────────────────────────────────────────────────
    if not os.path.exists(RAW_PATH):
        print(f"❌ {RAW_PATH} not found.")
        print("   Run step1_download_news.py first.")
        return

    news_df = pd.read_parquet(RAW_PATH)
    print(f"Loaded {len(news_df):,} headlines across {news_df['ticker'].nunique()} tickers")
    print(f"Date range: {news_df['date'].min()} → {news_df['date'].max()}")

    # ── Device ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── Load FinBERT ──────────────────────────────────────────────────────
    tokenizer, model = load_finbert(device)

    # ── Score all headlines ───────────────────────────────────────────────
    headlines = news_df["headline"].fillna("").tolist()
    print(f"\nScoring {len(headlines):,} headlines...")

    scores = score_headlines_batched(headlines, tokenizer, model, device, BATCH_SIZE)
    news_df["spillover_score"] = scores

    # Show score distribution
    print(f"\nSpillover score distribution:")
    print(f"  Mean:   {news_df['spillover_score'].mean():.3f}")
    print(f"  Median: {news_df['spillover_score'].median():.3f}")
    strong = (news_df["spillover_score"] >= MIN_SCORE).sum()
    print(f"  ≥{MIN_SCORE} (strong signal): {strong:,} / {len(news_df):,} articles")

    # Save scored headlines
    scored_path = RAW_PATH.replace(".parquet", "_scored.parquet")
    news_df.to_parquet(scored_path, index=False)
    print(f"\n✅ Scored headlines → {scored_path}")

    # ── Build semantic edges ──────────────────────────────────────────────
    semantic_df = build_semantic_edges(news_df)

    if semantic_df.empty:
        print("\n⚠️  No semantic edges generated.")
        print("   This means no two tickers shared a news day with score ≥ 0.3")
        print(f"   Try lowering MIN_SCORE (currently {MIN_SCORE})")
        # Save empty file so downstream scripts don't crash
        semantic_df = pd.DataFrame(columns=["date","ticker_a","ticker_b","spillover_score","headline"])

    semantic_df.to_parquet(SEMANTIC_PATH, index=False)

    print(f"\n{'='*55}")
    print(f"SEMANTIC EDGES SUMMARY")
    print(f"{'='*55}")
    print(f"  Total edges:          {len(semantic_df):,}")
    print(f"  Days with edges:      {semantic_df['date'].nunique() if len(semantic_df) else 0}")
    if len(semantic_df) > 0:
        print(f"  Avg spillover score:  {semantic_df['spillover_score'].mean():.3f}")
        print(f"  Top pairs by score:")
        top = semantic_df.nlargest(5, "spillover_score")[["date","ticker_a","ticker_b","spillover_score"]]
        print(top.to_string(index=False))
    print(f"\n  Saved → {SEMANTIC_PATH}")
    print(f"\n  Next: python step3_rebuild_graphs.py")


if __name__ == "__main__":
    main()