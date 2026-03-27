import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def compute_log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series,
                 fast: int = 12, slow: int = 26, signal: int = 9
                 ) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger_bands(close: pd.Series,
                             window: int = 20, num_std: float = 2.0
                             ) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def compute_rolling_volatility(log_returns: pd.Series, window: int = 30) -> pd.Series:
    return log_returns.rolling(window).std() * np.sqrt(252)  # Annualised


# ── Main feature builder ───────────────────────────────────────────────────────

def build_node_features(
    ohlcv_dir: str = "data/raw/ohlcv",
    output_dir: str = "data/processed/features",
    lookback: int = 30,
) -> None:

    os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(os.listdir(ohlcv_dir), desc="Feature Engineering"):
        if not fname.endswith(".csv"):
            continue
        ticker = fname.replace(".csv", "")
        out_path = os.path.join(output_dir, f"{ticker}.parquet")
        if os.path.exists(out_path):
            continue

        try:
            path = os.path.join(ohlcv_dir, fname)
            df = pd.read_csv(path)

            # Expected layout is the same as in load_all_close_prices:
            # header: Price,Close,High,Low,Open,Volume
            # row 0:  Ticker,ABBV,...
            # row 1:  Date,,,,,
            # row 2+: 2018-01-02, ...
            if "Price" not in df.columns:
                continue

            df = df.rename(columns={"Price": "Date"})
            df = df[~df["Date"].isin(["Ticker", "Date"])]
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")

            df.columns = [c.lower() for c in df.columns]

            # Ensure numeric dtypes for price/volume columns
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "close" not in df.columns or len(df) < lookback + 10:
                continue

            feat = pd.DataFrame(index=df.index)

            # Price & volume (z-score normalise volume)
            feat["open"]   = df["open"]
            feat["high"]   = df["high"]
            feat["low"]    = df["low"]
            feat["close"]  = df["close"]
            feat["volume"] = (df["volume"] - df["volume"].rolling(30).mean()) / \
                             (df["volume"].rolling(30).std() + 1e-9)

            # Returns & volatility
            feat["log_return"]    = compute_log_returns(df["close"])
            feat["volatility_30d"] = compute_rolling_volatility(feat["log_return"])

            # Momentum indicators
            feat["rsi_14"]       = compute_rsi(df["close"])
            macd, macd_sig       = compute_macd(df["close"])
            feat["macd"]         = macd
            feat["macd_signal"]  = macd_sig

            # Bollinger Bands (normalise relative to price)
            bb_u, bb_m, bb_l = compute_bollinger_bands(df["close"])
            feat["bb_upper"] = (bb_u - df["close"]) / (df["close"] + 1e-9)
            feat["bb_lower"] = (df["close"] - bb_l) / (df["close"] + 1e-9)

            # Target: NEXT DAY log return (shifted backward by 1)
            feat["target_return"] = feat["log_return"].shift(-1)

            feat.dropna(inplace=True)
            feat.to_parquet(out_path)

        except Exception as e:
            print(f"  Skipped {ticker}: {e}")

    print("Feature engineering complete.")


def load_features_panel(
    feature_dir: str = "data/processed/features",
) -> dict[str, pd.DataFrame]:
    """
    Load all parquet files into a dict: {ticker: feature_df}
    """
    panel = {}
    for fname in os.listdir(feature_dir):
        if fname.endswith(".parquet"):
            ticker = fname.replace(".parquet", "")
            panel[ticker] = pd.read_parquet(os.path.join(feature_dir, fname))
    return panel


if __name__ == "__main__":
    build_node_features()
    panel = load_features_panel()
    print(f"Loaded features for {len(panel)} tickers")
    example_ticker = list(panel.keys())[0]
    print(panel[example_ticker].tail())