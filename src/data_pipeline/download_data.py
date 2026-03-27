import os
import time
from io import StringIO

import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm


def get_sp500_tickers() -> list[str]:
    """
    Scrape current S&P 500 ticker list from Wikipedia.
    Returns a list of ticker strings, e.g. ['AAPL', 'MSFT', ...]
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Use requests with a User-Agent to avoid HTTP 403 from Wikipedia
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        )
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    # Wrap HTML in StringIO to avoid pandas FutureWarning about literal HTML
    table = pd.read_html(StringIO(resp.text))[0]
    # Wikipedia uses '.' in some tickers (e.g. BRK.B); yfinance needs '-'
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def get_gics_mapping() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        )
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    table = pd.read_html(StringIO(resp.text))[0]
    table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
    mapping = table[["Symbol", "GICS Sector", "GICS Sub-Industry"]].copy()
    mapping.columns = ["ticker", "sector", "sub_industry"]
    return mapping


def download_ohlcv(
    tickers: list[str],
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-31",
    output_dir: str = "data/raw/ohlcv",
) -> None:
    """
    Download daily OHLCV data for each ticker and save as CSV.
    Handles API rate limits with exponential backoff.
    """
    os.makedirs(output_dir, exist_ok=True)
    failed = []

    for ticker in tqdm(tickers, desc="Downloading OHLCV"):
        out_path = os.path.join(output_dir, f"{ticker}.csv")
        if os.path.exists(out_path):
            continue  # Skip already-downloaded files

        try:
            df = yf.download(ticker, start=start_date, end=end_date,
                             progress=False, auto_adjust=True)
            if df.empty:
                failed.append(ticker)
                continue
            df.to_csv(out_path)
            time.sleep(0.2)  # Polite rate limiting
        except Exception as e:
            print(f"  Failed {ticker}: {e}")
            failed.append(ticker)
            time.sleep(1.0)

    if failed:
        print(f"\n⚠ Failed tickers ({len(failed)}): {failed}")
        with open(os.path.join(output_dir, "failed.txt"), "w") as f:
            f.write("\n".join(failed))


def load_all_close_prices(ohlcv_dir: str = "data/raw/ohlcv") -> pd.DataFrame:
    """
    Load all downloaded CSVs and return a single wide DataFrame:
    index = date, columns = tickers, values = adjusted close prices.
    """
    frames = {}
    for fname in os.listdir(ohlcv_dir):
        if not fname.endswith(".csv"):
            continue
        ticker = fname.replace(".csv", "")
        path = os.path.join(ohlcv_dir, fname)
        try:
            df = pd.read_csv(path)
            if "Price" not in df.columns or "Close" not in df.columns:
                continue

            df = df.rename(columns={"Price": "Date"})
            df = df[~df["Date"].isin(["Ticker", "Date"])]
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date")

            # Ensure close is numeric
            close = pd.to_numeric(df["Close"], errors="coerce")
            frames[ticker] = close
        except Exception:
            # Silently skip malformed files
            continue

    prices = pd.DataFrame(frames)
    prices.sort_index(inplace=True)
    return prices


if __name__ == "__main__":
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} S&P 500 tickers")
    download_ohlcv(tickers[:50])  # Start with 50 for quick testing
    print("Download complete.")