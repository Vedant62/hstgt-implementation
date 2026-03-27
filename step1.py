"""
Download news via Google News RSS — free, no auth, no rate limits.
Google News RSS search URL:
  https://news.google.com/rss/search?q=AAPL+stock&hl=en-US&gl=US&ceid=US:en
"""

import os, sys, time, json
import xml.etree.ElementTree as ET
import requests
import pandas as pd
from datetime import datetime
from email.utils import parsedate_to_datetime
from tqdm import tqdm

CACHE_DIR  = "data/raw/news"
os.makedirs(CACHE_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from src.data_pipeline.download_data import get_sp500_tickers
    TICKERS = get_sp500_tickers()[:50]
except Exception:
    TICKERS = [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","JNJ","V",
        "PG","UNH","MA","HD","BAC","ADBE","NFLX","CMCSA","XOM","VZ",
        "INTC","KO","PFE","MRK","T","PEP","ABT","CRM","CSCO","WMT",
        "ACN","CVX","MCD","DHR","NEE","NKE","AMD","LIN","COST","TXN",
        "HON","AMGN","PM","QCOM","UNP","LOW","BMY","ORCL","IBM","GS",
    ]

# Company name overrides for better Google News search results
# For tickers where the symbol alone gives poor results
TICKER_TO_NAME = {
    "GOOGL": "Alphabet Google",
    "META":  "Meta Platforms Facebook",
    "BRK-B": "Berkshire Hathaway",
    "T":     "AT&T stock",
    "V":     "Visa stock",
    "MA":    "Mastercard stock",
    "A":     "Agilent Technologies",
    "LOW":   "Lowes stock",
    "LIN":   "Linde stock",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


def fetch_google_news(ticker: str, force: bool = False) -> list[dict]:
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_gnews.json")
    if os.path.exists(cache_path) and not force:
        with open(cache_path) as f:
            cached = json.load(f)
        if cached:
            return cached

    # Use company name if available, otherwise ticker + "stock"
    query = TICKER_TO_NAME.get(ticker, f"{ticker} stock")
    query_enc = requests.utils.quote(query)

    url = (
        f"https://news.google.com/rss/search"
        f"?q={query_enc}"
        f"&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return []
    except Exception:
        return []

    records = []
    try:
        root  = ET.fromstring(resp.content)
        items = root.findall(".//item")

        for item in items:
            title   = item.findtext("title",   default="").strip()
            pubdate = item.findtext("pubDate", default="")
            source  = item.findtext("source",  default="")
            link    = item.findtext("link",    default="")

            if not title:
                continue

            # Remove source suffix Google appends: "Title - Source Name"
            if " - " in title:
                title = title.rsplit(" - ", 1)[0].strip()

            try:
                dt       = parsedate_to_datetime(pubdate)
                date_str = dt.strftime("%Y-%m-%d")
            except Exception:
                date_str = datetime.now().strftime("%Y-%m-%d")

            records.append({
                "ticker":   ticker,
                "date":     date_str,
                "headline": title,
                "source":   source,
                "url":      link,
            })

    except ET.ParseError:
        pass

    with open(cache_path, "w") as f:
        json.dump(records, f)

    return records


def main():
    print("=" * 55)
    print("Step 1: Google News RSS Downloader")
    print("=" * 55)

    # Quick sanity check
    print("Checking Google News RSS for AAPL...")
    test = fetch_google_news("AAPL", force=True)

    if not test:
        print("❌ Google News returned nothing.")
        print("\nDebug: testing raw request...")
        url = "https://news.google.com/rss/search?q=AAPL+stock&hl=en-US&gl=US&ceid=US:en"
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            print(f"  HTTP status:  {r.status_code}")
            print(f"  Content-Type: {r.headers.get('Content-Type','?')}")
            print(f"  Body preview: {r.text[:300]}")
        except Exception as e:
            print(f"  Request failed: {e}")
        return

    print(f"✅ Google News working — {len(test)} articles for AAPL")
    print(f"   Sample: [{test[0]['date']}] {test[0]['headline'][:65]}...")
    print()

    all_records = []
    empty_tickers = []

    for ticker in tqdm(TICKERS, desc="Fetching Google News"):
        records = fetch_google_news(ticker)
        if records:
            all_records.extend(records)
        else:
            empty_tickers.append(ticker)
        time.sleep(0.3)   # gentle, Google is fine with this pace

    if not all_records:
        print("❌ No articles found for any ticker.")
        return

    news_df = pd.DataFrame(all_records)
    news_df.drop_duplicates(subset=["ticker", "headline"], inplace=True)
    news_df = news_df[news_df["headline"].str.len() > 10]
    news_df.sort_values(["date", "ticker"], inplace=True)
    news_df.reset_index(drop=True, inplace=True)

    raw_path = os.path.join(CACHE_DIR, "all_news_raw.parquet")
    news_df.to_parquet(raw_path, index=False)

    print(f"\n{'='*55}")
    print(f"✅ Done.")
    print(f"   Articles:           {len(news_df):,}")
    print(f"   Tickers with news:  {news_df['ticker'].nunique()} / {len(TICKERS)}")
    if empty_tickers:
        print(f"   Empty tickers:      {empty_tickers}")
    print(f"   Date range:         {news_df['date'].min()} -> {news_df['date'].max()}")
    print(f"   Saved -> {raw_path}")
    print(f"\n   Sample headlines:")
    for _, row in news_df.sample(min(5, len(news_df))).iterrows():
        print(f"     [{row['date']}] {row['ticker']:6s}: {row['headline'][:60]}...")
    print(f"\n   Next: python step2_run_finbert.py")


if __name__ == "__main__":
    main()