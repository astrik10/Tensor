"""
scraper.py — PROVIDED, WORKING. Do not need to modify this file.

Polls the CoinGecko public API every 15 seconds for the current price of
Bitcoin (and Ethereum, for the optional stretch goal) and appends each
reading to a local CSV file (data/live_ticks.csv) as it comes in.

This is deliberately simple: one GET request, one JSON parse, no auth,
no rate-limit concerns at hackathon scale. The "scraping" is not meant
to be a differentiator between teams — your job starts once ticks land
in the CSV.

Usage:
    python scraper.py

Each row appended to data/live_ticks.csv has the columns:
    timestamp, coin, price_usd

Stop the scraper any time with Ctrl+C.
"""

import csv
import os
import time
from datetime import datetime, timezone

import requests

API_URL = "https://api.coingecko.com/api/v3/simple/price"
COINS = ["bitcoin", "ethereum"]
POLL_SECONDS = 15
OUTPUT_PATH = os.path.join("data", "live_ticks.csv")
MAX_RETRIES = 3


def fetch_prices():
    """Hit the CoinGecko API once and return {coin: price} or None on failure."""
    params = {"ids": ",".join(COINS), "vs_currencies": "usd"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return {coin: data[coin]["usd"] for coin in COINS if coin in data}
        except (requests.RequestException, KeyError, ValueError) as e:
            print(f"[scraper] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(2)
    print("[scraper] all retries failed for this poll — skipping this tick")
    return None


def ensure_output_file():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if not os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "coin", "price_usd"])


def append_rows(prices: dict):
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(OUTPUT_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        for coin, price in prices.items():
            writer.writerow([timestamp, coin, price])
    print(f"[scraper] {timestamp} -> {prices}")


def main():
    ensure_output_file()
    print(f"[scraper] polling {COINS} every {POLL_SECONDS}s. Writing to {OUTPUT_PATH}")
    print("[scraper] press Ctrl+C to stop.")
    try:
        while True:
            prices = fetch_prices()
            if prices:
                append_rows(prices)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\n[scraper] stopped.")


if __name__ == "__main__":
    main()
