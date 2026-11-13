"""
Download long-term gold (XAUUSD) price history from Dukascopy's free public data feed,
then turn it into 15-minute and 1-hour candles on HFM's server clock.

Why: MT5 only gave us 2-4 years of intraday data. Dukascopy has ~20 years,
which lets us test strategies through many different market conditions.

How the feed works: one small compressed file per day, holding that day's 1-minute
candles (bid prices, in UTC). Downloaded days are cached in data/dukascopy/raw/,
so if the script is stopped, running it again carries on where it left off.

Uses only Python's standard library plus pandas. READ-ONLY: never touches MT5 or trades.

Usage:  python download_dukascopy.py                 (2005 to today, 1-minute data -> M15; slow)
        python download_dukascopy.py 2024 2024       (just one year)
        python download_dukascopy.py build 2016 2023 (no downloading: build M15 from the days cached so far)
        python download_dukascopy.py hourly          (2005 to today, 1-hour data only; fast)

Days are fetched NEWEST FIRST, so recent years become usable long before the old ones arrive.
"""

import lzma
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

SYMBOL = "XAUUSD"
PRICE_SCALE = 1000          # Dukascopy stores gold prices as integers x1000
URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "dukascopy" / "raw"
RECORD = struct.Struct(">5if")  # seconds-from-midnight, open, close, low, high, volume

# Dukascopy gives bid prices only. HFM gold spreads are usually $0.2-0.4,
# so we assume a slightly pessimistic spread that scales with price.
def assumed_spread(price):
    return max(0.30, price * 0.0001)  # $0.30, or 0.01% of the price if bigger


def fetch_day(day):
    """Download one day's 1-minute candles (cached). Returns raw bytes, or b'' if none."""
    cache = RAW_DIR / f"{day:%Y%m%d}.bi5"
    if cache.exists():
        return cache.read_bytes()
    # Note: Dukascopy numbers months from 0 (January = 00)
    url = URL.format(sym=SYMBOL, y=day.year, m=day.month - 1, d=day.day)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                data = b""  # no file: market closed that day
                break
            time.sleep(2 * (attempt + 1))  # likely rate-limited: back off and retry
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 * (attempt + 1))
    else:
        return None  # failed 3 times; don't cache, retry next run
    # Write to a temporary name first, so an interrupted run never leaves a half-written file
    tmp = cache.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(cache)
    return data


def cached_day(day):
    """Build mode: only read what's already downloaded. None if the day isn't cached yet."""
    cache = RAW_DIR / f"{day:%Y%m%d}.bi5"
    return cache.read_bytes() if cache.exists() else None


HOUR_URL = "https://datafeed.dukascopy.com/datafeed/{sym}/{y}/{m:02d}/BID_candles_hour_1.bi5"


def fetch_month(month):
    """Download one month of 1-hour candles (cached). Much faster than minute data: 1 file per month."""
    cache = RAW_DIR / f"H{month:%Y%m}.bi5"
    if cache.exists():
        return cache.read_bytes()
    url = HOUR_URL.format(sym=SYMBOL, y=month.year, m=month.month - 1)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                data = b""
                break
            time.sleep(3 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(3 * (attempt + 1))
    else:
        return None
    if month.year == date.today().year and month.month == date.today().month:
        return data  # current month is still growing: don't cache it
    cache.write_bytes(data)
    return data


def parse_day(day, data):
    """Turn one day's compressed file into rows of (time, open, high, low, close)."""
    if not data:
        return []
    try:
        raw = lzma.decompress(data)
    except lzma.LZMAError:
        # Damaged file (e.g. from an interrupted older run): delete it so the next run re-downloads it
        (RAW_DIR / f"{day:%Y%m%d}.bi5").unlink(missing_ok=True)
        return None
    midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    rows = []
    for secs, o, c, lo, hi, vol in RECORD.iter_unpack(raw):
        if vol == 0:
            continue  # empty minutes (market closed) are filled with flat candles
        rows.append((midnight + timedelta(seconds=secs),
                     o / PRICE_SCALE, hi / PRICE_SCALE, lo / PRICE_SCALE, c / PRICE_SCALE))
    return rows


def to_server_time(utc_times):
    """HFM's server clock is New York time + 7 hours (GMT+2 in winter, GMT+3 in summer)."""
    return utc_times.dt.tz_convert("America/New_York").dt.tz_localize(None) + pd.Timedelta(hours=7)


def resample(minutes, rule):
    candles = minutes.resample(rule, on="time", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    candles = candles.dropna().reset_index()
    # Spread in MT5's units (points of $0.01), so the backtests can read it the same way
    candles["spread"] = (candles.close.map(assumed_spread) / 0.01).round().astype(int)
    return candles


def hourly_only(first_year):
    """Quick mode: 1-hour candles only, one file per month."""
    months, m = [], date(first_year, 1, 1)
    while m <= date.today():
        months.append(m)
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    print(f"Fetching {len(months)} months of hourly candles...")
    rows, failed = [], 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        for month, data in zip(months, pool.map(fetch_month, months)):
            if data is None:
                failed += 1
                continue
            # Same record layout as the minute files, but time counts from the start of the month
            rows.extend(parse_day(month, data))
    if failed:
        print(f"{failed} months failed - run again to retry them")
    hours = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    hours["time"] = to_server_time(hours.time)
    candles = resample(hours, "1h")
    path = DATA_DIR / "XAUUSD_H1_long.csv"
    candles.to_csv(path, index=False)
    print(f"H1: {len(candles):,} candles, {candles.time.iloc[0]:%Y-%m-%d} to {candles.time.iloc[-1]:%Y-%m-%d} -> {path.name}")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "hourly":
        hourly_only(int(sys.argv[2]) if len(sys.argv) > 2 else 2005)
        sys.exit()

    build_only = len(sys.argv) > 1 and sys.argv[1] == "build"
    years = [int(a) for a in sys.argv[1:] if a.isdigit()]
    first_year = years[0] if years else 2005
    last_year = years[1] if len(years) > 1 else date.today().year

    start, end = date(first_year, 1, 1), min(date(last_year, 12, 31), date.today() - timedelta(days=1))
    days = [start + timedelta(n) for n in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() != 5]  # gold doesn't trade on Saturdays
    days.reverse()  # newest first

    print(f"{'Building from cache' if build_only else 'Fetching'} {len(days):,} days ({start} to {end})...")
    rows, missing = [], 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        get = cached_day if build_only else fetch_day
        for n, (day, data) in enumerate(zip(days, pool.map(get, days)), 1):
            parsed = None if data is None else parse_day(day, data)
            if parsed is None:
                missing += 1
            else:
                rows.extend(parsed)
            if n % 500 == 0:
                print(f"  {n:,}/{len(days):,} days done", flush=True)
    if missing:
        print(f"{missing} days not available yet" + ("" if build_only else " (failed) - run the script again to retry them"))

    minutes = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close"])
    minutes["time"] = to_server_time(minutes.time)
    minutes = minutes.sort_values("time").reset_index(drop=True)
    print(f"{len(minutes):,} one-minute candles, {minutes.time.iloc[0]:%Y-%m-%d} to {minutes.time.iloc[-1]:%Y-%m-%d}")

    # Only M15 is written here: the 20-year H1 file comes from the faster "hourly" mode
    candles = resample(minutes, "15min")
    path = DATA_DIR / "XAUUSD_M15_long.csv"
    candles.to_csv(path, index=False)
    print(f"M15: {len(candles):,} candles -> {path.name}")
