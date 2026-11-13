"""
Step 2a: download gold's price history from MT5 and save it to a CSV file.

READ-ONLY: this only reads prices, it never trades.
Each row is one candle: open, high, low, close, and the spread at that time.
"""

import sys
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

SYMBOL = "XAUUSD"
TIMEFRAMES = {"H1": mt5.TIMEFRAME_H1, "M15": mt5.TIMEFRAME_M15, "D1": mt5.TIMEFRAME_D1}
MAX_BARS = 100_000  # MT5 returns as many as it has, up to this
DATA_DIR = Path(__file__).parent / "data"

if not mt5.initialize():
    sys.exit(f"Could not connect to MT5: {mt5.last_error()}")

mt5.symbol_select(SYMBOL, True)
DATA_DIR.mkdir(exist_ok=True)

for name, timeframe in TIMEFRAMES.items():
    # MT5 rejects requests above its "Max bars in chart" setting, so step down until it accepts
    rates, count = None, MAX_BARS
    while count >= 1_000:
        rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, count)
        if rates is not None and len(rates) > 0:
            break
        count //= 2
    if rates is None or len(rates) == 0:
        print(f"{name}: no data ({mt5.last_error()})")
        continue

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")  # MT5 gives seconds since 1970
    df = df[["time", "open", "high", "low", "close", "tick_volume", "spread"]]

    path = DATA_DIR / f"{SYMBOL}_{name}.csv"
    df.to_csv(path, index=False)
    print(f"{name}: {len(df):,} candles from {df.time.iloc[0]:%Y-%m-%d} to {df.time.iloc[-1]:%Y-%m-%d} -> {path.name}")

mt5.shutdown()
