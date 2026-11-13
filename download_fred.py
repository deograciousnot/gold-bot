"""
Download gold's main economic drivers from FRED (the St. Louis Federal Reserve's free database).

  DFII10    - 10-year US "real" interest rate (TIPS yield): interest minus expected inflation.
              Gold pays no interest, so when real rates rise, holding gold costs more
              in missed interest -> historically bad for gold.
  DTWEXBGS  - Broad US dollar index (since 2006). Gold is priced in dollars,
              so a stronger dollar usually means a lower gold price.
  DGS10     - 10-year US Treasury yield (nominal), for reference.

Uses only Python's standard library plus pandas. No account or API key needed.
"""

import io
import urllib.request
from pathlib import Path

import pandas as pd

SERIES = {"DFII10": "real_yield", "DTWEXBGS": "dollar", "DGS10": "yield10"}
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
OUT = Path(__file__).parent / "data" / "macro_daily.csv"

frames = []
for code, name in SERIES.items():
    with urllib.request.urlopen(URL.format(code), timeout=60) as r:
        s = pd.read_csv(io.BytesIO(r.read()), index_col=0, parse_dates=True, na_values=".")
    s.columns = [name]
    frames.append(s)
    print(f"{code:9} -> {name:10} {s.index[0]:%Y-%m-%d} to {s.index[-1]:%Y-%m-%d}, {s[name].notna().sum():,} values")

macro = pd.concat(frames, axis=1).sort_index()
macro.index.name = "date"
macro.to_csv(OUT)
print(f"Saved {OUT.name}")
