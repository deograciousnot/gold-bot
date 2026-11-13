# gold-bot: an honest search for a trading edge in gold

I wanted to know whether any trading strategy, from textbook indicators to a friend's "smart money"
method to 1-minute scalps, really makes money on gold after costs. So I built one honest simulator
and ran 33 strategies through it on 20 years of data. Then I built a bot that can only trade on a demo account.

**Short answer: almost nothing survives.** One strategy (the inside-day breakout) held up on years it had
never seen, but it didn't carry over to other markets, so I treat it as "unproven" and run it on demo only.

## The rules (fixed before looking at results)

- **Data:** Dukascopy 1-minute and hourly prices, 2005-2026, converted to broker server time.
- **Costs:** every trade pays a realistic spread and 4%/yr overnight financing.
- **No peeking:** signals use closed candles only; entry is the *next* candle's open. If a stop and
  a target are hit in the same candle, the stop is assumed to have been hit first.
- **Three periods:** develop on 2005-2018, judge on 2019-2023, and keep 2024-2026 **sealed** for one final exam.
- **Luck correction:** after testing dozens of ideas, some will look good by chance. A strategy only
  passes with a confidence score of **t ≥ 3.2** (3.3 for scalps) and profits in both periods.
- **Sanity check:** random entries through the same simulator average about −0.05R per trade (the
  cost of spreads), which is what an honest engine should show.

Results are in **R**, where 1R is the amount risked on one trade.

## What I found

| Test | Result |
|---|---|
| Tournament: 25 strategies and blends on H1/M15 (`tournament.py`) | 0 passed. Inside-day breakout came closest (t = 3.17) |
| Inside-day breakout on the sealed 2024-26 data | 92 trades, profit factor 1.74, +20.9%. Sells also profitable over 2005-23 |
| Friend's SMC strategy, mechanical version (`backtest_smc.py`) | −38.5% over 315 trades (t = −2.57) |
| London breakout (`backtest_breakout.py`) | Great 2-year result, −69% over 20 years |
| Trend following (`backtest_trend.py`) | About break-even (PF 1.04), −39% worst drawdown |
| News-triggered trades around NFP / FOMC (`fetch_events.py`) | No edge after costs |
| 7 classic 1-minute scalps (`scalp.py`) | All lose about 100%. The best (+0.10R) before costs became −0.46R after the spread |
| Inside-day breakout on 7 other markets (`multi_market.py`) | Didn't carry over: all markets together PF 0.96, t ≈ −1 |
| Real yields and dollar signals (`research_macro.py`) | No predictive edge |

The lesson: a simple price-chart rule that clearly worked would be traded away by people with better data
and lower costs. What's left looks like noise, and retail costs turn noise into a steady loss.

## The bot (`run_bot.py`)

Runs on Windows through the MetaTrader 5 Python API.

- **Demo-only lock**, hard-coded in `bot/safety.py`. It refuses to run on a real or contest account.
- 1% risk per trade, a lot-size cap, a 3% daily loss limit and one position at a time.
- Skips wide spreads and the rollover hours; closes before the weekend and after 24 hours.
- **Kill switch:** create a file called `STOP` and it closes its trades and exits.
- Every decision goes to `logs/bot.log` and every order to `logs/journal.csv`.
- 23 unit tests: `python -m unittest discover tests`.

Settings live in `bot/config.py`; strategies live in `bot/strategies.py`.

## Running the research

```bash
pip install pandas numpy
python download_dukascopy.py          # gold 1-minute data (slow: Dukascopy throttles)
python download_dukascopy.py hourly
python tournament.py                  # the 25-strategy tournament
python scalp.py                       # the scalping tournament
python multi_market.py                # inside-day breakout on 7 other markets
```

Price data isn't included in this repo; the scripts download it.

*This is a learning project, not financial advice. Most retail traders lose money.*
