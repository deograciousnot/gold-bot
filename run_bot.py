"""
The demo trading bot.

  python run_bot.py          run until stopped (Ctrl+C, or create a file called STOP in this folder)
  python run_bot.py --once   do a single check and exit (handy for testing)

What it does every POLL_SECONDS:
  1. Confirms MT5 is still on a DEMO account (refuses to continue otherwise).
  2. Checks the kill switch, the daily loss limit and the weekend cutoff -> closes bot trades if needed.
  3. When a new candle has closed, asks the strategy for a signal.
  4. Runs every safety rule, sizes the trade for 1% risk, and sends it (or just logs it in DRY_RUN).

Everything it decides is written to logs/bot.log, and every order to logs/journal.csv.
"""

import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bot import broker, config, safety
from bot.risk import compute_lots
from bot.strategies import STRATEGIES

BASE = Path(__file__).parent
LOGS = BASE / "logs"
STATE_FILE = LOGS / "state.json"
JOURNAL = LOGS / "journal.csv"

LOGS.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    handlers=[logging.FileHandler(LOGS / "bot.log", encoding="utf-8"),
                              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("bot")


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def journal(**row):
    new = not JOURNAL.exists()
    with JOURNAL.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "action", "side", "lots", "price", "stop", "target",
                                          "dry_run", "reason"])
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in w.fieldnames})


def close_all(reason):
    for p in broker.bot_positions():
        if config.DRY_RUN:
            log.info(f"[DRY RUN] would close position {p.ticket} ({reason})")
        else:
            broker.close_position(p)
            log.info(f"Closed position {p.ticket} ({reason})")
        journal(time=datetime.now(), action="close", lots=p.volume, price=p.price_current,
                dry_run=config.DRY_RUN, reason=reason)


def check_once(state, strategy):
    acct = broker.account()
    safety.require_demo(acct)  # re-checked every loop, in case MT5 is switched to another account

    now = broker.server_time()
    today = now.strftime("%Y-%m-%d")
    if state.get("day") != today:  # first check of a new day: remember the starting equity
        state.update(day=today, day_start_equity=acct.equity)
        save_state(state)
        log.info(f"New trading day {today}: starting equity {acct.equity:,.2f} {acct.currency}")

    close_now, why = safety.must_close_all(server_time=now, equity=acct.equity,
                                           day_start_equity=state["day_start_equity"], base_dir=BASE)
    if close_now:
        close_all(why)
        if safety.kill_switch_on(BASE):
            log.info("Kill switch: bot stopped.")
            return False
        return True

    # Time limit: close any bot trade that has been open longer than MAX_HOLD_HOURS
    for p in broker.bot_positions():
        opened = datetime.fromtimestamp(p.time, tz=timezone.utc).replace(tzinfo=None)   # server time, like ticks
        if safety.too_old(opened, now):
            reason = f"open for {config.MAX_HOLD_HOURS}h (time limit)"
            if config.DRY_RUN:
                log.info(f"[DRY RUN] would close position {p.ticket} ({reason})")
            else:
                broker.close_position(p)
                log.info(f"Closed position {p.ticket} ({reason})")
            journal(time=now, action="close", lots=p.volume, price=p.price_current, dry_run=config.DRY_RUN, reason=reason)

    candles = broker.candles()
    last_closed = str(candles.time.iloc[-1])
    if state.get("last_candle") == last_closed:
        return True  # nothing new yet
    state["last_candle"] = last_closed
    save_state(state)

    signal = strategy(candles)
    if signal is None:
        log.info(f"Candle {last_closed} closed: no signal")
        return True
    side_name = "BUY" if signal.side > 0 else "SELL"
    log.info(f"Signal: {side_name} - {signal.reason}")

    t = broker.tick()
    day_start = datetime.strptime(today, "%Y-%m-%d")
    ok, why = safety.can_open_trade(server_time=now, spread=t.ask - t.bid,
                                    open_positions=len(broker.bot_positions()),
                                    trades_today=broker.trades_today(day_start),
                                    equity=acct.equity, day_start_equity=state["day_start_equity"])
    if not ok:
        log.info(f"Skipped: {why}")
        return True

    entry = t.ask if signal.side > 0 else t.bid
    if (signal.side > 0 and signal.stop >= entry) or (signal.side < 0 and signal.stop <= entry):
        log.info("Skipped: price already moved past the stop loss")
        return True
    info = broker.symbol()
    lots = compute_lots(equity=acct.equity, risk_fraction=config.RISK_PER_TRADE,
                        loss_per_lot=broker.loss_per_lot(signal.side, entry, signal.stop),
                        volume_step=info.volume_step, volume_min=info.volume_min,
                        volume_max=info.volume_max, cap=config.MAX_LOTS)
    if lots == 0:
        log.info("Skipped: even the smallest trade would risk more than the limit")
        return True

    if config.DRY_RUN:
        log.info(f"[DRY RUN] would {side_name} {lots} lots at {entry:.2f}, "
                 f"stop {signal.stop:.2f}, target {signal.target:.2f}")
    else:
        result = broker.open_position(signal.side, lots, round(signal.stop, info.digits),
                                      round(signal.target, info.digits))
        log.info(f"Opened {side_name} {lots} lots at {result.price:.2f} (ticket {result.order})")
    journal(time=now, action="open", side=side_name, lots=lots, price=entry, stop=signal.stop,
            target=signal.target, dry_run=config.DRY_RUN, reason=signal.reason)
    return True


def main():
    strategy = STRATEGIES[config.STRATEGY]
    acct = broker.connect()
    try:
        safety.require_demo(acct)
        log.info(f"Connected: DEMO account {acct.login} on {acct.server}, equity {acct.equity:,.2f} "
                 f"{acct.currency}. Strategy '{config.STRATEGY}' on {config.SYMBOL} {config.TIMEFRAME}"
                 f"{' - DRY RUN, no orders will be sent' if config.DRY_RUN else ''}")
        state = load_state()
        once = "--once" in sys.argv
        while True:
            try:
                if not check_once(state, strategy):
                    break
            except safety.RealAccountError:
                raise
            except Exception as e:  # network blips, rejected orders... log and keep going
                log.error(f"Error: {e}")
            if once:
                break
            time.sleep(config.POLL_SECONDS)
    except safety.RealAccountError as e:
        log.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C). Open positions keep their stop loss and take profit.")
    finally:
        broker.disconnect()


if __name__ == "__main__":
    main()
