"""
Thin wrapper around the MetaTrader5 Python library: everything that talks to MT5 lives here.
Only ever touches positions tagged with the bot's MAGIC number - your manual trades are left alone.
"""

import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import pandas as pd

from . import config

TIMEFRAMES = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
              "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
              "D1": mt5.TIMEFRAME_D1}


class BrokerError(RuntimeError):
    pass


def connect():
    if not mt5.initialize():
        raise BrokerError(f"Could not connect to MT5 ({mt5.last_error()}). Is MetaTrader 5 open and logged in?")
    if not mt5.symbol_select(config.SYMBOL, True):
        raise BrokerError(f"Symbol {config.SYMBOL} not available on this account")
    # Right after selecting a symbol MT5 may not have a price yet: wait for the first live tick
    for _ in range(30):
        t = mt5.symbol_info_tick(config.SYMBOL)
        if t is not None and t.time > 0:
            break
        time.sleep(1)
    else:
        raise BrokerError("No live price from MT5 after 30s. Is the market open and MT5 connected?")
    if not config.DRY_RUN and not mt5.terminal_info().trade_allowed:
        raise BrokerError("MT5's 'Algo Trading' button is off. Click it in the MT5 toolbar so it turns "
                          "green, then start the bot again.")
    return mt5.account_info()


def disconnect():
    mt5.shutdown()


def account():
    return mt5.account_info()


def symbol():
    return mt5.symbol_info(config.SYMBOL)


def tick():
    return mt5.symbol_info_tick(config.SYMBOL)


def server_time():
    """MT5 timestamps are server time stored as if they were UTC."""
    t = tick()
    if t is None or t.time == 0:
        raise BrokerError("No live price from MT5 right now")
    return datetime.fromtimestamp(t.time, tz=timezone.utc).replace(tzinfo=None)


def candles(count=300):
    """The last `count` CLOSED candles (the still-forming one is dropped)."""
    rates = mt5.copy_rates_from_pos(config.SYMBOL, TIMEFRAMES[config.TIMEFRAME], 1, count)
    if rates is None or len(rates) == 0:
        raise BrokerError(f"No candles returned ({mt5.last_error()})")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    # Guard against stale history (MT5 still syncing): the last closed candle must be recent
    candle_minutes = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}[config.TIMEFRAME]
    age = server_time() - df.time.iloc[-1].to_pydatetime()
    if age.total_seconds() > 3 * candle_minutes * 60 and server_time().weekday() < 5:
        raise BrokerError(f"Candle history looks stale (last closed candle {df.time.iloc[-1]}); "
                          "MT5 may still be syncing - will retry")
    return df


def bot_positions():
    return [p for p in (mt5.positions_get(symbol=config.SYMBOL) or []) if p.magic == config.MAGIC]


def trades_today(day_start):
    """Number of trades the bot opened since `day_start` (server time)."""
    deals = mt5.history_deals_get(day_start, server_time()) or []
    return sum(1 for d in deals if d.magic == config.MAGIC and d.entry == mt5.DEAL_ENTRY_IN)


def loss_per_lot(side, entry, stop):
    """Money lost by 1.00 lot between entry and stop, in the account currency."""
    order_type = mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL
    profit = mt5.order_calc_profit(order_type, config.SYMBOL, 1.0, entry, stop)
    if profit is None:
        raise BrokerError(f"order_calc_profit failed ({mt5.last_error()})")
    return abs(profit)


def _filling_mode():
    modes = symbol().filling_mode  # bit flags: 1 = fill-or-kill, 2 = immediate-or-cancel
    if modes & 1:
        return mt5.ORDER_FILLING_FOK
    if modes & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def _send(request):
    check = mt5.order_check(request)
    if check is None or check.retcode != 0:
        raise BrokerError(f"Order check failed: {check.comment if check else mt5.last_error()}")
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        raise BrokerError(f"Order rejected: {result.comment if result else mt5.last_error()} "
                          f"(code {result.retcode if result else '?'})")
    return result


def open_position(side, lots, stop, target):
    t = tick()
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": config.SYMBOL, "volume": lots,
        "type": mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL,
        "price": t.ask if side > 0 else t.bid, "sl": stop, "tp": target,
        "deviation": config.DEVIATION_POINTS, "magic": config.MAGIC, "comment": config.COMMENT,
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": _filling_mode(),
    }
    return _send(request)


def close_position(position):
    t = tick()
    is_buy = position.type == mt5.POSITION_TYPE_BUY
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": config.SYMBOL, "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
        "position": position.ticket, "price": t.bid if is_buy else t.ask,
        "deviation": config.DEVIATION_POINTS, "magic": config.MAGIC, "comment": config.COMMENT + " close",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": _filling_mode(),
    }
    return _send(request)
