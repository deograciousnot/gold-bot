"""
Step 1: connect Python to the MT5 app and show which account it's logged into.

This script is READ-ONLY. It never places, changes or closes trades.
MT5 must be open and logged in; Python talks to it through the app,
so no password is needed here.
"""

import MetaTrader5 as mt5

SYMBOL = "XAUUSD"

# Connect to the MT5 terminal that's already running on this PC
if not mt5.initialize():
    print("Could not connect to MT5:", mt5.last_error())
    print("Make sure MetaTrader 5 is open and logged in, then try again.")
    raise SystemExit(1)

account = mt5.account_info()

# trade_mode tells us what kind of account this is
ACCOUNT_TYPES = {
    mt5.ACCOUNT_TRADE_MODE_DEMO: "DEMO (fake money - safe)",
    mt5.ACCOUNT_TRADE_MODE_CONTEST: "CONTEST",
    mt5.ACCOUNT_TRADE_MODE_REAL: "REAL (real money!)",
}

print("=== Account ===")
print("Type:     ", ACCOUNT_TYPES.get(account.trade_mode, "unknown"))
print("Server:   ", account.server)
print("Broker:   ", account.company)
print("Currency: ", account.currency)
print("Balance:  ", account.balance)
print("Leverage:  1:%d" % account.leverage)

# Gold's name differs between brokers (XAUUSD, XAUUSD.r, GOLD...), so search for it
gold_symbols = [s.name for s in mt5.symbols_get() if "XAU" in s.name or "GOLD" in s.name.upper()]
print("\n=== Gold symbols on this account ===")
print(", ".join(gold_symbols) or "none found")

symbol = SYMBOL if SYMBOL in gold_symbols else (gold_symbols[0] if gold_symbols else None)
if symbol:
    mt5.symbol_select(symbol, True)
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    print("\n=== %s contract ===" % symbol)
    print("1 lot =       %s ounces" % info.trade_contract_size)
    print("Smallest lot:", info.volume_min)
    print("Price now:    buy %s / sell %s" % (tick.ask, tick.bid))
    print("Spread now:   %.2f dollars" % (tick.ask - tick.bid))

mt5.shutdown()
