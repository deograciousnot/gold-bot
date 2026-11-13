"""
Position sizing: how big a trade can be so that hitting the stop loss loses about RISK_PER_TRADE.
"""

import math


def compute_lots(*, equity, risk_fraction, loss_per_lot, volume_step, volume_min, volume_max, cap):
    """
    equity        - account equity, in the account currency (KES for your demo)
    loss_per_lot  - how much 1.00 lot would lose if the stop is hit, same currency
    Returns 0.0 when even the smallest allowed trade would risk too much.
    """
    if loss_per_lot <= 0 or equity <= 0:
        return 0.0
    lots = equity * risk_fraction / loss_per_lot
    lots = math.floor(lots / volume_step + 1e-9) * volume_step   # round DOWN, never risk more
    lots = min(lots, volume_max, cap)
    if lots < volume_min:
        return 0.0
    decimals = max(0, -int(math.floor(math.log10(volume_step))))
    return round(lots, decimals)
