"""Deterministic market-data generator for the offline demo and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_market_data(samples: int, seed: int = 7) -> pd.DataFrame:
    """Generate Binance-shaped bid, ask, trade, depth, and volume columns."""
    if samples < 2:
        raise ValueError("samples must be at least 2")
    random = np.random.default_rng(seed)
    price_steps = random.choice(
        np.array([-0.1, 0.0, 0.1], dtype=np.float64),
        size=samples,
        p=(0.28, 0.44, 0.28),
    )
    last = 100.0 + np.cumsum(price_steps)
    spread = np.full(samples, 0.2, dtype=np.float64)
    bid = last - spread / 2
    ask = last + spread / 2
    bid_volume = random.lognormal(mean=2.2, sigma=0.35, size=samples)
    ask_volume = random.lognormal(mean=2.2, sigma=0.35, size=samples)
    trade_volume = random.lognormal(mean=-0.5, sigma=0.5, size=samples)
    cumulative_volume = np.cumsum(trade_volume)

    start = pd.Timestamp("2025-01-01T00:00:00Z")
    timestamps = start + pd.to_timedelta(np.arange(samples) * 100, unit="ms")
    return pd.DataFrame(
        {
            "local_time": timestamps.astype(str),
            "symbol": "DEMOUSDT",
            "last": last,
            "bid": bid,
            "bid_vol": bid_volume,
            "ask": ask,
            "ask_vol": ask_volume,
            "volume": cumulative_volume,
        }
    )
