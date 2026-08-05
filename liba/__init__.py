"""Volume-impulse analysis package."""

from __future__ import annotations

from .kernels import (
    AOT_ENABLED,
    EgiDataChannel,
    InconsistencySnapshot,
    PulsaEngine,
    backend_status,
)


def quick_binance_setup(*args, **kwargs):
    """Create the live Binance runner without importing the GUI eagerly."""
    from .pulsa_runner import quick_binance_setup as _quick_binance_setup

    return _quick_binance_setup(*args, **kwargs)


__all__ = [
    "AOT_ENABLED",
    "EgiDataChannel",
    "InconsistencySnapshot",
    "PulsaEngine",
    "backend_status",
    "quick_binance_setup",
]
