"""Command-line entry point for the volume-impulse demo."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from liba import PulsaEngine, backend_status, quick_binance_setup
from liba.demo_data import generate_synthetic_market_data


LOGGER = logging.getLogger("vol_impulse")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse demo arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Infer price impact from Binance volume/order-book behavior and "
            "measure instantaneous volume-price inconsistency."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("live", "offline", "backend"),
        default="live",
        help="live PySide6 chart, deterministic offline demo, or backend info",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance futures symbol")
    parser.add_argument("--initial-samples", type=int, default=2000)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--snapshots-per-sample", type=int, default=10)
    parser.add_argument("--render-interval-ms", type=int, default=1000)
    parser.add_argument("--min-window", type=int, default=10)
    parser.add_argument("--max-window", type=int, default=120)
    parser.add_argument("--iceberg-ratio", type=float, default=2.0)
    parser.add_argument("--peak-factor", type=float, default=3.0)
    parser.add_argument("--secondary-peak-factor", type=float, default=0.3)
    parser.add_argument("--update-every", type=int, default=2)
    parser.add_argument("--max-plot-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=7, help="offline demo seed")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    """Configure concise console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_offline(args: argparse.Namespace) -> int:
    """Run the complete inference path on deterministic synthetic market data."""
    if args.initial_samples < max(32, args.max_window * 2):
        raise ValueError(
            "offline initial-samples must be at least max(32, 2 * max-window)"
        )
    frame = generate_synthetic_market_data(args.initial_samples, seed=args.seed)
    engine = PulsaEngine(
        capacity=args.initial_samples,
        tick_size=0.1,
        max_iceberg_ratio=args.iceberg_ratio,
        min_window=args.min_window,
        max_window=args.max_window,
        peak_factor=args.peak_factor,
        peak_factor2=args.secondary_peak_factor,
    )
    processed = engine.load_data(frame)
    snapshot = engine.latest_inconsistency()
    if processed <= 0 or snapshot is None:
        raise RuntimeError("the offline analysis pipeline produced no result")

    summary = {
        **backend_status(),
        "processed_samples": processed,
        "channels": int(snapshot.residual.size),
        "peak_normalized_inconsistency": snapshot.peak_score,
        "mean_absolute_residual": float(abs(snapshot.residual).mean()),
        "max_excess_intensity": float(snapshot.excess_intensity.max()),
    }
    print(json.dumps(summary, indent=2))
    return 0


def run_live(args: argparse.Namespace) -> int:
    """Connect to Binance and start the PySide6 real-time chart."""
    runner = quick_binance_setup(
        symbol=args.symbol,
        iceberg_ratio=args.iceberg_ratio,
        min_window=args.min_window,
        max_window=args.max_window,
        initial_samples=args.initial_samples,
        max_plot_samples=args.max_plot_samples,
        peak_factor=args.peak_factor,
        secondary_peak_factor=args.secondary_peak_factor,
        sample_interval_ms=args.sample_interval_ms,
        snapshots_per_sample=args.snapshots_per_sample,
        update_every=args.update_every,
        render_interval_ms=args.render_interval_ms,
    )
    LOGGER.info(
        "Starting %s with %d ms PySide6 rendering",
        args.symbol.upper(),
        args.render_interval_ms,
    )
    runner.drawer.start(runner.update)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected demo mode."""
    args = parse_args(argv)
    configure_logging(args.verbose)
    if args.mode == "backend":
        print(json.dumps(backend_status(), indent=2))
        return 0
    if args.mode == "offline":
        return run_offline(args)
    return run_live(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.error("%s", error)
        raise SystemExit(2) from error
