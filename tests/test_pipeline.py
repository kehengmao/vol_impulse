"""Deterministic smoke tests for the complete numerical pipeline."""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ["VOL_IMPULSE_AOT"] = "0"

from liba import PulsaEngine, backend_status
from liba.demo_data import generate_synthetic_market_data
from liba.kernels.pulsa_engine import impact


class DemoDataTests(unittest.TestCase):
    def test_generated_data_matches_the_binance_schema(self):
        frame = generate_synthetic_market_data(64, seed=11)

        self.assertEqual(len(frame), 64)
        self.assertEqual(
            list(frame.columns),
            [
                "local_time",
                "symbol",
                "last",
                "bid",
                "bid_vol",
                "ask",
                "ask_vol",
                "volume",
            ],
        )
        self.assertTrue((frame["ask"] > frame["bid"]).all())
        self.assertTrue((frame["volume"].diff().dropna() > 0).all())


class ImpactModelTests(unittest.TestCase):
    def test_forward_power_law_preserves_force_direction(self):
        force = np.array([[-9.0], [0.0], [9.0]], dtype=np.float64)
        parameters = np.array([[2.0, 0.5, 0.0]], dtype=np.float64)

        result = impact.forward(force, parameters)

        np.testing.assert_allclose(result[:, 0], np.array([-6.0, 0.0, 6.0]))


class PipelineTests(unittest.TestCase):
    def test_offline_pipeline_returns_finite_inconsistency(self):
        frame = generate_synthetic_market_data(128, seed=7)
        engine = PulsaEngine(
            capacity=128,
            tick_size=0.1,
            min_window=4,
            max_window=16,
            peak_factor=3,
            peak_factor2=0.3,
        )

        processed = engine.load_data(frame)
        snapshot = engine.latest_inconsistency()

        self.assertEqual(processed, 128)
        self.assertIsNotNone(snapshot)
        self.assertGreater(snapshot.residual.size, 0)
        self.assertTrue(np.isfinite(snapshot.normalized_score).all())
        self.assertLessEqual(snapshot.peak_score, 1.0 + 1e-12)
        self.assertEqual(backend_status()["backend"], "numba-jit")


if __name__ == "__main__":
    unittest.main()
