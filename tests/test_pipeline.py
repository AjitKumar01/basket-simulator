import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
from basket_simulator.example_data import generate
from basket_simulator.pipeline import (
    staged_fit,
    response_start,
    category_scale_start,
    memory_estimate,
)
from basket_simulator.train import load_data
from basket_simulator.trainable import TrainableBasketModel


class PipelineTests(unittest.TestCase):
    def test_staged_training_and_parent_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = generate(root / "data", 120, 40, 91)
            train = load_data(root / "data/train.npz", config)
            val = load_data(root / "data/validation.npz", config)
            with contextlib.redirect_stdout(io.StringIO()):
                model, result = staged_fit(
                    config,
                    train,
                    val,
                    root / "fit",
                    ridges=(1.0, 10.0),
                    maxiter=5,
                    maxcalls=12,
                    batch_size=64,
                )
            self.assertGreaterEqual(result["structure_validation_ll_gain"], -1e-12)
            checkpoint = TrainableBasketModel.from_checkpoint(
                root / "fit/best.pt", root / "fit/config.json"
            )
            np.testing.assert_allclose(
                model.log_probability(*val).detach(),
                checkpoint.log_probability(*val).detach(),
                atol=1e-12,
            )
            self.assertIn("No joint optimum", result["stationarity_scope"])

    def test_response_migration_and_optional_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = generate(root / "data", 120, 40, 93)
            cfg.update(learn_count_offset=True, fit_response=True)
            train = load_data(root / "data/train.npz", cfg)
            val = load_data(root / "data/validation.npz", cfg)
            with contextlib.redirect_stdout(io.StringIO()):
                m, result = staged_fit(
                    cfg, train, val, root / "fit", maxiter=5, maxcalls=12, batch_size=64
                )
            self.assertEqual(result["selected_stage"], "response")
            self.assertGreaterEqual(
                result["response"]["best_validation_ll"],
                result["structure"]["best_validation_ll"] - 1e-12,
            )
            # Nonzero cross links must survive the new response coordinates.
            import torch

            with torch.no_grad():
                m.cross_p.add_(0.03)
                m.cross_q.add_(0.02)
            migrated = response_start(m)
            np.testing.assert_allclose(
                m.log_probability(*val).detach(),
                migrated.log_probability(*val).detach(),
                atol=1e-12,
            )
            self.assertTrue(migrated.count_offset.requires_grad)
            self.assertFalse(migrated.house_effect.requires_grad)
            calibrated = category_scale_start(migrated)
            np.testing.assert_allclose(
                migrated.log_probability(*val).detach(),
                calibrated.log_probability(*val).detach(),
                atol=1e-12,
            )
            self.assertTrue(calibrated.tau.requires_grad)
            self.assertFalse(calibrated.house_effect.requires_grad)
            cfg["memory_budget_mib"] = 1e-4
            with self.assertRaises(ValueError):
                staged_fit(cfg, train, val, root / "reject")
            self.assertFalse((root / "reject").exists())
            self.assertGreater(memory_estimate(cfg)["dense_preference_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
