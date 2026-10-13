import unittest
import contextlib
import io
import tempfile
from pathlib import Path
import numpy as np
import torch
from basket_simulator.trainable import TrainableBasketModel
from basket_simulator.train import fit


class AnchorTests(unittest.TestCase):
    def test_count_penalty_does_not_change_category_presence(self):
        cfg = {
            "C": 3,
            "P": 4,
            "R": 4,
            "users": 2,
            "stores": 2,
            "calendar_support": [1, 30],
            "preference_mode": "direct",
            "learn_inclusive_price": True,
            "category_anchor": "independent",
            "trainable_parameters": ["rho"],
        }
        m = TrainableBasketModel(cfg, 1)
        ctx = {
            "house": 1,
            "store": 1,
            "week": 15,
            **{k: np.full((3, 4), 0.1) for k in ("dlp", "disp", "mail")},
        }
        before = m.reference().predict(ctx)
        with torch.no_grad():
            m.rho.add_(2.0)
        after = m.reference().predict(ctx)
        np.testing.assert_allclose(
            before["category_presence"], after["category_presence"], atol=1e-14
        )
        self.assertLess(
            after["expected_distinct_size"], before["expected_distinct_size"]
        )
        batch = {
            k: torch.tensor(
                np.asarray(ctx[k])[None],
                dtype=torch.long if k in ("house", "store") else torch.float64,
            )
            for k in ctx
        }
        y = torch.zeros((1, 3, 4), dtype=torch.float64)
        y[0, 0, 0] = 1
        loss = -m.log_probability(batch, y).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(m.rho.grad).all())
        self.assertTrue(
            all(not p.requires_grad for n, p in m.named_parameters() if n != "rho")
        )
        self.assertAlmostEqual(
            float(loss.detach()),
            -m.reference().log_probability(ctx, y[0].numpy()),
            places=13,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                result = fit(
                    m,
                    (batch, y),
                    (batch, y),
                    Path(tmp) / "rho_only",
                    maxiter=2,
                    ridge=0.0,
                )
            self.assertLessEqual(result["final_objective"], result["initial_objective"])


if __name__ == "__main__":
    unittest.main()
