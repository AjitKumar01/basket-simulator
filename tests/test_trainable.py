import contextlib
import io
import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from basket_simulator.trainable import TrainableBasketModel, log_esp
from basket_simulator.train import fit, load_data
from basket_simulator.example_data import generate


class TrainableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_esp_against_subset_enumeration(self):
        u = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
        actual = log_esp(u, 3)
        for n in range(4):
            values = torch.stack(
                [u[list(idx)].sum() for idx in itertools.combinations(range(3), n)]
            )
            self.assertAlmostEqual(
                float(actual[n].detach()),
                float(torch.logsumexp(values, 0).detach()),
                places=13,
            )
        actual.sum().backward()
        self.assertTrue(torch.isfinite(u.grad).all())

    def test_batched_capped_esp_gradients_against_enumeration(self):
        u = torch.tensor(
            [[0.3, -0.2, 0.1, 0.8, -1.0], [-3.0, 2.0, -0.5, 0.2, 1.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        actual = log_esp(u, 3)
        expected = torch.stack(
            [
                torch.logsumexp(
                    torch.stack(
                        [
                            u[:, list(idx)].sum(-1)
                            for idx in itertools.combinations(range(5), n)
                        ],
                        -1,
                    ),
                    -1,
                )
                for n in range(4)
            ],
            -1,
        )
        np.testing.assert_allclose(actual.detach(), expected.detach(), atol=1e-13)
        weights = torch.tensor([0.0, 0.2, 0.5, 0.3], dtype=torch.float64)
        gradient = torch.autograd.grad((actual * weights).sum(), u, retain_graph=True)[
            0
        ]
        oracle = torch.autograd.grad((expected * weights).sum(), u)[0]
        np.testing.assert_allclose(gradient, oracle, atol=1e-13)

    def test_probability_law_and_gradient(self):
        cfg = {
            "C": 2,
            "P": 3,
            "R": 3,
            "users": 2,
            "stores": 2,
            "calendar_support": [1, 30],
        }
        model = TrainableBasketModel(cfg, 31)
        baskets = torch.tensor(
            list(itertools.product((0.0, 1.0), repeat=6))[1:], dtype=torch.float64
        ).reshape(-1, 2, 3)
        N = len(baskets)
        ctx = {
            "house": torch.ones(N, dtype=torch.long),
            "store": torch.ones(N, dtype=torch.long),
            "week": torch.full((N,), 10.0, dtype=torch.float64),
            **{
                k: torch.full((N, 2, 3), 0.1, dtype=torch.float64)
                for k in ("dlp", "disp", "mail")
            },
        }
        ll = model.log_probability(ctx, baskets)
        self.assertAlmostEqual(float(ll.exp().sum().detach()), 1.0, places=12)
        ref = model.reference()
        single = {
            k: (
                int(v[0])
                if k in ("house", "store")
                else float(v[0]) if k == "week" else v[0].numpy()
            )
            for k, v in ctx.items()
        }
        np.testing.assert_allclose(
            ll.detach().numpy(),
            [ref.log_probability(single, y.numpy()) for y in baskets],
            atol=1e-12,
        )
        loss = -ll.mean()
        loss.backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )
        for name, index in (
            ("choice_price", 0),
            ("count_price_scale", 0),
            ("category_price", (0, 0)),
        ):
            p = getattr(model, name)
            gradient = float(p.grad[index])
            old = float(p[index].detach())
            eps = 1e-5
            with torch.no_grad():
                p[index] = old + eps
                plus = float((-model.log_probability(ctx, baskets).mean()))
                p[index] = old - eps
                minus = float((-model.log_probability(ctx, baskets).mean()))
                p[index] = old
            self.assertAlmostEqual(gradient, (plus - minus) / (2 * eps), places=7)

    def test_end_to_end_fit_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = generate(root / "data", 180, 60, 17)
            model = TrainableBasketModel(cfg, 0)
            train = load_data(root / "data/train.npz", cfg)
            val = load_data(root / "data/validation.npz", cfg)
            with contextlib.redirect_stdout(io.StringIO()):
                result = fit(model, train, val, root / "run", maxiter=6, batch_size=64)
            self.assertLess(result["final_objective"], result["initial_objective"])
            self.assertIn("best_stationarity_passed", result)
            reloaded = TrainableBasketModel.from_checkpoint(
                root / "run/best.pt", root / "run/config.json"
            )
            np.testing.assert_allclose(
                model.log_probability(*val).detach(),
                reloaded.log_probability(*val).detach(),
                atol=1e-13,
            )
            for filename in (
                "best.pt",
                "final.pt",
                "latest.pt",
                "fit.json",
                "history.json",
                "config.json",
                "protocol.json",
            ):
                self.assertTrue((root / "run" / filename).exists())


if __name__ == "__main__":
    unittest.main()
