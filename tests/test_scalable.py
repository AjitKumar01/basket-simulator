import itertools
import unittest
from unittest.mock import patch
import numpy as np
import torch
from basket_simulator.model import toy_model
from basket_simulator.trainable import TrainableBasketModel


class ScalableTests(unittest.TestCase):
    def test_direct_sampler_distribution(self):
        model, context = toy_model()
        model.s["order"] = np.array([1, 0])
        patterns = np.array(list(itertools.product((0, 1), repeat=6)))[1:].reshape(
            -1, 2, 3
        )
        mass = np.exp([model.log_probability(context, y) for y in patterns])
        self.assertAlmostEqual(float(mass.sum()), 1.0, places=12)
        y = model.generate(context, 100000, 21)
        # Test the joint size law, category pair and item marginals, not just support.
        for feature in (
            lambda a: a.sum((1, 2)),
            lambda a: a[:, 0].any(1) * a[:, 1].any(1),
        ):
            v = feature(patterns)
            mean = mass @ v
            variance = mass @ (v - mean) ** 2
            self.assertLess(
                abs(feature(y).mean() - mean), 6 * np.sqrt(variance / len(y))
            )
        np.testing.assert_allclose(
            model.predict(context)["incidence"],
            np.einsum("b,bcj->cj", mass, patterns),
            atol=1e-12,
        )
        mc = model.predict(context, draws=100000, seed=44, method="mc")
        self.assertTrue(
            (
                abs(mc["incidence"] - model.predict(context)["incidence"])
                < 6 * mc["incidence_standard_error"] + 1e-12
            ).all()
        )

    def test_extremely_rare_nonempty_no_rejection(self):
        model, context = toy_model()
        model.s["kappa"][:] = -1000
        y = model.generate(context, 5000, 17)
        self.assertTrue(y.any((1, 2)).all())
        self.assertTrue(np.isfinite(model.predict(context)["incidence"]).all())
        cfg = {
            "C": 2,
            "P": 3,
            "R": 3,
            "users": 2,
            "stores": 2,
            "calendar_support": [1, 30],
        }
        m = TrainableBasketModel(cfg, 1)
        with torch.no_grad():
            m.kappa.fill_(-1000.0)
        ctx = {
            "house": torch.ones(1, dtype=torch.long),
            "store": torch.ones(1, dtype=torch.long),
            "week": torch.full((1,), 10.0, dtype=torch.float64),
            **{
                k: torch.zeros((1, 2, 3), dtype=torch.float64)
                for k in ("dlp", "disp", "mail")
            },
        }
        y = m.sample(ctx, 19).double()
        ll = m.log_probability(ctx, y)
        self.assertTrue(torch.isfinite(ll).all())
        (-ll.mean()).backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in m.parameters()
            )
        )

    def test_large_catalogue_never_enumerates(self):
        cfg = {
            "C": 64,
            "P": 32,
            "R": 4,
            "users": 3,
            "stores": 2,
            "calendar_support": [1, 30],
            "chain_rank": 4,
        }
        m = TrainableBasketModel(cfg, 7)
        s = m.reference()
        self.assertEqual(m.V.numel(), 0)
        ctx = {
            "house": 1,
            "store": 1,
            "week": 12,
            **{k: np.zeros((64, 32)) for k in ("dlp", "disp", "mail")},
        }
        with patch(
            "basket_simulator.model.patterns",
            side_effect=AssertionError("Enumeration used"),
        ):
            y = s.generate(ctx, 200, 9)
            prediction = s.predict(ctx, draws=1024, seed=9)
            self.assertTrue(np.isfinite(s.log_probability(ctx, y[0])))
            response = s.what_if(ctx, 0, 0.9, draws=1024, seed=9)
        self.assertEqual(y.shape, (200, 64, 32))
        self.assertTrue(y.any((1, 2)).all())
        self.assertEqual(prediction["draws"], 1024)
        self.assertIn("incidence_change_standard_error", response)
        self.assertGreaterEqual(response["size_change_standard_error"], 0.0)

    def test_new_price_geometry_contains_independent_law(self):
        cfg = {
            "C": 2,
            "P": 3,
            "R": 3,
            "users": 3,
            "stores": 2,
            "calendar_support": [1, 30],
            "preference_mode": "direct",
            "learn_inclusive_price": True,
            "chain_rank": 2,
        }
        m = TrainableBasketModel(cfg, 3)
        with torch.no_grad():
            m.lam.copy_(torch.tensor([-0.3, 0.2, -0.1, -0.4, 0.1, -0.7]))
            m.rho.zero_()
            m.count_price_scale.fill_(1.0)
            m.inclusive_price_weight.fill_(1.0)
            m.category_price.zero_()
            m.kappa.zero_()
            m.tau.fill_(np.log(np.expm1(1.0)))
            m.cross_p.zero_()
            m.omega.zero_()
            m.linear_p.zero_()
        y = torch.tensor(
            list(itertools.product((0.0, 1.0), repeat=6))[1:], dtype=torch.float64
        ).reshape(-1, 2, 3)
        N = len(y)
        ctx = {
            "house": torch.ones(N, dtype=torch.long),
            "store": torch.ones(N, dtype=torch.long),
            "week": torch.full((N,), 12.0, dtype=torch.float64),
            **{
                k: torch.full((N, 2, 3), 0.1, dtype=torch.float64)
                for k in ("dlp", "disp", "mail")
            },
        }
        u = m.baseline(ctx) - m.choice_price.reshape(2, 3) * ctx["dlp"]
        log0 = torch.nn.functional.logsigmoid(-u).sum((1, 2))
        direct = (y * u - torch.nn.functional.softplus(u)).sum((1, 2)) - torch.log(
            -torch.expm1(log0)
        )
        actual = m.log_probability(ctx, y)
        np.testing.assert_allclose(actual.detach(), direct.detach(), atol=1e-12)
        (-actual.mean()).backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in m.parameters()
                if p.requires_grad
            )
        )
        repeated = {k: v[:1].expand(30000, *v.shape[1:]) for k, v in ctx.items()}
        generated = m.sample(repeated, seed=28).numpy()
        expected = (u[0].sigmoid() / (-torch.expm1(log0[0]))).detach().numpy()
        error = np.sqrt(expected * (1 - expected) / len(generated))
        self.assertTrue((abs(generated.mean(0) - expected) < 6 * error).all())
        np.testing.assert_array_equal(generated, m.sample(repeated, seed=28).numpy())


if __name__ == "__main__":
    unittest.main()
