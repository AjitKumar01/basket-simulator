"""Whole-basket applications using ESP dynamic programming and direct chain sampling.

Likelihood and sampling never enumerate category patterns or product subsets.
Small-category marginal expectations can be exact; larger ones report Monte Carlo
error with exact within-category expectations (Rao-Blackwellization).
"""

import itertools
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp


def patterns(n):
    if n > 10:
        raise ValueError(
            "Exhaustive reference tables require at most10 categories/items"
        )
    return np.array(list(itertools.product((0.0, 1.0), repeat=n)))


def esp_prefix(u, R):
    C, P = u.shape
    out = np.full((C, P + 1, R + 1), -np.inf)
    out[:, :, 0] = 0.0
    for j in range(P):
        out[:, j + 1, 1:] = np.logaddexp(out[:, j, 1:], u[:, j, None] + out[:, j, :-1])
    return out


def esp_suffix(u, R):
    return esp_prefix(u[:, ::-1], R)[:, ::-1]


class BasketSimulator:
    """Single-context inference using the fitted category/count/subset law.

    Generate baskets directly, score their exact probability, and integrate
    product/category marginals exactly for small category counts or with
    reported Monte Carlo error for larger catalogues.
    """

    def __init__(self, state, config):
        self.s = {k: np.asarray(v) for k, v in state.items()}
        self.config = config
        self.C, self.P, self.R = (int(config[k]) for k in ("C", "P", "R"))
        if self.C < 1 or not 1 <= self.R <= self.P:
            raise ValueError("Require C>=1 and 1<=R<=P")
        if any(not np.isfinite(v).all() for v in self.s.values()):
            raise ValueError("Nonfinite checkpoint")
        if not np.array_equal(np.sort(self.s["order"]), np.arange(self.C)):
            raise ValueError("Invalid category order")
        if not np.array_equal(
            self.s["layout"], np.arange(self.C * self.P).reshape(self.C, self.P)
        ):
            raise ValueError("Require uniform, fully offered contiguous catalogue")
        support = self.s["calendar_support"]
        if support.shape != (2,) or support[0] > support[1]:
            raise ValueError("Invalid calendar support")
        fixed = {
            "gamma": 0.0,
            "history_coefficient": 0.0,
            "mu_single": 1.0,
            "mu_multi": 1.0,
            "log_taste_scale": 0.0,
            "log_price_single": 0.0,
            "log_price_multi": 0.0,
            "history_single": 0.0,
            "history_multi": 0.0,
            "item_price_loading": 1.0,
            "price_sd": 0.0,
            "count_price_gate": 1.0,
            "category_price_gate": 1.0,
        }
        for name, value in fixed.items():
            if name in self.s and not np.allclose(
                self.s[name], value, atol=1e-12, rtol=0
            ):
                raise ValueError("Unsupported active extension: " + name)
        if "choice_mapping" in self.s and np.any(self.s["choice_mapping"] != 0):
            raise ValueError("Nested choices unsupported")
        for name in ("choice_price", "category_price", "count_price_scale"):
            if (self.s[name] < 0).any():
                raise ValueError("Negative physical price parameter: " + name)
        if (
            "inclusive_price_weight" in self.s
            and (
                (self.s["inclusive_price_weight"] < 0)
                | (self.s["inclusive_price_weight"] > 1)
            ).any()
        ):
            raise ValueError("inclusive_price_weight must be in [0,1]")

    @classmethod
    def load(cls, checkpoint, config):
        import torch

        cfg = json.loads(Path(config).read_text())
        if cfg.get("model_family") == "product_interactions":
            from .product_interactions import ProductInteractionSimulator

            return ProductInteractionSimulator.load(checkpoint, config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        return cls({k: v.detach().numpy() for k, v in state.items()}, cfg)

    def _stage(self, context):
        """Validate context-only input and compute utilities, ESP/count terms and category logits."""
        if set(context) != {"house", "store", "week", "dlp", "disp", "mail"}:
            raise ValueError("Supply context only, no targets")
        s = self.s
        h, store = context["house"], context["store"]
        if not isinstance(h, (int, np.integer)) or not 0 <= h < len(s["theta"]):
            raise ValueError("Invalid internal household ID")
        if not isinstance(store, (int, np.integer)) or not 0 <= store < len(s["xi"]):
            raise ValueError("Invalid internal store ID")
        price, display, mail = (
            np.asarray(context[k], dtype=float) for k in ("dlp", "disp", "mail")
        )
        if any(
            v.shape != (self.C, self.P) or not np.isfinite(v).all()
            for v in (price, display, mail)
        ):
            raise ValueError("Require finite C-by-P input arrays")
        if not np.isfinite(context["week"]):
            raise ValueError("week must be finite")
        phase = (np.clip(context["week"], *s["calendar_support"]) - 1) * 2 * np.pi / 52
        cal = np.array(
            [np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)]
        )
        if "house_effect" in s:
            b = (
                s["lam"]
                + s["house_effect"][h]
                + s["store_effect"][store]
                + cal @ s["mu"].T
            )
        else:
            b = (
                s["lam"]
                + s["theta"][h] @ s["alpha"].T
                + s["xi"][store] @ s["zeta"].T
                + cal @ s["mu"].T
            )
        b = (
            b.reshape(self.C, self.P)
            + s["w_dsp"].reshape(self.C, self.P) * display
            + s["w_mlr"].reshape(self.C, self.P) * mail
        )
        u = b - s["choice_price"].reshape(self.C, self.P) * price
        prefix = esp_prefix(u, self.R)
        e = prefix[:, -1]
        e0 = esp_prefix(b, self.R)[:, -1]
        n = np.arange(1, self.R + 1)
        penalty = s["rho"][:, None] * (n * (n - 1) / 2) - s.get(
            "count_offset", np.zeros(self.C)
        )[:, None] * (n - 1)
        a0 = e0[:, 1:] - penalty
        a = a0 + s["count_price_scale"][:, None] * (e[:, 1:] - e0[:, 1:])
        count = a - logsumexp(a, axis=1, keepdims=True)
        iv0 = logsumexp(a0, axis=1)
        weight = s.get("inclusive_price_weight", np.zeros(self.C))
        iv = (
            logsumexp(e[:, 1:], axis=1) - (s["category_price"] * price).sum(1)
            if bool(s.get("independent_category_anchor", False))
            else (
                iv0
                + weight * (logsumexp(e[:, 1:] - penalty, axis=1) - iv0)
                - (s["category_price"] * price).sum(1)
                if bool(s["separate_category_price"])
                else logsumexp(e[:, 1:] - penalty, axis=1)
            )
        )
        cross = s["cross_p"] @ (s["cross_q"].T @ iv) - iv * (
            s["cross_p"] * s["cross_q"]
        ).sum(1)
        base = (
            s["kappa"]
            + np.logaddexp(0.0, s["tau"]) * iv
            + cross
            + s["U"] @ s["theta"][h]
        )
        return u, e, count, base, prefix

    def _eta(self, base, patterns):
        s = self.s
        x = patterns[..., s["order"].astype(int)]
        contrib = x[..., None] * s["W"]
        hidden = (
            np.maximum(0.0, np.cumsum(contrib, axis=-2) - contrib + s["h0"])
            * s["omega"]
        ).sum(-1)
        count = np.cumsum(x, axis=-1) - x
        if "linear_p" in s:
            contribution = x[..., None] * s["linear_q"]
            linear = (
                (np.cumsum(contribution, axis=-2) - contribution) * s["linear_p"]
            ).sum(-1)
        else:
            linear = x @ np.tril(s["V"], -1).T
        return (
            base[s["order"].astype(int)]
            + linear
            + hidden
            + s["psi0"][count.astype(int)]
            + s["psi1"] * count
        )

    def _empty_suffix(self, base):
        eta = self._eta(base, np.zeros(self.C))
        log_yes = -np.logaddexp(0.0, -eta)
        log_no = -np.logaddexp(0.0, eta)
        suffix = np.full(self.C + 1, -np.inf)
        for c in range(self.C - 1, -1, -1):
            suffix[c] = np.logaddexp(log_yes[c], log_no[c] + suffix[c + 1])
        return eta, suffix

    def _category_ll(self, base, category):
        x = category[..., self.s["order"].astype(int)].astype(float)
        eta = self._eta(base, category)
        logmass = (-x * np.logaddexp(0.0, -eta) - (1 - x) * np.logaddexp(0.0, eta)).sum(
            -1
        )
        return logmass - self._empty_suffix(base)[1][0]

    def _category_table(self, base):
        cats = patterns(self.C)[1:]
        logmass = self._category_ll(base, cats)
        return cats, np.exp(logmass - logsumexp(logmass))

    def _within_marginals(self, stage):
        u, e, count, _, prefix = stage
        suffix = esp_suffix(u, self.R)
        weights = np.exp(count)
        result = np.zeros((self.C, self.P))
        for j in range(self.P):
            for n in range(1, self.R + 1):
                left = np.arange(n)
                leave = logsumexp(
                    prefix[:, j, left] + suffix[:, j + 1, n - 1 - left], axis=1
                )
                result[:, j] += weights[:, n - 1] * np.exp(u[:, j] + leave - e[:, n])
        return result

    def _categories(self, base, draws, seed):
        """Exact conditioned-nonempty ancestral sampler; no rejection or enumeration."""
        s = self.s
        order = s["order"].astype(int)
        empty, suffix = self._empty_suffix(base)
        rng = np.random.default_rng(seed)
        x = np.zeros((draws, self.C), bool)
        hidden = np.broadcast_to(s["h0"], (draws, len(s["h0"]))).copy()
        counts = np.zeros(draws, int)
        linear = np.zeros((draws, s["linear_p"].shape[1])) if "linear_p" in s else None
        for i, c in enumerate(order):
            link = (
                linear @ s["linear_p"][i]
                if linear is not None
                else x[:, order[:i]] @ s["V"][i, :i]
            )
            eta = (
                base[c]
                + link
                + (np.maximum(0.0, hidden) * s["omega"][i]).sum(-1)
                + s["psi0"][counts]
                + s["psi1"][i] * counts
            )
            p = expit(eta)
            first = np.exp(-np.logaddexp(0.0, -empty[i]) - suffix[i])
            p = np.where(counts > 0, p, first)
            selected = rng.random(draws) < np.clip(p, 0, 1)
            x[:, c] = selected
            hidden += selected[:, None] * s["W"][i]
            counts += selected
            if linear is not None:
                linear += selected[:, None] * s["linear_q"][i]
        if not x.any(1).all():
            raise RuntimeError("Nonempty chain sampler failed")
        return x

    def tables(self, context):
        """Tiny exhaustive reference only; never used by the scalable sampler."""
        if self.P > 10:
            raise ValueError("Subset reference tables require P<=10")
        stage = self._stage(context)
        u, e, count, base, _ = stage
        cats = patterns(self.C)
        mass = np.r_[0.0, self._category_table(base)[1]]
        subsets = patterns(self.P)[1:]
        subsets = subsets[subsets.sum(1) <= self.R]
        sizes = subsets.sum(1).astype(int)
        within = count[:, sizes - 1] + u @ subsets.T - e[:, sizes]
        return (
            np.log(np.where(mass > 0, mass, 1.0)) + np.where(mass > 0, 0.0, -np.inf),
            within,
        )

    def predict(self, context, draws=8192, seed=0, method="auto"):
        """Return product/category marginals and expected distinct basket size.

        method="auto" enumerates category patterns only when C <= 10; otherwise
        use seeded category draws with exact within-category marginals. Input
        contains covariates only, never target products or basket size.
        """
        if method not in ("auto", "exact", "mc"):
            raise ValueError("method must be auto, exact or mc")
        stage = self._stage(context)
        within = self._within_marginals(stage)
        exact = method == "exact" or (method == "auto" and self.C <= 10)
        if exact:
            cats, mass = self._category_table(stage[3])
            q = mass @ cats
            metadata = {"integration": "exact"}
        else:
            if not isinstance(draws, int) or draws < 2:
                raise ValueError("MC requires draws>=2")
            samples = self._categories(stage[3], draws, seed)
            q = samples.mean(0)
            error = samples.std(0, ddof=1) / np.sqrt(draws)
            metadata = {
                "integration": "category Monte Carlo; exact within-category",
                "draws": draws,
                "seed": seed,
                "category_standard_error": error,
                "incidence_standard_error": error[:, None] * within,
            }
        incidence = q[:, None] * within
        return {
            "incidence": incidence,
            "category_presence": q,
            "expected_distinct_size": float(incidence.sum()),
            **metadata,
        }

    def log_probability(self, context, basket):
        """Return the normalized log probability of one binary supported basket."""
        y = np.asarray(basket)
        if y.shape != (self.C, self.P) or not np.isin(y, [0, 1]).all() or not y.any():
            raise ValueError("Nonempty binary C-by-P basket required")
        n = y.sum(1).astype(int)
        if (n > self.R).any():
            return -np.inf
        u, e, count, base, _ = self._stage(context)
        present = n > 0
        rows = np.flatnonzero(present)
        within = (u * y).sum(1)[rows] - e[rows, n[rows]] + count[rows, n[rows] - 1]
        return float(self._category_ll(base, present) + within.sum())

    def generate(self, context, draws=1, seed=0):
        """Draw nonempty binary baskets with shape (draws, C, P) from the fitted law."""
        if not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be positive")
        u, e, count, base, _ = self._stage(context)
        cats = self._categories(base, draws, seed)
        rng = np.random.default_rng(seed ^ 0x5A17)
        suffix = esp_suffix(u, self.R)
        out = np.zeros((draws, self.C, self.P), bool)
        for c in range(self.C):
            required = rng.choice(self.R, draws, p=np.exp(count[c])) + 1
            required = np.where(cats[:, c], required, 0)
            for j in range(self.P):
                remaining = required.copy()
                positive = remaining > 0
                logp = np.full(draws, -np.inf)
                logp[positive] = (
                    u[c, j]
                    + suffix[c, j + 1, remaining[positive] - 1]
                    - suffix[c, j, remaining[positive]]
                )
                take = rng.random(draws) < np.clip(np.exp(logp), 0, 1)
                take |= remaining == self.P - j
                out[:, c, j] = take
                required -= take
            if required.any():
                raise RuntimeError("ESP sampler did not satisfy the count")
        return out

    def recommend(self, context, revealed=None, draws=8192, seed=0):
        """Rank marginal purchases, or one missing product conditional on a revealed basket."""
        if revealed is None:
            scores = self.predict(context, draws, seed)["incidence"].ravel()
            return np.argsort(-scores, kind="stable"), scores
        y = np.asarray(revealed).copy()
        if y.shape != (self.C, self.P) or not np.isin(y, [0, 1]).all():
            raise ValueError("Binary revealed basket required")
        scores = np.full(self.C * self.P, -np.inf)
        for j in np.flatnonzero(~y.astype(bool).ravel()):
            candidate = y.copy()
            candidate.flat[j] = 1
            scores[j] = self.log_probability(context, candidate)
        candidates = np.flatnonzero(np.isfinite(scores))
        return candidates[np.argsort(-scores[candidates], kind="stable")], scores

    def what_if(self, context, product, factor, draws=8192, seed=0):
        """Apply a multiplicative price change and compare paired model predictions.

        factor=0.9 means a 10% discount. Add log(factor) to the chosen product
        price covariate while keeping the training reference and other context
        fixed. Model responses are not automatically causal effects.
        """
        if not isinstance(product, int) or not 0 <= product < self.C * self.P:
            raise ValueError("Invalid product ID")
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Positive finite price factor required")
        changed = {**context, "dlp": np.array(context["dlp"], dtype=float, copy=True)}
        changed["dlp"].flat[product] += np.log(factor)
        before, after = self.predict(context, draws, seed), self.predict(
            changed, draws, seed
        )
        result = {
            "before": before,
            "after": after,
            "incidence_change": after["incidence"] - before["incidence"],
            "size_change": after["expected_distinct_size"]
            - before["expected_distinct_size"],
        }
        if self.C > 10:
            a, b = self._stage(context), self._stage(changed)
            wa, wb = self._within_marginals(a), self._within_marginals(b)
            xa, xb = self._categories(a[3], draws, seed), self._categories(
                b[3], draws, seed
            )
            pair = (xa & xb).mean(0)[:, None]
            qa, qb = xa.mean(0)[:, None], xb.mean(0)[:, None]
            mean = qb * wb - qa * wa
            variance = qb * wb * wb + qa * wa * wa - 2 * pair * wa * wb - mean * mean
            result["incidence_change_standard_error"] = np.sqrt(
                np.maximum(0.0, variance) / (draws - 1)
            )
            delta = xb @ wb.sum(1) - xa @ wa.sum(1)
            result["size_change_standard_error"] = float(
                delta.std(ddof=1) / np.sqrt(draws)
            )
            result["integration"] = (
                "paired common-random category Monte Carlo; exact within-category"
            )
        return result


def toy_model():
    """Illustrative parameters, NOT a fitted model or a scientific benchmark."""
    C, P, J = 2, 3, 6
    s = {
        "lam": np.array([0.4, 0.1, -0.2, 0.2, 0.0, -0.3]),
        "theta": np.zeros((1, 1)),
        "alpha": np.zeros((J, 1)),
        "xi": np.zeros((1, 1)),
        "zeta": np.zeros((J, 1)),
        "mu": np.zeros((J, 4)),
        "w_dsp": np.zeros(J),
        "w_mlr": np.zeros(J),
        "rho": np.full(C, 2.0),
        "choice_price": np.ones(J),
        "category_price": np.full((C, P), 0.2),
        "count_price_scale": np.full(C, 0.2),
        "calendar_support": np.array([1, 30]),
        "separate_category_price": np.array(True),
        "layout": np.arange(J).reshape(C, P),
        "kappa": np.full(C, -1.0),
        "tau": np.zeros(C),
        "cross_p": np.zeros((C, 1)),
        "cross_q": np.zeros((C, 1)),
        "U": np.zeros((C, 1)),
        "order": np.arange(C),
        "V": np.array([[0.0, 0.0], [0.4, 0.0]]),
        "W": np.zeros((C, 1)),
        "omega": np.zeros((C, 1)),
        "h0": np.zeros(1),
        "psi0": np.zeros(C + 1),
        "psi1": np.zeros(C),
    }
    context = {
        "house": 0,
        "store": 0,
        "week": 10,
        **{k: np.zeros((C, P)).tolist() for k in ("dlp", "disp", "mail")},
    }
    return BasketSimulator(s, {"C": C, "P": P, "R": P}), context
