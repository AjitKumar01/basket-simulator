"""Product-aware ESP demand with price-aware, approximate future utility.

Likelihood and sampling are exact for this normalized sequential law. The future
utility is an approximation based on a frozen independent-category parent; it is
not exact inference in a global pairwise energy model or causal identification.
"""

import copy
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .trainable import TrainableBasketModel, log_esp


def nonempty(logits):
    no = F.logsigmoid(-logits)
    return torch.logsumexp(F.logsigmoid(logits) + no.cumsum(-1) - no, -1)


class ProductInteractionModel(nn.Module):
    """Add low-rank product interactions to a frozen category-chain parent.

    Average normalized basket laws over a bounded collection of category orders.
    Earlier selections affect later utilities; future-offer summaries provide
    an approximate anticipatory adjustment. Each resulting sequential law is
    normalized exactly, even though the anticipatory utility is approximate.
    """

    def __init__(self, parent, rank=4, seed=0, orders=None):
        super().__init__()
        if rank < 1:
            raise ValueError("Positive product rank required")
        self.parent = copy.deepcopy(parent)
        for p in self.parent.parameters():
            p.requires_grad_(False)
        # The reference expectations are independent-category expectations. A
        # projected chain parent is explicitly recorded, and validation can keep
        # the original chain instead of accepting the new head.
        names = ("V", "linear_p", "omega", "psi0", "psi1")
        projected = any(
            hasattr(parent, n) and bool(getattr(parent, n).detach().ne(0).any())
            for n in names
        )
        with torch.no_grad():
            for name in names:
                if hasattr(self.parent, name):
                    getattr(self.parent, name).zero_()
        self.C, self.P, self.R = parent.C, parent.P, parent.R
        self.rank = int(rank)
        g = torch.Generator().manual_seed(seed)
        self.A = nn.Parameter(torch.zeros((self.C, self.P, rank), dtype=torch.float64))
        self.B = nn.Parameter(
            0.1 * torch.randn((self.C, self.P, rank), generator=g, dtype=torch.float64)
        )
        if self.C <= 10:
            self.category = nn.Parameter(
                torch.zeros((self.C, self.C), dtype=torch.float64)
            )
        else:
            self.category_p = nn.Parameter(
                torch.zeros((self.C, rank), dtype=torch.float64)
            )
            self.category_q = nn.Parameter(
                0.1 * torch.randn((self.C, rank), generator=g, dtype=torch.float64)
            )
        if orders is None:
            if self.C <= 3:
                orders = list(itertools.permutations(range(self.C)))
            else:
                rng = np.random.default_rng(seed)
                orders = [
                    tuple(range(self.C)),
                    tuple(reversed(range(self.C))),
                    tuple(rng.permutation(self.C)),
                    tuple(rng.permutation(self.C)),
                ]
        self.orders = list(dict.fromkeys(tuple(int(c) for c in o) for o in orders))
        if not self.orders or any(
            sorted(o) != list(range(self.C)) for o in self.orders
        ):
            raise ValueError("Category orders must be permutations")
        if len(self.orders) > 6:
            raise ValueError("At most6 mixture orders supported")
        self.config = {
            **parent.config,
            "model_family": "product_interactions",
            "parent_config": dict(parent.config),
            "product_rank": rank,
            "product_orders": [list(o) for o in self.orders],
            "product_seed": seed,
            "projected_category_parent": projected,
            "lookahead": "frozen independent-category expected future purchases",
        }

    def cache(self, ctx):
        """Cache frozen parent terms shared by repeated product-head objective calls."""
        if "_u" in ctx:
            return ctx
        with torch.no_grad():
            u, e, count, base = self.parent.stage_terms(ctx)
        # d log ESP_n / d u_j is the exact fixed-size inclusion probability.
        with torch.enable_grad():
            ug = u.detach().requires_grad_(True)
            eg = log_esp(ug, self.R)
            within = torch.autograd.grad(
                (count.detach().exp() * eg[..., 1:]).sum(), ug
            )[0].detach()
        q = base.sigmoid() / (-torch.expm1(F.logsigmoid(-base).sum(-1)))[:, None]
        return {
            **ctx,
            "_u": u.detach(),
            "_esp": e.detach(),
            "_count": count.detach(),
            "_base": base.detach(),
            "_future": within * q[..., None],
            "_presence": q.detach(),
        }

    def _order_terms(self, ctx, order):
        """Reorder cached category terms and summarize offers later in this order."""
        f = ctx["_future"]
        pa = (f[..., None] * self.A).sum(2)
        pb = (f[..., None] * self.B).sum(2)
        ia = torch.tensor(order, dtype=torch.long)
        pa = pa[:, ia]
        pb = pb[:, ia]
        fa = pa.flip(1).cumsum(1).flip(1) - pa
        fb = pb.flip(1).cumsum(1).flip(1) - pb
        if hasattr(self, "category"):
            s = self.category + self.category.T
            future = []
            for i, c in enumerate(order):
                ids = list(order[i + 1 :])
                future.append((ctx["_presence"][:, ids] * s[c, ids]).sum(-1))
            fc = torch.stack(future, -1)
        else:
            qp = ctx["_presence"][..., None] * self.category_p
            qq = ctx["_presence"][..., None] * self.category_q
            qp, qq = qp[:, ia], qq[:, ia]
            fp = qp.flip(1).cumsum(1).flip(1) - qp
            fq = qq.flip(1).cumsum(1).flip(1) - qq
            fc = (fp * self.category_q[ia] + fq * self.category_p[ia]).sum(-1)
        return fa, fb, fc

    def _local(self, ctx, c, sa, sb, cat):
        """Compute one category offer conditional on the already selected products."""
        shift = 0.5 * (sa @ self.B[c].T + sb @ self.A[c].T)
        u = ctx["_u"][:, c] + shift
        e = log_esp(u, self.R)
        raw = ctx["_count"][:, c] + e[:, 1:] - ctx["_esp"][:, c, 1:]
        return (
            u,
            e,
            F.log_softmax(raw, -1),
            ctx["_base"][:, c] + cat + torch.logsumexp(raw, -1),
        )

    def _empty(self, ctx, order, terms):
        """Evaluate the all-empty category path used in nonempty conditioning."""
        fa, fb, fc = terms
        return self._local_all(ctx, order, fa, fb, fc)[3]

    def _local_all(self, ctx, order, sa, sb, cat):
        """Vectorize category-local terms for a fixed observed basket and order."""
        ia = torch.tensor(order, dtype=torch.long)
        shift = 0.5 * (
            torch.einsum("bcr,cpr->bcp", sa, self.B[ia])
            + torch.einsum("bcr,cpr->bcp", sb, self.A[ia])
        )
        u = ctx["_u"][:, ia] + shift
        e = log_esp(u, self.R)
        raw = ctx["_count"][:, ia] + e[..., 1:] - ctx["_esp"][:, ia, 1:]
        return (
            u,
            e,
            F.log_softmax(raw, -1),
            ctx["_base"][:, ia] + cat + torch.logsumexp(raw, -1),
        )

    def order_ll(self, ctx, y, order):
        """Score a basket under one category order, including nonempty conditioning."""
        terms = self._order_terms(ctx, order)
        fa, fb, fc = terms
        ia = torch.tensor(order, dtype=torch.long)
        ordered = y[:, ia]
        pa = (ordered[..., None] * self.A[ia]).sum(2)
        pb = (ordered[..., None] * self.B[ia]).sum(2)
        sa = pa.cumsum(1) - pa
        sb = pb.cumsum(1) - pb
        n = ordered.sum(-1).long()
        present = n > 0
        x = present.double()
        if hasattr(self, "category"):
            s = (self.category + self.category.T)[ia][:, ia]
            cat = x @ torch.tril(s, -1).T
        else:
            pp = x[..., None] * self.category_p[ia]
            pq = x[..., None] * self.category_q[ia]
            cat = (
                (pp.cumsum(1) - pp) * self.category_q[ia]
                + (pq.cumsum(1) - pq) * self.category_p[ia]
            ).sum(-1)
        u, e, count, eta = self._local_all(ctx, order, sa + fa, sb + fb, cat + fc)
        within = (
            (u * ordered).sum(-1)
            - e.gather(-1, n[..., None]).squeeze(-1)
            + count.gather(-1, (n - 1).clamp(min=0)[..., None]).squeeze(-1)
        )
        ll = torch.where(present, F.logsigmoid(eta) + within, F.logsigmoid(-eta)).sum(
            -1
        )
        return ll - nonempty(self._empty(ctx, order, terms))

    def log_probability(self, ctx, y):
        """Average normalized order probabilities in log space, not their utilities."""
        if (
            y.shape != (len(ctx["house"]), self.C, self.P)
            or not torch.isfinite(y).all()
            or not ((y == 0) | (y == 1)).all()
        ):
            raise ValueError("Binary batch-by-C-by-P baskets required")
        if not y.sum((1, 2)).gt(0).all() or y.sum(-1).gt(self.R).any():
            raise ValueError("Nonempty supported baskets required")
        ctx = self.cache(ctx)
        return torch.logsumexp(
            torch.stack([self.order_ll(ctx, y, o) for o in self.orders], -1), -1
        ) - math.log(len(self.orders))

    def parameter_bounds(self):
        return [
            (-15.0, 15.0)
            for p in self.parameters()
            if p.requires_grad
            for _ in range(p.numel())
        ]

    @torch.no_grad()
    def sample(self, ctx, seed=0):
        """Choose an order, then sample its normalized nonempty sequential basket law."""
        ctx = self.cache(ctx)
        B = len(ctx["house"])
        out = torch.zeros((B, self.C, self.P), dtype=torch.float64)
        gen = torch.Generator().manual_seed(seed)
        choice = torch.randint(len(self.orders), (B,), generator=gen)
        for oi, order in enumerate(self.orders):
            ids = torch.where(choice == oi)[0]
            if not len(ids):
                continue
            cx = {k: v[ids] for k, v in ctx.items()}
            N = len(ids)
            y = torch.zeros((N, self.C, self.P), dtype=torch.float64)
            terms = self._order_terms(cx, order)
            fa, fb, fc = terms
            eta0 = self._empty(cx, order, terms)
            suffix = [torch.full((N,), -torch.inf, dtype=torch.float64)]
            for i in range(self.C - 1, -1, -1):
                suffix.append(
                    torch.logaddexp(
                        F.logsigmoid(eta0[:, i]), F.logsigmoid(-eta0[:, i]) + suffix[-1]
                    )
                )
            suffix = list(reversed(suffix))
            active = torch.zeros(N, dtype=torch.bool)
            sa = torch.zeros((N, self.rank), dtype=torch.float64)
            sb = sa.clone()
            cp = sa.clone()
            cq = sa.clone()
            catstate = (
                torch.zeros((N, self.C), dtype=torch.float64)
                if hasattr(self, "category")
                else None
            )
            for i, c in enumerate(order):
                cat = (
                    catstate[:, c]
                    if catstate is not None
                    else (cp * self.category_q[c] + cq * self.category_p[c]).sum(-1)
                )
                u, e, count, eta = self._local(
                    cx, c, sa + fa[:, i], sb + fb[:, i], cat + fc[:, i]
                )
                q = torch.where(
                    active, eta.sigmoid(), (F.logsigmoid(eta0[:, i]) - suffix[i]).exp()
                )
                present = torch.rand(N, generator=gen) < q.clamp(0, 1)
                active |= present
                required = (
                    torch.multinomial(count.exp(), 1, generator=gen).ravel() + 1
                ) * present
                esp = torch.full(
                    (N, self.P + 1, self.R + 1), -1e30, dtype=torch.float64
                )
                esp[:, :, 0] = 0
                for j in range(self.P - 1, -1, -1):
                    esp[:, j, 1:] = torch.logaddexp(
                        esp[:, j + 1, 1:], u[:, j, None] + esp[:, j + 1, :-1]
                    )
                for j in range(self.P):
                    num = (
                        esp[:, j + 1]
                        .gather(1, (required - 1).clamp(min=0)[:, None])
                        .ravel()
                    )
                    den = esp[:, j].gather(1, required[:, None]).ravel()
                    qj = torch.where(
                        required > 0, (u[:, j] + num - den).exp(), 0
                    ).clamp(0, 1)
                    take = (torch.rand(N, generator=gen) < qj) | (
                        required == self.P - j
                    )
                    y[:, c, j] = take
                    required -= take.long()
                if required.any():
                    raise RuntimeError("ESP count sampling failed")
                sa += y[:, c] @ self.A[c]
                sb += y[:, c] @ self.B[c]
                if catstate is not None:
                    catstate += present[:, None] * (
                        self.category[:, c] + self.category[c, :]
                    )
                else:
                    cp += present[:, None] * self.category_p[c]
                    cq += present[:, None] * self.category_q[c]
            if not active.all():
                raise RuntimeError("Nonempty sampling failed")
            out[ids] = y
        return out.bool()

    def reference(self):
        return ProductInteractionSimulator(self)


class ProductInteractionSimulator:
    """Application API with paired whole-basket Monte Carlo uncertainty."""

    def __init__(self, model):
        self.model = model
        self.config = model.config
        self.C, self.P, self.R = model.C, model.P, model.R
        self._validator = model.parent.reference()

    @classmethod
    def load(cls, checkpoint, config):
        cfg = json.loads(Path(config).read_text())
        parent = TrainableBasketModel(cfg["parent_config"])
        m = ProductInteractionModel(
            parent, cfg["product_rank"], cfg["product_seed"], cfg["product_orders"]
        )
        m.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        if any(not torch.isfinite(v).all() for v in m.state_dict().values()):
            raise ValueError("Nonfinite checkpoint")
        m.config = cfg
        return cls(m)

    def _context(self, ctx, draws=1):
        # Reuse the parent application's input validation, including ID bounds.
        """Validate a single context and expand it into batched Torch covariates."""
        self._validator._stage(ctx)
        return {
            k: torch.as_tensor(
                np.asarray(v),
                dtype=torch.long if k in ("house", "store") else torch.float64,
            )
            .unsqueeze(0)
            .expand(draws, *np.asarray(v).shape)
            for k, v in ctx.items()
        }

    def generate(self, context, draws=1, seed=0):
        """Return binary baskets with shape (draws, C, P) from the fitted head."""
        if not isinstance(draws, int) or draws < 1:
            raise ValueError("draws must be positive")
        base = self.model.cache(self._context(context))
        parts = []
        # Bound temporary offer/utility arrays, and compute the frozen reference
        # expectations once per context instead of once per generated basket.
        for start in range(0, draws, 1024):
            n = min(1024, draws - start)
            ctx = {k: v.expand(n, *v.shape[1:]) for k, v in base.items()}
            parts.append(self.model.sample(ctx, (int(seed) + start) % (2**63)).numpy())
        return np.concatenate(parts)

    def log_probability(self, context, basket):
        """Return the exact mixture log probability of one supported binary basket."""
        y = np.asarray(basket)
        if y.shape != (self.C, self.P) or not np.isin(y, [0, 1]).all() or not y.any():
            raise ValueError("Nonempty binary C-by-P basket required")
        if (y.sum(-1) > self.R).any():
            return -np.inf
        with torch.no_grad():
            return float(
                self.model.log_probability(
                    self._context(context), torch.tensor(y[None], dtype=torch.float64)
                )[0]
            )

    def predict(self, context, draws=8192, seed=0, method="auto"):
        """Estimate product/category marginals and simulation errors from sampled baskets."""
        if method not in ("auto", "mc", "exact"):
            raise ValueError("method must be auto, mc or exact")
        if method == "exact":
            J = self.C * self.P
            if J > 16:
                raise ValueError(
                    "Exact diagnostic expectations require<=16products; use MC"
                )
            y = ((np.arange(1, 1 << J)[:, None] >> np.arange(J)) & 1).reshape(
                -1, self.C, self.P
            )
            y = y[(y.sum(-1) <= self.R).all(-1)]
            parts = []
            with torch.no_grad():
                for k in range(0, len(y), 512):
                    parts.append(
                        self.model.log_probability(
                            self._context(context, len(y[k : k + 512])),
                            torch.tensor(y[k : k + 512], dtype=torch.float64),
                        )
                        .exp()
                        .numpy()
                    )
            p = np.concatenate(parts)
            inc = (p @ y.reshape(-1, J)).reshape(self.C, self.P)
            cat = p @ (y.sum(-1) > 0)
            return {
                "incidence": inc,
                "category_presence": cat,
                "expected_distinct_size": float(inc.sum()),
                "integration": "exact tiny-catalogue diagnostic",
            }
        if not isinstance(draws, int) or draws < 2:
            raise ValueError("MC requires draws>=2")
        y = self.generate(context, draws, seed)
        inc = y.mean(0)
        cat = y.any(-1).mean(0)
        return {
            "incidence": inc,
            "category_presence": cat,
            "expected_distinct_size": float(inc.sum()),
            "integration": "whole-basket Monte Carlo",
            "draws": draws,
            "seed": seed,
            "incidence_standard_error": y.std(0, ddof=1) / np.sqrt(draws),
            "category_standard_error": y.any(-1).std(0, ddof=1) / np.sqrt(draws),
        }

    def recommend(self, context, revealed=None, draws=8192, seed=0):
        """Rank products before a basket, or rank one-item completions by joint probability."""
        if revealed is None:
            scores = self.predict(context, draws, seed)["incidence"].ravel()
            return np.argsort(-scores, kind="stable"), scores
        y = np.asarray(revealed)
        if y.shape != (self.C, self.P) or not np.isin(y, [0, 1]).all():
            raise ValueError("Binary revealed basket required")
        scores = np.full(self.C * self.P, -np.inf)
        for j in np.flatnonzero(~y.astype(bool).ravel()):
            candidate = y.copy()
            candidate.flat[j] = 1
            scores[j] = self.log_probability(context, candidate)
        ids = np.flatnonzero(np.isfinite(scores))
        return ids[np.argsort(-scores[ids], kind="stable")], scores

    def what_if(self, context, product, factor, draws=8192, seed=0):
        """Reprice one product and compare paired before/after model predictions."""
        if not isinstance(product, int) or not 0 <= product < self.C * self.P:
            raise ValueError("Invalid product ID")
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Positive finite price factor required")
        if not isinstance(draws, int) or draws < 2:
            raise ValueError("Paired MC requires draws>=2")
        changed = {**context, "dlp": np.array(context["dlp"], dtype=float, copy=True)}
        changed["dlp"].flat[product] += np.log(factor)
        a = self.generate(context, draws, seed)
        z = self.generate(changed, draws, seed)
        delta = z.astype(float) - a

        def summary(y):
            inc = y.mean(0)
            return {
                "incidence": inc,
                "category_presence": y.any(-1).mean(0),
                "expected_distinct_size": float(inc.sum()),
                "integration": "whole-basket Monte Carlo",
            }

        return {
            "before": summary(a),
            "after": summary(z),
            "incidence_change": delta.mean(0),
            "size_change": float(delta.sum((1, 2)).mean()),
            "incidence_change_standard_error": delta.std(0, ddof=1) / np.sqrt(draws),
            "size_change_standard_error": float(
                delta.sum((1, 2)).std(ddof=1) / np.sqrt(draws)
            ),
            "integration": "paired common-random whole-basket Monte Carlo",
        }
