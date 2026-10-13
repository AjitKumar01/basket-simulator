"""Latent-mission and autoregressive comparators used in current benchmarks."""

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from basket_simulator.trainable import log_esp


def frozen(parent):
    """Copy a fitted parent and disable gradients for all of its parameters."""
    model = copy.deepcopy(parent)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def nonempty(logits):
    """Compute log P(at least one success) by summing first-success events.

    Each product contributes its success probability times the probability
    that all preceding products failed. Log space avoids subtracting a nearly
    unit empty probability from one. For autoregression, use empty-prefix logits.
    """
    log_failure = F.logsigmoid(-logits)
    earlier_log_failure = log_failure.cumsum(-1) - log_failure
    return torch.logsumexp(F.logsigmoid(logits) + earlier_log_failure, -1)


class BinaryAR(nn.Module):
    """Predict each binary product decision from the preceding decisions.

    Products use the saved flattened catalogue order. ``neural=True`` adds a
    16-unit hidden layer; otherwise preceding products enter linearly. The
    benchmark's scalable Neural subclass replaces the dense cross-price matrix.
    Both versions condition the entire probability law on a nonempty basket.
    """

    def __init__(self, parent, neural=False, seed=0):
        super().__init__()
        self.parent = frozen(parent)
        self.J = parent.C * parent.P
        self.C = parent.C
        self.P = parent.P
        self.neural = neural
        self.config = {
            **parent.config,
            "benchmark_baseline": (
                "conditional_nade_H16" if neural else "pairwise_product_autoregression"
            ),
        }
        self.lam = nn.Parameter(torch.zeros(self.J, dtype=torch.float64))
        self.house = nn.Parameter(
            torch.zeros((parent.config["users"], self.J), dtype=torch.float64)
        )
        self.price = nn.Parameter(parent.choice_price.detach().clone())
        self.cross_price = nn.Parameter(
            torch.zeros((self.J, self.J), dtype=torch.float64)
        )
        if neural:
            gen = torch.Generator().manual_seed(seed)
            # Keep these checkpoint names stable: E embeds preceding products,
            # Q embeds prices, and O maps the hidden state to each product logit.
            self.E = nn.Parameter(
                0.05 * torch.randn(self.J, 16, generator=gen, dtype=torch.float64)
            )
            self.Q = nn.Parameter(
                0.05 * torch.randn(self.J, 16, generator=gen, dtype=torch.float64)
            )
            self.O = nn.Parameter(torch.zeros((self.J, 16), dtype=torch.float64))
        else:
            self.W = nn.Parameter(torch.zeros((self.J, self.J), dtype=torch.float64))

    def logits(self, ctx, y):
        """Compute product logits from context and only the earlier basket indicators.

        ctx contains batched covariates; y has shape (batch, J). The exclusive
        prefix sum excludes each product from its own hidden state.
        """
        log_prices = ctx["dlp"].reshape(-1, self.J)
        logits = (
            self.parent.baseline(ctx).reshape(-1, self.J)
            + self.lam
            + self.house[ctx["house"]]
            - self.price * log_prices
        )
        cross_price_weights = self.cross_price - torch.diag(self.cross_price.diag())
        logits = logits + log_prices @ cross_price_weights.T
        if self.neural:
            contributions = y[..., None] * self.E
            earlier_embeddings = contributions.cumsum(1) - contributions
            logits = logits + (
                torch.sigmoid(earlier_embeddings + (log_prices @ self.Q)[:, None])
                * self.O
            ).sum(-1)
        else:
            logits = logits + y @ torch.tril(self.W, -1).T
        return logits

    def log_probability(self, ctx, member):
        """Score the ordered Bernoulli decisions, then condition on a nonempty basket."""
        basket = member.reshape(-1, self.J)
        logits = self.logits(ctx, basket)
        empty_path_logits = self.logits(ctx, torch.zeros_like(basket))
        return (
            basket * F.logsigmoid(logits) + (1 - basket) * F.logsigmoid(-logits)
        ).sum(-1) - nonempty(empty_path_logits)

    def parameter_bounds(self):
        return [
            ((0.0, 8.001) if name == "price" else (-15.0, 15.0))
            for name, p in self.named_parameters()
            if p.requires_grad
            for _ in range(p.numel())
        ]

    @torch.no_grad()
    def sample(self, ctx, seed=0):
        """Sample the autoregression conditioned on at least one selected product.

        Until the first success, adjust each probability using the empty-path
        suffix mass. After a success, ordinary conditional probabilities apply.
        This avoids rejection sampling even when empty baskets are very likely.
        """
        batch_size = len(ctx["house"])
        basket = torch.zeros((batch_size, self.J), dtype=torch.float64)
        generator = torch.Generator().manual_seed(seed)
        empty_path_logits = self.logits(ctx, basket)
        nonempty_suffix = [torch.full((batch_size,), -torch.inf, dtype=torch.float64)]
        for product_index in range(self.J - 1, -1, -1):
            nonempty_suffix.append(
                torch.logaddexp(
                    F.logsigmoid(empty_path_logits[:, product_index]),
                    F.logsigmoid(-empty_path_logits[:, product_index])
                    + nonempty_suffix[-1],
                )
            )
        nonempty_suffix = list(reversed(nonempty_suffix))
        has_selected_product = torch.zeros(batch_size, dtype=torch.bool)
        for product_index in range(self.J):
            logit = self.logits(ctx, basket)[:, product_index]
            selection_probability = torch.where(
                has_selected_product,
                logit.sigmoid(),
                (
                    F.logsigmoid(empty_path_logits[:, product_index])
                    - nonempty_suffix[product_index]
                ).exp(),
            )
            chosen = torch.rand(
                batch_size, generator=generator
            ) < selection_probability.clamp(0, 1)
            basket[:, product_index] = chosen
            has_selected_product |= chosen
        assert has_selected_product.all()
        return basket.reshape(batch_size, self.C, self.P)


class MissionESP(nn.Module):
    """Mix two shopping missions with shopper-specific mixture weights.

    Each mission has its own product tastes, category participation and purchase
    count parameters. Normalize each component on nonempty baskets first, then
    mix its basket probabilities; mixing unnormalized utilities would change
    the model. The fitted independent parent supplies frozen observable controls.
    """

    def __init__(self, parent, seed=0):
        super().__init__()
        self.parent = frozen(parent)
        self.C = parent.C
        self.P = parent.P
        self.R = parent.R
        self.config = {**parent.config, "benchmark_baseline": "latent_mission_esp_K2"}
        g = torch.Generator().manual_seed(seed)
        self.taste = nn.Parameter(
            0.05 * torch.randn((2, self.C, self.P), generator=g, dtype=torch.float64)
        )
        self.category = nn.Parameter(
            0.05 * torch.randn((2, self.C), generator=g, dtype=torch.float64)
        )
        self.nu = nn.Parameter(torch.zeros((2, self.C), dtype=torch.float64))
        self.rho = nn.Parameter(parent.rho.detach().repeat(2, 1))
        self.tau = nn.Parameter(parent.tau.detach().repeat(2, 1))
        self.weights = nn.Parameter(
            torch.zeros((parent.config["users"], 2), dtype=torch.float64)
        )

    def terms(self, ctx, m):
        """Build subset, count and participation terms for mission index m."""
        b = self.parent.baseline(ctx) + self.taste[m]
        u = b - self.parent.choice_price.reshape(self.C, self.P) * ctx["dlp"]
        esp = log_esp(u, self.R)
        ref = log_esp(b, self.R)
        n = torch.arange(1, self.R + 1, dtype=torch.float64)
        nu = (
            getattr(
                self.parent, "count_offset", torch.zeros(self.C, dtype=torch.float64)
            )
            + self.nu[m]
        )
        penalty = self.rho[m, :, None] * (n * (n - 1) / 2) - nu[:, None] * (n - 1)
        a0 = ref[..., 1:] - penalty
        a = a0 + self.parent.count_price_scale[None, :, None] * (
            esp[..., 1:] - ref[..., 1:]
        )
        iv0 = torch.logsumexp(a0, -1)
        iv = (
            iv0
            + self.parent.inclusive_price_weight
            * (torch.logsumexp(esp[..., 1:] - penalty, -1) - iv0)
            - (self.parent.category_price * ctx["dlp"]).sum(-1)
        )
        base = self.parent.kappa + self.category[m] + F.softplus(self.tau[m]) * iv
        return u, esp, F.log_softmax(a, -1), base

    def component_ll(self, ctx, y, m):
        """Score one independently normalized nonempty mission component."""
        u, e, count, base = self.terms(ctx, m)
        n = y.sum(-1).long()
        x = (n > 0).to(u.dtype)
        cat = (x * F.logsigmoid(base) + (1 - x) * F.logsigmoid(-base)).sum(
            -1
        ) - nonempty(base)
        within = (
            (u * y).sum(-1)
            - e.gather(-1, n[..., None]).squeeze(-1)
            + count.gather(-1, (n - 1).clamp(min=0)[..., None]).squeeze(-1)
        )
        return cat + (within * x).sum(-1)

    def log_probability(self, ctx, y):
        """Mix the two component probabilities using this shopper's learned weights."""
        weights = F.log_softmax(self.weights[ctx["house"]], -1)
        return torch.logsumexp(
            torch.stack([self.component_ll(ctx, y, m) for m in range(2)], -1) + weights,
            -1,
        )

    def parameter_bounds(self):
        return [
            (
                (-5.0, 10.0)
                if name == "rho"
                else (-10.0, 8.0) if name == "tau" else (-15.0, 15.0)
            )
            for name, p in self.named_parameters()
            if p.requires_grad
            for _ in range(p.numel())
        ]
