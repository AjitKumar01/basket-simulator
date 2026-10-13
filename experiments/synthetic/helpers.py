"""Procedural mission generator, independent of the fitted ESP law."""

import itertools
import numpy as np
from scipy.special import expit


def world_parts(w, house, price, mission):
    index = price.mean(-1)
    eta = np.array([-0.5, -0.8, -0.6]) + w["house_need"][house] + w["missions"][mission]
    eta = eta - index * np.array([1.0, 1.6, 2.2]) + 0.25 * np.roll(index, 1, axis=-1)
    eta -= 0.35 * np.maximum(index, 0.0)
    q = expit(eta)
    patterns = np.array(list(itertools.product((0.0, 1.0), repeat=3)))
    mass = np.prod(np.where(patterns[None] > 0, q[:, None], 1 - q[:, None]), axis=-1)
    mass[:, 0] = 0.0
    mass /= mass.sum(-1, keepdims=True)
    extra = expit(
        np.array([-0.8, -1.1, -0.6])
        + 0.15 * w["missions"][mission]
        - index * np.array([0.8, 1.2, 1.6])
    )
    count = np.stack([(1 - extra) ** 2, 2 * extra * (1 - extra), extra**2], axis=-1)
    utility = (
        w["quality"]
        + w["taste"][house]
        + w["mission_taste"][mission]
        - w["beta"] * price
    )
    return patterns, mass, count, utility


def synthetic_sample(w, N, seed):
    rng = np.random.default_rng(seed)
    h = rng.integers(0, 16, N)
    price = rng.uniform(-0.35, 0.35, (N, 3, 4))
    y = np.zeros_like(price)
    mission = rng.choice(2, N, p=[0.55, 0.45])
    for m in range(2):
        rows = np.flatnonzero(mission == m)
        patterns, mass, count, u = world_parts(w, h[rows], price[rows], m)
        chosen = (rng.random((len(rows), 1)) > mass.cumsum(-1)).sum(-1).clip(max=7)
        present = patterns[chosen]
        size = 1 + (rng.random((len(rows), 3, 1)) > count.cumsum(-1)).sum(-1).clip(
            max=2
        )
        order = np.argsort(-(u + rng.gumbel(size=u.shape)), axis=-1)
        for c in range(3):
            for k in range(3):
                take = (present[:, c] > 0) & (size[:, c] > k)
                y[rows[take], c, order[take, c, k]] = 1.0
    return {
        "house": h + 1,
        "store": np.zeros(N, int),
        "week": rng.integers(1, 53, N),
        "dlp": price,
        "disp": np.zeros_like(price),
        "mail": np.zeros_like(price),
        "member": y,
    }
