"""Exact small-catalogue oracle comparison; oracle is accessed after selection only."""

import json
import numpy as np
import torch
from .. import common as c
from . import worlds as w
from .metrics import generation, global_promotion


@torch.no_grad()
def mass(model, context):
    """Enumerate the 4,095 supported sets, reusing product-head context caches."""
    if model.C != 3 or model.P != 4:
        raise ValueError("Oracle integration requires the full 3-by-4 catalogue")
    result = []
    baskets = torch.tensor(w.SETS.reshape(-1, 3, 4), dtype=torch.float64)
    for start in range(0, len(context["house"]), 8):
        ctx = c.subset(context, slice(start, start + 8))
        if isinstance(model, c.ProductInteractionModel):
            ctx = model.cache(ctx)
        B = len(ctx["house"])
        rows = []
        for first in range(0, len(baskets), 256):
            y = baskets[first : first + 256]
            repeated = {k: v.repeat_interleave(len(y), 0) for k, v in ctx.items()}
            rows.append(
                model.log_probability(repeated, y.repeat(B, 1, 1)).reshape(B, len(y))
            )
        result.append(torch.cat(rows, -1).exp().numpy())
    return w.normalized(np.concatenate(result))


def main(models, test):
    protocol = json.loads((c.OUT / "protocol.json").read_text())
    world = w.World({"family": protocol["family"], "seed": protocol["seed"]})
    count = min(len(test), 1024)
    context, _ = test.batch(slice(0, count))
    truth = world.probability(context["house"].numpy(), context["dlp"].numpy())
    rng = np.random.default_rng(world.seed + 499)
    price = rng.uniform(-0.1, 0.1, (64, 3, 4))
    house = rng.integers(1, 17, 64)
    scenarios = {
        "house": torch.tensor(house),
        "store": torch.zeros(64, dtype=torch.long),
        "week": torch.full((64,), 20.0, dtype=torch.float64),
        "dlp": torch.tensor(price),
        "disp": torch.zeros((64, 3, 4), dtype=torch.float64),
        "mail": torch.zeros((64, 3, 4), dtype=torch.float64),
    }
    actions = w.actions()
    actual = [
        w.moments(world.probability(house, price + a["shift"].reshape(3, 4)))
        for a in actions
    ]
    result = {
        "scope": "Exact generator truth; same fixed catalogue and scenarios; hypothetical distinct-product margin",
        "family": world.family,
        "seed": world.seed,
        "models": {},
    }
    for name, model in models.items():
        c.status("oracle_evaluate", model=name)
        prediction = mass(model, context)
        effects = []
        for action in actions:
            shifted = {
                **scenarios,
                "dlp": torch.tensor(price + action["shift"].reshape(3, 4)),
            }
            effects.append(w.moments(mass(model, shifted)))
        result["models"][name] = {
            "generation": generation(prediction, truth, context["house"].numpy()),
            "what_if": w.response_score(effects, actual, actions),
            "global_promotion": global_promotion(effects, actual, price, actions),
        }
        c.write("oracle_comparison.json", result)
    return result
