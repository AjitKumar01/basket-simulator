"""Prepare independent known-truth mission, complement or zero-price worlds."""

import json
import numpy as np
from .. import common as c
from .worlds import World


def main():
    protocol = json.loads((c.OUT / "protocol.json").read_text())
    world = World({"seed": protocol["seed"], "family": protocol["family"]})
    data = c.OUT / "data"
    data.mkdir(exist_ok=False)
    audit = {
        "dataset": "synthetic",
        "family": world.family,
        "seed": world.seed,
        "causal_truth": "Exact independently specified generator; no oracle used in fitting",
        "splits": {},
    }
    for split, size in protocol["sizes"].items():
        offset = {"train": 101, "validation": 102, "test": 103}[split]
        arrays = world.sample(size, world.seed + offset)
        arrays.update(raw_household=arrays["house"], raw_store=arrays["store"])
        np.savez_compressed(data / f"{split}.npz", **arrays)
        audit["splits"][split] = {"baskets_projected": size}
    np.save(data / "reference_price.npy", np.ones(12))
    c.write(
        "data/config.json",
        {
            "C": 3,
            "P": 4,
            "R": 4,
            "users": 17,
            "stores": 1,
            "calendar_support": [1, 52],
            "preference_mode": "direct",
            "learn_inclusive_price": True,
            "chain_rank": 0,
            "lbfgs_history": 3,
            "memory_budget_mib": 14000,
            "scope": "Synthetic conditional nonempty distinct-product baskets; full support shared across models",
        },
    )
    c.write("data_audit.json", audit)
    c.write("prepared_manifest.json", {str(p): c.digest(p) for p in data.iterdir()})
    c.status("prepared", dataset="synthetic")
