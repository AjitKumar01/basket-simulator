"""Exact oracle generation and hypothetical global-plan regret."""

import numpy as np
from . import worlds as b


def global_promotion(pred, truth, price, acts):
    ids = [i for i, a in enumerate(acts) if a["kind"] != "increase"]
    rows = []
    for cost in (0.3, 0.6, 0.85):
        for funding in (0.0, 0.1):
            p = []
            t = []
            for i in ids:
                action = acts[i]
                margin = np.exp(price.reshape(-1, 12) + action["shift"]) - cost
                margin[:, action["items"]] += funding
                p.append((pred[i]["incidence"].reshape(-1, 12) * margin).sum(-1))
                t.append((truth[i]["incidence"].reshape(-1, 12) * margin).sum(-1))
            p = np.stack(p)
            t = np.stack(t)
            selected = int(p.mean(-1).argmax())
            best = int(t.mean(-1).argmax())
            rows.append(
                {
                    "cost": cost,
                    "funding": funding,
                    "selected": acts[ids[selected]]["name"],
                    "oracle_selected": acts[ids[best]]["name"],
                    "achieved_margin": float(t[selected].mean()),
                    "best_margin": float(t[best].mean()),
                    "regret": float(t[best].mean() - t[selected].mean()),
                    "optimal": selected == best,
                }
            )
    return rows


def generation(prob, oracle, house):
    p = b.moments(prob)
    t = b.moments(oracle)
    dist = lambda q, z, n: np.bincount(z, weights=q.mean(0), minlength=n)
    pairs = np.stack(
        [b.CAT[:, i] * b.CAT[:, j] for i, j in ((0, 1), (0, 2), (1, 2))], -1
    )
    multiple = (b.SETS.reshape(-1, 3, 4).sum(-1) > 1).mean(-1)
    return {
        "size_tv": float(
            0.5 * abs(dist(prob, b.SIZE, 13) - dist(oracle, b.SIZE, 13)).sum()
        ),
        "product_frequency_mae": float(
            abs(p["incidence"].mean(0) - t["incidence"].mean(0)).mean()
        ),
        "category_pair_mae": float(
            abs(prob.mean(0) @ pairs - oracle.mean(0) @ pairs).mean()
        ),
        "multiple_rate_error": float(
            abs(prob.mean(0) @ multiple - oracle.mean(0) @ multiple)
        ),
        "shopper_mean_size_mae": float(
            np.mean(
                [
                    abs(
                        p["basket_size"][house == h].mean()
                        - t["basket_size"][house == h].mean()
                    )
                    for h in np.unique(house)
                ]
            )
        ),
        "mean_size": float(p["basket_size"].mean()),
        "oracle_mean_size": float(t["basket_size"].mean()),
    }
