"""Full later likelihood, shared-context generation, paired uncertainty and scenarios."""

import itertools
import json
import time
import resource
import html
import csv
import sys
from pathlib import Path
from . import common as c
import numpy as np
import torch
from torch.nn import functional as F

cfg = (
    json.loads((c.OUT / "data/config.json").read_text())
    if (c.OUT / "data/config.json").exists()
    else {"C": 3, "P": 2}
)
C, P = cfg["C"], cfg["P"]
J = C * P
PAIRS = list(itertools.combinations(range(C), 2))
rng = np.random.default_rng(972010)
TRIPLES = [
    tuple(sorted(int(v) for v in rng.choice(C, 3, replace=False))) for _ in range(512)
]
TRIPLES = list(dict.fromkeys(TRIPLES))[:256]
SEEDS = (972401, 972402, 972403)
DRAWS = 128


@torch.no_grad()
def esp_draw(u, required, g):
    """Draw exact weighted subsets of the required size using ESP prefix recursion."""
    P = u.shape[-1]
    R = int(required.max())
    esp = torch.full((*u.shape[:-1], P + 1, R + 1), -1e30, dtype=torch.float64)
    esp[..., 0] = 0
    for j in range(P - 1, -1, -1):
        esp[..., j, 1:] = torch.logaddexp(
            esp[..., j + 1, 1:], u[..., j, None] + esp[..., j + 1, :-1]
        )
    y = torch.zeros_like(u, dtype=torch.bool)
    n = required.clone()
    for j in range(P):
        num = esp[..., j + 1, :].gather(-1, (n - 1).clamp(min=0)[..., None]).squeeze(-1)
        den = esp[..., j, :].gather(-1, n[..., None]).squeeze(-1)
        q = torch.where(n > 0, (u[..., j] + num - den).exp(), 0).clamp(0, 1)
        take = (torch.rand(n.shape, generator=g) < q) | (n == P - j)
        y[..., j] = take
        n -= take.long()
    assert not n.any()
    return y


@torch.no_grad()
def independent_presence(base, g):
    """Draw independent category indicators conditioned on at least one being present."""
    lognone = F.logsigmoid(-base)
    total = -lognone.flip(-1).cumsum(-1).flip(-1)
    ln = torch.log(-torch.expm1(-total))
    active = torch.zeros(len(base), dtype=torch.bool)
    x = torch.zeros_like(base, dtype=torch.bool)
    for j in range(base.shape[1]):
        q = torch.where(
            active, base[:, j].sigmoid(), (F.logsigmoid(base[:, j]) - ln[:, j]).exp()
        )
        take = torch.rand(len(base), generator=g) < q.clamp(0, 1)
        x[:, j] = take
        active |= take
    assert active.all()
    return x


class Adapter:
    """Provide a common cache-and-sample interface for every fitted comparator.

    Cache only context-dependent terms; each sample call draws a new basket.
    This lets generation, recommendation and scenarios share contexts, draw
    budgets and seeds without forcing different models to share a probability law.
    """

    def __init__(self, m):
        self.m = m

    @torch.no_grad()
    def cache(self, ctx):
        """Precompute fixed offer terms for the concrete model implementation."""
        m = self.m
        if isinstance(m, c.ProductInteractionModel):
            return m.cache(ctx)
        if isinstance(m, c.Independent):
            u = (m.baseline(ctx) - m.choice_price.reshape(C, P) * ctx["dlp"]).reshape(
                -1, J
            )
            return {"_ind": u}
        if isinstance(m, c.SizeESP):
            u, q = m.terms(ctx)
            return {"_size_u": u, "_size_q": q}
        if isinstance(m, c.Neural):
            d = ctx["dlp"].reshape(-1, J)
            u = (
                m.parent.baseline(ctx).reshape(-1, J)
                + m.lam
                + m.house[ctx["house"]]
                - m.price * d
                + (d @ m.cross_right) @ m.cross_left.T
                - d * (m.cross_left * m.cross_right).sum(-1)
            )
            return {"_neural_u": u, "_neural_price": d @ m.Q}
        if isinstance(m, c.MissionESP):
            out = {"_mix": F.softmax(m.weights[ctx["house"]], -1)}
            for k in range(2):
                u, e, q, b = m.terms(ctx, k)
                out.update({f"_u{k}": u, f"_q{k}": q, f"_base{k}": b})
            return out
        u, e, q, b = m.stage_terms(ctx)
        return {"_u": u, "_q": q, "_base": b}

    @torch.no_grad()
    def sample(self, x, seed):
        """Generate one basket per cached context under the selected model's own law."""
        m = self.m
        g = torch.Generator().manual_seed(seed)
        if isinstance(m, c.ProductInteractionModel):
            return m.sample(x, seed)
        if "_ind" in x or "_neural_u" in x:
            u = x.get("_ind", x.get("_neural_u"))
            B = len(u)
            neural = "_neural_u" in x
            empty = (
                u + (torch.sigmoid(x["_neural_price"][:, None]) * m.O).sum(-1)
                if neural
                else u
            )
            ln = torch.log(
                -torch.expm1(-F.softplus(empty).flip(-1).cumsum(-1).flip(-1))
            )
            active = torch.zeros(B, dtype=torch.bool)
            y = torch.zeros((B, J), dtype=torch.bool)
            hidden = torch.zeros((B, 16), dtype=torch.float64) if neural else None
            for j in range(J):
                eta = (
                    u[:, j]
                    + (torch.sigmoid(hidden + x["_neural_price"]) * m.O[j]).sum(-1)
                    if neural
                    else u[:, j]
                )
                q = torch.where(
                    active, eta.sigmoid(), (F.logsigmoid(empty[:, j]) - ln[:, j]).exp()
                )
                take = torch.rand(B, generator=g) < q.clamp(0, 1)
                y[:, j] = take
                active |= take
                if neural:
                    hidden += take[:, None] * m.E[j]
            assert active.all()
            return y.reshape(B, C, P)
        if "_size_u" in x:
            n = torch.multinomial(x["_size_q"].exp(), 1, generator=g).ravel() + 1
            # Bound global ESP suffix memory; same law for every chunk.
            out = []
            for start in range(0, len(n), 128):
                out.append(
                    esp_draw(
                        x["_size_u"][start : start + 128], n[start : start + 128], g
                    )
                )
            return torch.cat(out).reshape(-1, C, P)
        if "_mix" in x:
            k = torch.multinomial(x["_mix"], 1, generator=g).ravel()
            out = torch.zeros((len(k), C, P), dtype=torch.bool)
            for z in range(2):
                ids = torch.where(k == z)[0]
                if not len(ids):
                    continue
                present = independent_presence(x[f"_base{z}"][ids], g)
                n = (
                    torch.multinomial(
                        x[f"_q{z}"][ids].exp().reshape(-1, m.R), 1, generator=g
                    ).reshape(len(ids), C)
                    + 1
                ) * present
                out[ids] = esp_draw(x[f"_u{z}"][ids], n, g)
            return out
        base = x["_base"]
        B = len(base)
        empty = m.category_logits(base, torch.zeros_like(base))
        ln = torch.log(-torch.expm1(-F.softplus(empty).flip(-1).cumsum(-1).flip(-1)))
        active = torch.zeros(B, dtype=torch.bool)
        counts = torch.zeros(B, dtype=torch.long)
        present = torch.zeros((B, C), dtype=torch.bool)
        hidden = m.h0.expand(B, -1).clone()
        linear = (
            torch.zeros((B, m.linear_p.shape[1]), dtype=torch.float64)
            if hasattr(m, "linear_p")
            else None
        )
        for i, cat in enumerate(m.order):
            link = (
                linear @ m.linear_p[i]
                if linear is not None
                else present[:, m.order[:i]].double() @ m.V[i, :i]
            )
            eta = (
                base[:, cat]
                + link
                + (F.relu(hidden) * m.omega[i]).sum(-1)
                + m.psi0[counts]
                + m.psi1[i] * counts
            )
            q = torch.where(
                active, eta.sigmoid(), (F.logsigmoid(empty[:, i]) - ln[:, i]).exp()
            )
            take = torch.rand(B, generator=g) < q.clamp(0, 1)
            present[:, cat] = take
            active |= take
            counts += take
            hidden += take[:, None] * m.W[i]
            if linear is not None:
                linear += take[:, None] * m.linear_q[i]
        assert active.all()
        n = (
            torch.multinomial(x["_q"].exp().reshape(-1, m.R), 1, generator=g).reshape(
                B, C
            )
            + 1
        ) * present
        return esp_draw(x["_u"], n, g)


def hist(x, bins):
    """Convert per-context categorical draws into empirical probability histograms."""
    out = np.zeros((x.shape[0], bins), np.float32)
    np.add.at(
        out,
        (np.repeat(np.arange(len(x)), x.shape[1]), x.ravel().astype(int)),
        1 / x.shape[1],
    )
    return out


def draws_stats(adapter, data, ids, draws, seed):
    """Estimate generation/recommendation summaries on the supplied context indices."""
    out = {k: [] for k in ("inc", "size", "cats", "pairs", "triples", "multiple")}
    for start in range(0, len(ids), 16):
        ctx, _ = data.batch(ids[start : start + 16])
        cached = adapter.cache(ctx)
        repeated = {k: v.repeat_interleave(draws, 0) for k, v in cached.items()}
        raw = (
            adapter.sample(repeated, seed + start * 1009)
            .numpy()
            .reshape(len(ctx["house"]), draws, C, P)
        )
        assert raw.any((2, 3)).all()
        cats = raw.any(-1)
        out["inc"].append(raw.mean(1).reshape(len(raw), J).astype(np.float32))
        out["size"].append(hist(raw.sum((2, 3)), J + 1))
        out["cats"].append(hist(cats.sum(-1), C + 1))
        pair = (
            np.einsum(
                "bdc,bde->bce",
                cats.astype(np.float32),
                cats.astype(np.float32),
                optimize=True,
            )
            / draws
        )
        out["pairs"].append(pair[:, [a for a, b in PAIRS], [b for a, b in PAIRS]])
        out["triples"].append(
            np.stack(
                [
                    (cats[:, :, a] & cats[:, :, b] & cats[:, :, z]).mean(1)
                    for a, b, z in TRIPLES
                ],
                -1,
            ).astype(np.float32)
        )
        out["multiple"].append((raw.sum(-1) > 1).mean((1, 2)).astype(np.float32))
        if start and start % 512 == 0:
            c.status("evaluation_generation", contexts=start, total=len(ids), seed=seed)
    return {k: np.concatenate(v) for k, v in out.items()}


def observed(y):
    """Compute the corresponding summaries of held-out observed binary baskets."""
    y = y.numpy().astype(bool)
    cat = y.any(-1)
    n = y.sum((1, 2))
    return {
        "inc": y.reshape(-1, J).astype(np.float32),
        "size": np.eye(J + 1, dtype=np.float32)[n],
        "cats": np.eye(C + 1, dtype=np.float32)[cat.sum(-1)],
        "pairs": np.stack([cat[:, a] & cat[:, b] for a, b in PAIRS], -1).astype(
            np.float32
        ),
        "triples": np.stack(
            [cat[:, a] & cat[:, b] & cat[:, z] for a, b, z in TRIPLES], -1
        ).astype(np.float32),
        "multiple": (y.sum(-1) > 1).mean(-1),
    }


def features(p, o, draws=384):
    """Build per-context paired metric contributions for clustered uncertainty.

    Correct Brier for Monte Carlo variance. Preserve signed frequency/histogram
    differences until after cluster aggregation, then take absolute errors.
    """
    inc = p["inc"]
    y = o["inc"]
    rank = np.argsort(-inc, axis=-1, kind="stable")
    take = np.take_along_axis(y, rank[:, :10], -1)
    n = y.sum(-1)
    discount = 1 / np.log2(np.arange(2, 7))
    ideal = np.array([discount[: min(int(x), 5)].sum() for x in n])
    re5 = take[:, :5].sum(-1) / n
    re10 = take.sum(-1) / n
    ndcg = (take[:, :5] * discount).sum(-1) / ideal
    brier = ((inc - y) ** 2 - inc * (1 - inc) / (draws - 1)).mean(-1)
    bins = np.minimum((inc * 10).astype(int), 9)
    en = np.zeros((len(y), 10))
    ep = en.copy()
    ey = en.copy()
    rows = np.repeat(np.arange(len(y)), J)
    for a, val in [(en, np.ones(inc.size)), (ep, inc.ravel()), (ey, y.ravel())]:
        np.add.at(a, (rows, bins.ravel()), val)
    return np.concatenate(
        [
            re5[:, None],
            re10[:, None],
            ndcg[:, None],
            brier[:, None],
            inc - y,
            p["size"] - o["size"],
            p["cats"] - o["cats"],
            p["pairs"] - o["pairs"],
            p["triples"] - o["triples"],
            (p["multiple"] - o["multiple"])[:, None],
            ep - ey,
        ],
        -1,
    ).astype(np.float64)


def values(z):
    """Reduce aggregated metric contributions into named scalar evaluation metrics."""
    pos = 0

    def take(n):
        nonlocal pos
        a = z[..., pos : pos + n]
        pos += n
        return a

    a = take(4)
    inc = take(J)
    size = take(J + 1)
    cats = take(C + 1)
    pairs = take(len(PAIRS))
    triples = take(len(TRIPLES))
    multiple = take(1)
    ece = take(10)
    return {
        "recall5": a[..., 0],
        "recall10": a[..., 1],
        "ndcg5": a[..., 2],
        "brier": a[..., 3],
        "product_frequency_mae": abs(inc).mean(-1),
        "size_tv": abs(size).sum(-1) / 2,
        "category_count_tv": abs(cats).sum(-1) / 2,
        "category_pair_mae": abs(pairs).mean(-1),
        "category_triple_mae": abs(triples).mean(-1),
        "multiple_absolute_error": abs(multiple[..., 0]),
        "ece10": abs(ece).sum(-1) / J,
    }


def infer(stats, obs, groups):
    """Compute model metrics and paired cluster-bootstrap intervals, keeping MC replicas separate."""
    unique, idx = np.unique(groups, return_inverse=True)
    H = len(unique)
    count = np.bincount(idx)
    weight = (
        np.random.default_rng(972501)
        .multinomial(H, np.full(H, 1 / H), 1000)
        .astype(float)
    )
    den = weight @ count
    points = {}
    boots = {}
    mc = {}
    for name, s in stats.items():
        pred = {k: s[k].mean(0) for k in obs}
        f = features(pred, obs)
        agg = np.zeros((H, f.shape[1]))
        np.add.at(agg, idx, f)
        points[name] = {k: float(v) for k, v in values(f.mean(0)).items()}
        boots[name] = values((weight @ agg) / den[:, None])
        mean = pred["inc"].sum(-1)
        real = obs["inc"].sum(-1)
        hdelta = np.bincount(idx, weights=mean - real) / count
        points[name]["shopper_mean_size_mae"] = float(abs(hdelta).mean())
        boots[name]["shopper_mean_size_mae"] = (weight @ abs(hdelta)) / H
        points[name]["generated_mean_size"] = float(mean.mean())
        points[name]["observed_mean_size"] = float(real.mean())
        points[name]["multiple_rate"] = float(pred["multiple"].mean())
        mc[name] = [
            {
                k: float(v)
                for k, v in values(
                    features({key: s[key][r] for key in obs}, obs, DRAWS).mean(0)
                ).items()
            }
            for r in range(3)
        ]
    ref = "product" if "product" in stats else "basic"
    paired = {
        name: {
            k: {
                "difference": points[name][k] - points[ref][k],
                "ci95": np.quantile(
                    boots[name][k] - boots[ref][k], [0.025, 0.975]
                ).tolist(),
            }
            for k in boots[name]
        }
        for name in stats
    }
    return {
        "models": points,
        "paired_reference": ref,
        "paired": paired,
        "model_ci95": {
            name: {k: np.quantile(a, [0.025, 0.975]).tolist() for k, a in b.items()}
            for name, b in boots.items()
        },
        "mc_replicas": mc,
        "groups": H,
        "baskets": len(groups),
    }


def ll_infer(ll, groups):
    """Bootstrap paired average basket log-likelihood differences by the supplied clusters."""
    h, idx = np.unique(groups, return_inverse=True)
    H = len(h)
    counts = np.bincount(idx)
    weights = np.random.default_rng(972502).multinomial(H, np.full(H, 1 / H), 1000)
    den = weights @ counts
    result = {}
    samples = {}
    for name, a in ll.items():
        z = np.bincount(idx, weights=a, minlength=H)
        b = (weights @ z) / den
        result[name] = {
            "mean": float(a.mean()),
            "ci95": np.quantile(b, [0.025, 0.975]).tolist(),
        }
        samples[name] = b
    ref = "product" if "product" in ll else "basic"
    for name, a in ll.items():
        result[name]["difference_to_" + ref] = {
            "difference": float((a - ll[ref]).mean()),
            "ci95": np.quantile(samples[name] - samples[ref], [0.025, 0.975]).tolist(),
        }
    return result


def scenarios(models, test):
    """Evaluate identical discount plans under each fitted model.

    Reuse contexts and seeds before/after repricing. Costs and funding are
    explicitly hypothetical; real causal errors, achieved margin and regret
    remain unavailable rather than being inferred from model predictions.
    """
    ids = np.random.default_rng(972601).choice(
        len(test), min(len(test), 32), replace=False
    )
    train = c.Data("train")
    traininc = train.y.sum(0).numpy()
    products = [cat * P + int(traininc[cat].argmax()) for cat in range(C)]
    plans = [[]] + [[j] for j in products]
    r = np.random.default_rng(972602)
    plans += [
        [products[a], products[b]]
        for a, b in [r.choice(C, 2, replace=False) for _ in range(8)]
    ]
    ref = np.load(c.OUT / "data/reference_price.npy")
    limits = train.ctx["dlp"]
    limits = (
        limits.values[np.unique(limits.rows)]
        if isinstance(limits, c.Shared)
        else limits.numpy()
    )
    low = limits.min(0).reshape(-1)
    high = limits.max(0).reshape(-1)
    del train
    rows = {}
    allpred = {}
    B = len(ids)
    for name, m in models.items():
        path = c.OUT / (name + "_scenarios.json")
        if path.exists():
            rows[name] = json.loads(path.read_text())
            continue
        c.status("scenarios_start", model=name)
        a = Adapter(m)
        pred = []
        effects = []
        ctx, _ = test.batch(ids)
        base = []
        for seed in SEEDS:
            cache = a.cache(ctx)
            full = {k: v.repeat_interleave(DRAWS, 0) for k, v in cache.items()}
            base.append(a.sample(full, seed).reshape(B, DRAWS, C, P).numpy())
        base = np.concatenate(base, axis=1)
        baseline = base.mean(1).reshape(B, J)
        pred.append(baseline)
        for pi, plan in enumerate(plans[1:], 1):
            changed = {**ctx, "dlp": ctx["dlp"].clone()}
            for j in plan:
                changed["dlp"][:, j // P, j % P] += np.log(0.9)
            raw = []
            for seed in SEEDS:
                cache = a.cache(changed)
                full = {k: v.repeat_interleave(DRAWS, 0) for k, v in cache.items()}
                raw.append(a.sample(full, seed).reshape(B, DRAWS, C, P).numpy())
            raw = np.concatenate(raw, 1)
            delta = raw.astype(float) - base
            effect = delta.mean((0, 1)).reshape(-1)
            pred.append(raw.mean(1).reshape(B, J))
            owned = set(plan)
            sibling = {
                k for j in plan for k in range(j // P * P, (j // P + 1) * P)
            } - owned
            cross = set(range(J)) - owned - sibling
            mean_context_delta = delta.sum((2, 3)).mean(0)
            effects.append(
                {
                    "plan": pi,
                    "products": plan,
                    "own_incidence_change": float(effect[list(owned)].sum()),
                    "sibling_incidence_change": float(effect[list(sibling)].sum()),
                    "cross_incidence_change": float(effect[list(cross)].sum()),
                    "size_change": float(effect.sum()),
                    "size_mc_se": float(
                        mean_context_delta.std(ddof=1)
                        / np.sqrt(len(mean_context_delta))
                    ),
                    "category_participation_changes": (
                        raw.any(-1).mean((0, 1)) - base.any(-1).mean((0, 1))
                    ).tolist(),
                    "outside_training_proxy_price_fraction": float(
                        (
                            (changed["dlp"].reshape(B, J).numpy()[:, plan] < low[plan])
                            | (
                                changed["dlp"].reshape(B, J).numpy()[:, plan]
                                > high[plan]
                            )
                        ).mean()
                    ),
                    "actual_response_error": None,
                }
            )
            if pi % 16 == 0:
                c.status("scenario_progress", model=name, plans=pi, total=len(plans))
        pred = np.array(pred)
        decisions = []
        for cost in (0.3, 0.6, 0.85):
            for funding in (0.0, 0.1):
                margins = []
                for pi, plan in enumerate(plans):
                    price = ref.copy()
                    price[plan] *= 0.9
                    fund = funding * (ref - price)
                    margin = price - cost * ref + fund
                    margins.append(float((pred[pi] * margin).sum(-1).mean()))
                best = int(np.argmax(margins))
                decisions.append(
                    {
                        "cost_fraction": cost,
                        "funding_fraction": funding,
                        "selected_plan": best,
                        "products": plans[best],
                        "predicted_margin": margins[best],
                        "predicted_incremental_margin": margins[best] - margins[0],
                        "actual_achieved_margin": None,
                        "real_regret": None,
                    }
                )
        rows[name] = {
            "contexts": ids.tolist(),
            "draws_per_context_plan": 384,
            "effects": effects,
            "hypothetical_decisions": decisions,
        }
        c.write(name + "_scenarios.json", rows[name])
    result = {
        "plans": plans,
        "models": rows,
        "scope": "Predictions with lagged price proxies and hypothetical costs per incidence; actual causal accuracy and retailer profit/regret unavailable.",
    }
    c.write("scenarios.json", result)
    return result


def report(result):
    """Render the common comparison results as one standalone HTML report."""
    from .report import render

    render(result)


def main():
    """Score frozen selected checkpoints across predictive, generation, scenario and runtime panels."""
    test = c.Data("test")
    ids = np.sort(
        np.random.default_rng(972601).choice(
            len(test), min(len(test), 4096), replace=False
        )
    )
    c.write(
        "evaluation_contexts.json",
        {
            "indices": ids.tolist(),
            "triples": TRIPLES,
            "seeds": SEEDS,
            "draws_per_seed": DRAWS,
        },
    )
    from .provenance import freeze

    freeze(
        c.OUT, "evaluation_manifest.json", include_data=True, include_checkpoints=True
    )
    names = [
        p.name
        for p in (c.OUT / "selected").iterdir()
        if p.is_dir() and p.name != "product_fallback"
    ]
    models = {n: c.load_model(n) for n in names}
    stats = {}
    lls = {}
    timings = {}
    for name, m in models.items():
        c.status("evaluate_start", model=name)
        path = c.OUT / (name + "_evaluation.npz")
        t = time.monotonic()
        if path.exists():
            s = dict(np.load(path))
            timings[name] = json.loads(
                (c.OUT / (name + "_evaluation_time.json")).read_text()
            )
        else:
            with torch.no_grad():
                ll = torch.cat(
                    [
                        m.log_probability(*test.batch(slice(i, i + 256)))
                        for i in range(0, len(test), 256)
                    ]
                ).numpy()
            a = Adapter(m)
            pilot_start = time.monotonic()
            pilot = draws_stats(a, test, ids[:16], 128, SEEDS[0])
            pilotseconds = time.monotonic() - pilot_start
            c.write(
                name + "_generation_pilot.json",
                {
                    "seconds": pilotseconds,
                    "estimated4096context384draw_seconds": pilotseconds * 4096 / 16 * 3,
                },
            )
            replicas = [draws_stats(a, test, ids, DRAWS, seed) for seed in SEEDS]
            s = {k: np.stack([r[k] for r in replicas]) for k in pilot}
            s["ll"] = ll
            np.savez_compressed(path, **s)
            timings[name] = {
                "seconds": time.monotonic() - t,
                "contexts": len(ids),
                "draws": 384,
            }
            c.write(name + "_evaluation_time.json", timings[name])
        stats[name] = {k: v for k, v in s.items() if k != "ll"}
        lls[name] = s["ll"]
    obs = observed(test.y[ids])
    result = infer(stats, obs, test.raw["raw_household"][ids])
    sensitivity = infer(
        stats, obs, test.raw["raw_store"][ids] * 1000 + test.raw["week"][ids]
    )
    result["full_likelihood"] = ll_infer(lls, test.raw["raw_household"])
    sensitivity["full_likelihood"] = ll_infer(
        lls, test.raw["raw_store"] * 1000 + test.raw["week"]
    )
    c.write("store_period_sensitivity.json", sensitivity)
    for n in result["models"]:
        result["models"][n]["basket_ll"] = result["full_likelihood"][n]["mean"]
    result["timings"] = timings
    result["uncertainty_scope"] = (
        "1000paired HH bootstrap; separate store-period sensitivity,3MC replicas; conditional on parameters, no retraining/two-way uncertainty."
    )
    c.write("comparison.json", result)
    protocol = json.loads((c.OUT / "protocol.json").read_text())
    if protocol.get("dataset") == "synthetic":
        from .synthetic.evaluate import main as oracle_evaluate

        oracle_evaluate(models, test)
    scenarios(models, test)
    # Separate fresh processes provide per-model application/RSS checks.
    import subprocess

    practical = {}
    for name in models:
        p = subprocess.run(
            [sys.executable, "-m", "experiments.practical", name],
            cwd=str(c.ROOT),
            capture_output=True,
            text=True,
        )
        if p.returncode:
            c.write(
                name + "_practical_failure.json",
                {"stderr": p.stderr, "stdout": p.stdout},
            )
            raise RuntimeError("Practical worker failed " + name)
        practical[name] = json.loads((c.OUT / (name + "_practical.json")).read_text())
    c.write("practical.json", practical)
    report(result)
    c.write(
        "result.json",
        {
            "state": "complete",
            "models": list(models),
            "report": str(c.OUT / "comparisons.html"),
            "fully_fitted_catalogue": J,
            "categories": C,
            "all_five_panels": True,
            "real_causal_accuracy_and_regret": None,
        },
    )
    c.status("complete", jobs_running=False)


if __name__ == "__main__":
    main()
