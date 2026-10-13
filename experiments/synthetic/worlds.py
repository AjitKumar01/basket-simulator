"""Known-truth worlds and exact metrics from the current synthetic study."""

import itertools
import numpy as np
from scipy.special import softmax, logsumexp
from . import helpers as old

J = 12
C = 3
P = 4
BITS = ((np.arange(1 << J)[:, None] >> np.arange(J)) & 1).astype(float)
SETS = BITS[1:]
SIZE = SETS.sum(1).astype(int)
CAT = (SETS.reshape(-1, C, P).sum(-1) > 0).astype(float)
LOCAL = ((np.arange(16)[:, None] >> np.arange(P)) & 1).astype(float)
LOCALSIZE = LOCAL.sum(1).astype(int)
CODE = (SETS.reshape(-1, C, P) @ 2 ** np.arange(P)).astype(int)
PATS = np.array(list(itertools.product((0.0, 1.0), repeat=C)))
CATCODE = (CAT @ np.array([4, 2, 1])).astype(int)


def normalized(p):
    """Normalize nonnegative set weights into a finite probability distribution."""
    assert np.isfinite(p).all() and (p >= 0).all()
    assert np.max(abs(p.sum(-1) - 1)) < 2e-9, np.max(abs(p.sum(-1) - 1))
    return p


class World:
    """An independent known-truth world for robustness and intervention diagnostics.

    Enumerate all nonempty 12-product sets. Mission worlds create latent shopping
    needs; the Ising world creates product complements. These oracle laws never
    enter model fitting or validation selection.
    """

    def __init__(self, row):
        self.row = row
        self.seed = row["seed"]
        self.family = row["family"]
        r = np.random.default_rng(self.seed)
        self.w = {
            "taste": r.normal(0, 0.5, (16, C, P)),
            "quality": r.normal(0, 0.4, (C, P)),
            "beta": r.uniform(0.8, 1.8, (C, P)),
            "house_need": r.normal(0, 0.25, (16, C)),
            "missions": np.array([[0.7, -0.4, 0.3], [-0.5, 0.6, -0.2]]),
            "mission_taste": r.normal(0, 0.15, (2, C, P)),
        }
        if self.family == "product_ising":
            self.taste = r.normal(0, 0.45, (16, J))
            self.quality = r.normal(-1.8, 0.25, J)
            self.beta = r.uniform(1.0, 2.5, J)
            self.pair = np.zeros((J, J))
            for c in range(C):
                for a, b in itertools.combinations(range(c * P, (c + 1) * P), 2):
                    self.pair[a, b] = -0.7
            for a, b in ((0, 4), (1, 8), (5, 9), (3, 11)):
                self.pair[a, b] = 1.5
            self.energy = (SETS @ self.pair * SETS).sum(1) - 0.08 * (SIZE - 1) ** 2

    def probability(self, house, price):
        """Return exact set probabilities for one shopper and offered log-price vector."""
        if self.family == "product_ising":
            d = price.reshape(-1, J)
            u = (
                self.quality
                + self.taste[house - 1]
                - self.beta * d
                - 0.15 * np.maximum(d, 0)
            )
            return normalized(softmax(u @ SETS.T + self.energy, axis=-1))
        if "zero_price" in self.family:
            price = np.zeros_like(price)
        out = np.zeros((len(house), len(SETS)))
        for m, weight in enumerate((0.55, 0.45)):
            pats, mass, count, u = old.world_parts(self.w, house - 1, price, m)
            within = np.zeros((len(house), C, 16))
            within[:, :, 0] = 1
            for code in range(1, 16):
                items = np.flatnonzero(LOCAL[code])
                n = len(items)
                if n > 3:
                    continue
                prob = np.zeros((len(house), C))
                for order in itertools.permutations(items):
                    logits = u.copy()
                    q = np.ones((len(house), C))
                    for j in order:
                        q *= softmax(logits, axis=-1)[..., j]
                        logits[..., j] = -np.inf
                    prob += q
                within[:, :, code] = count[:, :, n - 1] * prob
            joint = mass[:, CATCODE]
            for c in range(C):
                joint = joint * within[:, c, CODE[:, c]]
            out += weight * joint
        return normalized(out)

    def sample(self, N, seed):
        """Draw observed covariates and target baskets independently from the model under test."""
        if self.family != "product_ising":
            if "zero_price" not in self.family:
                return old.synthetic_sample(self.w, N, seed)
            # Random offers remain observed, but all latent behavior sees price zero.
            r = np.random.default_rng(seed)
            h = r.integers(0, 16, N)
            price = r.uniform(-0.35, 0.35, (N, C, P))
            y = np.zeros_like(price)
            mission = r.choice(2, N, p=[0.55, 0.45])
            for m in range(2):
                rows = np.flatnonzero(mission == m)
                pats, mass, count, u = old.world_parts(
                    self.w, h[rows], np.zeros_like(price[rows]), m
                )
                chosen = (
                    (r.random((len(rows), 1)) > mass.cumsum(-1)).sum(-1).clip(max=7)
                )
                present = pats[chosen]
                size = 1 + (r.random((len(rows), C, 1)) > count.cumsum(-1)).sum(
                    -1
                ).clip(max=2)
                order = np.argsort(-(u + r.gumbel(size=u.shape)), axis=-1)
                for c in range(C):
                    for k in range(3):
                        take = (present[:, c] > 0) & (size[:, c] > k)
                        y[rows[take], c, order[take, c, k]] = 1
            return {
                "house": h + 1,
                "store": np.zeros(N, int),
                "week": r.integers(1, 53, N),
                "dlp": price,
                "disp": np.zeros_like(price),
                "mail": np.zeros_like(price),
                "member": y,
            }
        r = np.random.default_rng(seed)
        h = r.integers(1, 17, N)
        price = r.uniform(-0.35, 0.35, (N, C, P))
        y = []
        for k in range(0, N, 256):
            q = self.probability(h[k : k + 256], price[k : k + 256])
            cdf = q.cumsum(-1)
            cdf[:, -1] = 1
            code = (r.random((len(q), 1)) > cdf).sum(-1)
            y.append(SETS[code].reshape(-1, C, P))
        return {
            "house": h,
            "store": np.zeros(N, int),
            "week": r.integers(1, 53, N),
            "dlp": price,
            "disp": np.zeros_like(price),
            "mail": np.zeros_like(price),
            "member": np.concatenate(y),
        }


def moments(p):
    """Compute exact product, category and size moments from enumerated set probabilities."""
    return {
        "incidence": (p @ SETS).reshape(-1, C, P),
        "presence": p @ CAT,
        "basket_size": p @ SIZE,
    }


def actions():
    """Construct the common no-discount, single-product and paired discount plans."""
    out = [{"name": "no_promotion", "shift": np.zeros(J), "items": [], "kind": "none"}]
    for j in range(J):
        for frac in (0.05, 0.1, 0.2):
            d = np.zeros(J)
            d[j] = np.log(1 - frac)
            out.append(
                {
                    "name": f"discount_{j}_{frac}",
                    "shift": d,
                    "items": [j],
                    "kind": "single",
                }
            )
        d = np.zeros(J)
        d[j] = np.log(1.1)
        out.append(
            {"name": f"increase_{j}_0.1", "shift": d, "items": [j], "kind": "increase"}
        )
    for a, b in ((0, 4), (1, 5), (2, 8), (3, 9), (6, 10), (7, 11)):
        d = np.zeros(J)
        d[[a, b]] = np.log(0.9)
        out.append(
            {"name": f"pair_{a}_{b}", "shift": d, "items": [a, b], "kind": "paired"}
        )
    return out


def response_score(pred, truth, acts):
    """Compare estimated own, sibling and cross-category responses with known oracle truth."""
    result = {}
    for kind in ("single", "increase", "paired"):
        ids = [i for i, a in enumerate(acts) if a["kind"] == kind]
        row = {}
        for key in ("incidence", "presence", "basket_size"):
            t = np.stack([truth[i][key] - truth[0][key] for i in ids])
            p = np.stack([pred[i][key] - pred[0][key] for i in ids])
            mae = float(abs(p - t).mean())
            zero = float(abs(t).mean())
            den = float((t * t).sum())
            row[key] = {
                "mae": mae,
                "zero_effect_mae": zero,
                "relative_mae": mae / zero if zero > 1e-12 else None,
                "slope": float((p * t).sum() / den) if den > 1e-18 else None,
                "false_effect_fraction_above_0.005": (
                    float((abs(p) > 0.005).mean()) if zero < 1e-12 else None
                ),
            }
            row[key]["passed"] = bool(
                mae <= 0.005
                if zero < 1e-12
                else mae / zero <= 0.25 and 0.75 <= row[key]["slope"] <= 1.25
            )
        result[kind] = row
    for group in ("own", "same_category", "cross_category"):
        ps = []
        ts = []
        for i, a in enumerate(acts):
            if a["kind"] != "single":
                continue
            j = a["items"][0]
            idx = (
                [j]
                if group == "own"
                else [
                    k
                    for k in range(J)
                    if k != j
                    and (
                        k // P == j // P
                        if group == "same_category"
                        else k // P != j // P
                    )
                ]
            )
            ps.append(
                (pred[i]["incidence"] - pred[0]["incidence"])
                .reshape(-1, J)[:, idx]
                .ravel()
            )
            ts.append(
                (truth[i]["incidence"] - truth[0]["incidence"])
                .reshape(-1, J)[:, idx]
                .ravel()
            )
        p = np.concatenate(ps)
        t = np.concatenate(ts)
        z = float(abs(t).mean())
        den = float(t @ t)
        result[group] = {
            "mae": float(abs(p - t).mean()),
            "relative_mae": float(abs(p - t).mean() / z) if z > 1e-12 else None,
            "slope": float(p @ t / den) if den > 1e-18 else None,
        }
    names = {a["name"]: i for i, a in enumerate(acts)}
    p = []
    t = []
    for i, a in enumerate(acts):
        if a["kind"] != "paired":
            continue
        x, y = a["items"]
        ia = names[f"discount_{x}_0.1"]
        ib = names[f"discount_{y}_0.1"]
        p.append(
            pred[i]["incidence"]
            - pred[ia]["incidence"]
            - pred[ib]["incidence"]
            + pred[0]["incidence"]
        )
        t.append(
            truth[i]["incidence"]
            - truth[ia]["incidence"]
            - truth[ib]["incidence"]
            + truth[0]["incidence"]
        )
    p = np.stack(p)
    t = np.stack(t)
    result["paired_nonadditivity"] = {
        "mae": float(abs(p - t).mean()),
        "oracle_mean_absolute_signal": float(abs(t).mean()),
    }
    return result
