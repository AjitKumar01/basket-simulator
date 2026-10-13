"""Exact large-catalogue fitting adapters and lazy shared contexts."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import copy
import hashlib
import json
import sys
import time
import types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("BASKET_RUN_DIR", ROOT / "runs/benchmark")).resolve()
from basket_simulator.trainable import TrainableBasketModel, log_esp
from basket_simulator.product_interactions import ProductInteractionModel
from basket_simulator.pipeline import independent_start
from basket_simulator.train import fit, subset, gradient_norm
from . import baselines as historical_models
from .baselines import MissionESP, BinaryAR

historical_models.log_esp = log_esp
SEED = 972301
torch.set_num_threads(4)


def digest(p):
    """Hash a file in bounded chunks for input and checkpoint provenance."""
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write(name, x):
    """Atomically replace a run JSON file so readers never see a partial record."""
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def status(phase, **kw):
    """Persist and print the current worker phase for the read-only monitor."""
    row = {"phase": phase, "pid": os.getpid(), "time": time.time(), **kw}
    write("status.json", row)
    print(json.dumps(row), flush=True)


class Shared:
    """Index shared memory-mapped context rows as if they were per-basket tensors.

    rows maps each basket to a store/week context. Materialize only the requested
    batch, keeping the large-catalogue dataset out of dense in-memory tensors.
    """

    def __init__(self, rows, values, dtype=torch.float64):
        self.rows = rows
        self.values = values
        self.dtype = dtype

    def __getitem__(self, idx):
        return torch.tensor(
            np.array(self.values[self.rows[idx]], copy=True), dtype=self.dtype
        )


class Data:
    """Load and validate a prepared split in either dense or shared-context layout.

    ctx holds observable covariates; y holds binary basket targets. Unknown
    shopper/store IDs use row 0. Reject unsupported baskets rather than silently
    dropping or padding them.
    """

    def __init__(self, name):
        cfg = json.loads((OUT / "data/config.json").read_text())
        with np.load(OUT / "data" / f"{name}.npz", allow_pickle=False) as archive:
            d = {k: archive[k] for k in archive.files}
        self.raw = d
        C, P = cfg["C"], cfg["P"]
        if "packed_member" in d:
            member = np.unpackbits(d["packed_member"], axis=1)[:, : C * P].reshape(
                -1, C, P
            )
        else:
            member = d["member"]
        if (
            member.shape != (len(d["house"]), C, P)
            or not np.isin(member, (0, 1)).all()
            or not member.any((1, 2)).all()
        ):
            raise ValueError(
                "Require nonempty binary baskets with configured dimensions"
            )
        if (member.sum(-1) > cfg["R"]).any():
            raise ValueError(
                "Basket exceeds category support; do not silently discard it"
            )
        self.y = torch.from_numpy(member.astype(np.uint8))
        self.ctx = {
            k: torch.tensor(
                d[k], dtype=torch.long if k in ("house", "store") else torch.float64
            )
            for k in ("house", "store", "week")
        }
        if "context" in d:
            self.ctx.update(
                {
                    k: Shared(
                        d["context"], np.load(OUT / "data" / f"{k}.npy", mmap_mode="r")
                    )
                    for k in ("dlp", "disp", "mail")
                }
            )
        else:
            self.ctx.update(
                {
                    k: torch.tensor(d[k], dtype=torch.float64)
                    for k in ("dlp", "disp", "mail")
                }
            )
        for field, size in [("house", cfg["users"]), ("store", cfg["stores"])]:
            values = np.asarray(d[field])
            if (
                not np.isfinite(values).all()
                or not np.equal(values, values.astype(np.int64)).all()
                or values.min() < 0
                or values.max() >= size
            ):
                raise ValueError(f"{field}: IDs must be within training mapping")
        for field in ("week", "dlp", "disp", "mail"):
            values = d[field] if field in d else self.ctx[field].values
            expected = (
                (len(self.y),)
                if field == "week"
                else ((len(self.y), C, P) if field in d else (len(values), C, P))
            )
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError(
                    f"{field}: finite context with shape {expected} required"
                )
        if "raw_household" not in d:
            self.raw["raw_household"] = d["house"]
        if "raw_store" not in d:
            self.raw["raw_store"] = d["store"]

    def pair(self):
        """Return the context/target pair expected by the optimizer."""
        return self.ctx, self.y

    def batch(self, idx):
        """Materialize just the selected covariate and target rows."""
        return subset(self.ctx, idx), self.y[idx]

    def __len__(self):
        return len(self.y)


class Independent(TrainableBasketModel):
    def log_probability(self, ctx, y):
        """Score conditionally independent Bernoulli products given a nonempty basket."""
        u = self.baseline(ctx) - self.choice_price.reshape(self.C, self.P) * ctx["dlp"]
        a = F.softplus(u).sum((1, 2))
        return (u * y).sum((1, 2)) - a - torch.log(-torch.expm1(-a))

    @torch.no_grad()
    def sample(self, ctx, seed=0):
        """Draw the same nonempty independent law by conditioning the first success."""
        u = (
            self.baseline(ctx) - self.choice_price.reshape(self.C, self.P) * ctx["dlp"]
        ).reshape(-1, self.C * self.P)
        # First successful product conditioning; polynomial, no rejection.
        no = F.logsigmoid(-u)
        suffix = F.softplus(u.flip(-1)).cumsum(-1).flip(-1)
        log_nonempty = torch.log(-torch.expm1(-suffix))
        active = torch.zeros(len(u), dtype=torch.bool)
        y = torch.zeros_like(u, dtype=torch.bool)
        g = torch.Generator().manual_seed(seed)
        for j in range(u.shape[1]):
            q = torch.where(
                active,
                u[:, j].sigmoid(),
                (F.logsigmoid(u[:, j]) - log_nonempty[:, j]).exp(),
            )
            take = torch.rand(len(u), generator=g) < q.clamp(0, 1)
            y[:, j] = take
            active |= take
        assert active.all()
        return y.reshape(-1, self.C, self.P)


class CachedStructure(TrainableBasketModel):
    """Reuse frozen preference ESP terms while fitting structural parameters."""

    def stage_terms(self, ctx):
        """Use cached actual/reference offers when present; otherwise compute them."""
        if "_actual" not in ctx:
            return super().stage_terms(ctx)
        u, actual, reference = ctx["_u"], ctx["_actual"], ctx["_reference"]
        n = torch.arange(1, self.R + 1, dtype=torch.float64)
        penalty = self.rho[:, None] * (n * (n - 1) / 2) - getattr(
            self, "count_offset", torch.zeros_like(self.rho)
        )[:, None] * (n - 1)
        a0 = reference[..., 1:] - penalty
        a = a0 + self.count_price_scale[None, :, None] * (
            actual[..., 1:] - reference[..., 1:]
        )
        iv0 = torch.logsumexp(a0, -1)
        iv = (
            iv0
            + self.inclusive_price_weight
            * (torch.logsumexp(actual[..., 1:] - penalty, -1) - iv0)
            - (self.category_price * ctx["dlp"]).sum(-1)
        )
        cross = (iv @ self.cross_q) @ self.cross_p.T - iv * (
            self.cross_p * self.cross_q
        ).sum(-1)
        base = (
            self.kappa
            + F.softplus(self.tau) * iv
            + cross
            + self.theta[ctx["house"]] @ self.U.T
        )
        return u, actual, F.log_softmax(a, -1), base


class SizeESP(nn.Module):
    """Full-support count law; ESP degrees limited to needed n, not support truncated."""

    def __init__(self, parent, member):
        super().__init__()
        self.parent = copy.deepcopy(parent)
        for p in self.parent.parameters():
            p.requires_grad_(False)
        self.C, self.P, self.R = parent.C, parent.P, parent.C * parent.P
        self.J = self.R
        self.config = {
            **parent.config,
            "benchmark_baseline": "size_ESP_scalar_price_count",
        }
        counts = (
            np.bincount(member.sum((1, 2)).numpy().astype(int), minlength=self.J + 1)[
                1:
            ]
            + 0.5
        )
        self.lam = nn.Parameter(torch.tensor(np.log(counts / counts.sum())))
        self.need = nn.Parameter(
            torch.zeros(parent.config["users"], dtype=torch.float64)
        )
        self.price = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def terms(self, ctx):
        """Return product utilities and a full-support total-size distribution."""
        u = (
            self.parent.baseline(ctx)
            - self.parent.choice_price.reshape(self.C, self.P) * ctx["dlp"]
        ).reshape(-1, self.J)
        n = torch.arange(self.J, dtype=torch.float64)
        q = F.log_softmax(
            self.lam
            + (self.need[ctx["house"]] + self.price * ctx["dlp"].mean((1, 2)))[:, None]
            * n,
            -1,
        )
        return u, q

    def log_probability(self, ctx, y):
        """Score predicted total size times its normalized weighted subset probability."""
        u, q = self.terms(ctx)
        n = y.sum((1, 2)).long()
        if "_size_within" in ctx:
            within = ctx["_size_within"]
        else:
            e = log_esp(u, int(n.max()))
            within = (u * y.reshape(-1, self.J)).sum(-1) - e.gather(
                1, n[:, None]
            ).ravel()
        return within + q.gather(1, (n - 1)[:, None]).ravel()

    def parameter_bounds(self):
        return [
            (-15.0, 15.0)
            for p in self.parameters()
            if p.requires_grad
            for _ in range(p.numel())
        ]


class Neural(BinaryAR):
    """The benchmark's conditional neural autoregressive baseline.

    For product j, a 16-unit sigmoid hidden state sums embeddings of selected
    products before j and the current log-price vector. Its output adjusts the
    frozen parent's observable preferences plus trainable shopper/product and
    own-price terms. A rank-8 cross-price map adds direct substitution channels.
    The catalogue order is fixed; the inherited likelihood and sampler condition
    on nonempty baskets. This is a NADE-style comparator, not native SHOPPER.
    """

    def __init__(self, parent, seed=SEED):
        super().__init__(parent, True, seed)
        # Factor the direct cross-price map to avoid retaining J-by-J parameters.
        del self.cross_price
        self.cross_left = nn.Parameter(torch.zeros((self.J, 8), dtype=torch.float64))
        self.cross_right = nn.Parameter(
            0.01
            * torch.randn(
                (self.J, 8),
                generator=torch.Generator().manual_seed(seed),
                dtype=torch.float64,
            )
        )
        self.config["benchmark_baseline"] = "conditional_NADE_H16_crossprice_rank8"

    def logits(self, ctx, y):
        """Return a (batch, J) table of logits with no current/future target leakage."""
        log_prices = ctx["dlp"].reshape(-1, self.J)
        logits = (
            self.parent.baseline(ctx).reshape(-1, self.J)
            + self.lam
            + self.house[ctx["house"]]
            - self.price * log_prices
        )
        # Remove the diagonal: the explicit own-price coefficient handles it.
        logits = (
            logits
            + (log_prices @ self.cross_right) @ self.cross_left.T
            - log_prices * (self.cross_left * self.cross_right).sum(-1)
        )
        product_embeddings = y[..., None] * self.E
        # Exclude the current item to prevent leaking its target into its logit.
        earlier_embeddings = product_embeddings.cumsum(1) - product_embeddings
        return logits + (
            torch.sigmoid(earlier_embeddings + (log_prices @ self.Q)[:, None]) * self.O
        ).sum(-1)


def save_model(model, path):
    """Write checkpoint parameters and the configuration needed to reconstruct them."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "best.pt")
    (path / "config.json").write_text(json.dumps(model.config, indent=2) + "\n")


def load_model(name):
    """Reconstruct a named selected baseline or model and load its saved parameters."""
    config = json.loads((OUT / "selected" / name / "config.json").read_text())
    state = torch.load(OUT / "selected" / name / "best.pt", weights_only=True)
    if name == "product":
        model = ProductInteractionModel(
            TrainableBasketModel(config["parent_config"]),
            config["product_rank"],
            config["product_seed"],
            config["product_orders"],
        )
    elif name == "mission":
        model = MissionESP(Independent(config), SEED)
    elif name == "neural":
        model = Neural(Independent(config), SEED)
    elif name == "size_aware":
        model = SizeESP(
            Independent(config),
            torch.ones((1, config["C"], config["P"]), dtype=torch.uint8),
        )
    elif name in ("popularity", "independent", "no_price"):
        model = Independent(config)
    else:
        model = CachedStructure(config)
    model.load_state_dict(state)
    return model


def materialize_cache(model, data, name, kind):
    """Persist frozen parent terms on disk for structure or product-head fitting."""
    dest = OUT / "cache" / name
    dest.mkdir(parents=True, exist_ok=True)
    N = len(data)
    C, P, R = model.C, model.P, model.R
    if kind == "structure":
        shapes = {
            "_u": (N, C, P),
            "_actual": (N, C, R + 1),
            "_reference": (N, C, R + 1),
        }
    elif kind == "product":
        shapes = {
            "_u": (N, C, P),
            "_esp": (N, C, R + 1),
            "_count": (N, C, R),
            "_base": (N, C),
            "_future": (N, C, P),
            "_presence": (N, C),
        }
    arrays = {
        k: np.lib.format.open_memmap(
            dest / (k + ".npy"), mode="w+", dtype="float64", shape=s
        )
        for k, s in shapes.items()
    }
    for start in range(0, N, 512):
        ctx, _ = data.batch(slice(start, start + 512))
        if kind == "structure":
            with torch.no_grad():
                b = model.baseline(ctx)
                u = b - model.choice_price.reshape(C, P) * ctx["dlp"]
                terms = {"_u": u, "_actual": log_esp(u, R), "_reference": log_esp(b, R)}
        else:
            terms = model.cache(ctx)
        for k, a in arrays.items():
            a[start : start + 512] = terms[k].detach().numpy()
        if start % 8192 == 0:
            status("cache", model=name, rows=start, total=N)
    for a in arrays.values():
        a.flush()
    ctx = {
        **data.ctx,
        **{
            k: Shared(np.arange(N), np.load(dest / (k + ".npy"), mmap_mode="r"))
            for k in shapes
        },
    }
    return ctx, data.y


def gradient_diagnosis(model, train, ridge):
    """Recompute final objective gradients and rank active parameter blocks by residual."""
    model.zero_grad(set_to_none=True)
    ctx, y = train
    N = len(y)
    for i in range(0, N, 512):
        (
            -model.log_probability(subset(ctx, slice(i, i + 512)), y[i : i + 512]).sum()
            / N
        ).backward()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    pen = [
        p.square().sum()
        for n, p in params
        if n not in ("lam", "kappa", "rho", "count_offset")
    ]
    if pen:
        (ridge * torch.stack(pen).sum() / (2 * N)).backward()
    bounds = model.parameter_bounds()
    rows = []
    offset = 0
    for n, p in params:
        end = offset + p.numel()
        v = p.detach().numpy().ravel()
        g = p.grad.detach().numpy().ravel()
        rows.append(
            {"block": n, "projected_gradient": gradient_norm(v, g, bounds[offset:end])}
        )
        offset = end
    return sorted(rows, key=lambda x: x["projected_gradient"], reverse=True)


def run_fit(model, name, train, validation, ridge=10):
    """Fit one stage, diagnose failure, and permit at most one improving continuation.

    Preserve the original failure and compare checkpoints by validation only.
    Final-iterate stationarity and the selected checkpoint's stationarity are
    separate recorded facts. Existing unresolved stages are never overwritten.
    """
    path = OUT / "fits" / name
    if (path / "selection_record.json").exists():
        row = json.loads((path / "selection_record.json").read_text())
        model.load_state_dict(
            torch.load(row["predictive_checkpoint"], weights_only=True)
        )
        return model, row
    status("fit_start", model=name, ridge=ridge)
    # A stopped run is never silently restarted or overwritten.
    if (path / "fit.json").exists():
        raise RuntimeError(
            "Unresolved previous fit: inspect before resume " + str(path)
        )
    row = fit(
        model,
        train,
        validation,
        path,
        maxiter=350,
        maxcalls=500,
        time_limit=1800,
        batch_size=512,
        ridge=ridge,
    )
    row["predictive_checkpoint"] = str(path / "best.pt")
    row["ridge"] = ridge
    row["total_seconds"] = row["seconds"]
    if not row["final_stationarity_passed"]:
        beststate = copy.deepcopy(model.state_dict())
        model.load_state_dict(torch.load(path / "final.pt", weights_only=True))
        diag = gradient_diagnosis(model, train, ridge)
        write(
            str(path.relative_to(OUT) / "diagnosis.json"),
            {
                "blocks": diag,
                "objective_progress": row["initial_objective"] - row["final_objective"],
                "first_failure_retained": row,
                "action": "One same-objective final-state continuation only if finite and improving.",
            },
        )
        if (
            row["final_objective"] < row["initial_objective"]
            and np.isfinite([r["projected_gradient"] for r in diag]).all()
        ):
            later = fit(
                model,
                train,
                validation,
                path / "continuation",
                maxiter=350,
                maxcalls=500,
                time_limit=1800,
                batch_size=512,
                ridge=ridge,
            )
            later["initial_attempt"] = row
            later["ridge"] = ridge
            later["total_seconds"] = row["seconds"] + later["seconds"]
            later["predictive_checkpoint"] = str(path / "continuation/best.pt")
            if row["best_validation_ll"] >= later["best_validation_ll"]:
                model.load_state_dict(beststate)
                for k in (
                    "best_validation_ll",
                    "best_iteration",
                    "best_objective",
                    "best_projected_gradient",
                    "best_stationarity_passed",
                    "best_scaled_projected_gradient",
                ):
                    later[k] = row.get(k)
                later["predictive_checkpoint"] = str(path / "best.pt")
            row = later
        else:
            model.load_state_dict(beststate)
    write(str(path.relative_to(OUT) / "selection_record.json"), row)
    status("fit_complete", model=name, **row)
    return model, row
