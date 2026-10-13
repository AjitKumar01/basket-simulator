"""Native SHOPPER utility adapter and exact unordered-set scoring. No vendor edits."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import hashlib
import json
import itertools
from pathlib import Path
import numpy as np
import tensorflow as tf
from scipy.special import logsumexp
from choice_learn.basket_models import Shopper

OUT = Path(os.environ["BASKET_RUN_DIR"]) / "shopper_native"


def weights(model):
    return {w.name.split(":")[0]: w.numpy() for w in model.trainable_weights}


def epoch_rows(d, rng):
    n = len(d["house"])
    width = int(np.diff(d["offset"]).max() + 1)
    total = len(d["item"]) + n
    targets = np.empty(total, np.int32)
    past = np.full((total, width), -1, np.int32)
    future = past.copy()
    ids = np.empty(total, np.int32)
    p = 0
    for i in rng.permutation(n):
        basket = np.r_[
            rng.permutation(d["item"][d["offset"][i] : d["offset"][i + 1]]), 0
        ]
        for t, j in enumerate(basket):
            targets[p] = j
            past[p, :t] = basket[:t]
            future[p, : len(basket) - t - 1] = basket[t + 1 :]
            ids[p] = i
            p += 1
    order = rng.permutation(total)
    return targets[order], past[order], future[order], ids[order]


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ObservedShopper(Shopper):
    def add_observables(self, store, calendar, display, advert, nstores):
        self.obs_store = tf.constant(store, tf.int32)
        self.obs_calendar = tf.constant(calendar, tf.float32)
        self.obs_display = tf.constant(display, tf.float32)
        self.obs_advert = tf.constant(advert, tf.float32)
        self.store_control = tf.Variable(
            tf.zeros((nstores, self.n_items)), name="store_control"
        )
        self.calendar_control = tf.Variable(
            tf.zeros((self.n_items, 4)), name="calendar_control"
        )
        self.display_control = tf.Variable(
            tf.zeros(self.n_items), name="display_control"
        )
        self.advert_control = tf.Variable(tf.zeros(self.n_items), name="advert_control")

    @property
    def trainable_weights(self):
        base = super().trainable_weights
        return base + (
            [
                self.store_control,
                self.calendar_control,
                self.display_control,
                self.advert_control,
            ]
            if hasattr(self, "store_control")
            else []
        )

    def compute_batch_utility(
        self,
        item_batch,
        basket_batch,
        store_batch,
        week_batch,
        price_batch,
        user_batch,
        available_item_batch,
    ):
        utility = super().compute_batch_utility(
            item_batch,
            basket_batch,
            store_batch,
            week_batch,
            price_batch,
            user_batch,
            available_item_batch,
        )
        item = tf.cast(item_batch, tf.int32)
        rows = tf.cast(user_batch, tf.int32)
        if item.shape.rank == 2:
            rows = tf.broadcast_to(rows[:, None], tf.shape(item))
        pairs = tf.stack([rows, item], axis=-1)
        physical = tf.gather(self.obs_store, rows)
        extra = tf.gather_nd(self.store_control, tf.stack([physical, item], axis=-1))
        extra += tf.reduce_sum(
            tf.gather(self.obs_calendar, rows) * tf.gather(self.calendar_control, item),
            axis=-1,
        )
        extra += tf.gather_nd(self.obs_display, pairs) * tf.gather(
            self.display_control, item
        )
        extra += tf.gather_nd(self.obs_advert, pairs) * tf.gather(
            self.advert_control, item
        )
        return utility + extra


def arrays(d, calendar_support=(1, 30)):
    y = d["member"].reshape(len(d["house"]), -1)
    items = np.nonzero(y)[1] + 1
    offsets = np.r_[0, np.cumsum(y.sum(1).astype(int))]
    phase = (np.clip(d["week"], *calendar_support) - 1) * 2 * np.pi / 52
    return {
        **d,
        "member_flat": y,
        "item": items,
        "offset": offsets,
        "season": np.zeros(len(y), int),
        "calendar": np.stack(
            [np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)], -1
        ),
        "dlp": d["dlp"].reshape(len(y), -1),
        "display": np.pad(d["disp"].reshape(len(y), -1), ((0, 0), (1, 0))),
        "advert": np.pad(d["mail"].reshape(len(y), -1), ((0, 0), (1, 0))),
    }


def base_utility(w, d):
    h = d["house"]
    lp = np.pad(d["dlp"], ((0, 0), (1, 0)))
    return (
        np.r_[0, w["lambda_"]]
        + np.einsum("br,jr->bj", w["theta"][h], w["alpha"])
        - np.einsum("br,jr->bj", w["gamma"][h], w["beta"]) * lp
        + w["store_control"][d["store"]]
        + np.einsum("br,jr->bj", d["calendar"], w["calendar_control"])
        + d["display"] * w["display_control"]
        + d["advert"] * w["advert_control"]
    ).astype("float64")


def interaction_matrix(w):
    return np.einsum("ir,jr->ij", w["alpha"], w["rho"]).astype("float64")


def set_ll(base, W, y):
    """Exact probability of each unordered observed set, conditional nonempty."""
    sizes = y.sum(1).astype(int)
    out = np.zeros(len(y))
    J = base.shape[1]
    assert (
        np.all(sizes > 0) and sizes.max() <= 12
    ), "Guard exponential work; never silently truncate"
    for n in np.unique(sizes):
        allrows = np.flatnonzero(sizes == n)
        for start in range(0, len(allrows), 512):
            ids = allrows[start : start + 512]
            items = np.nonzero(y[ids])[1].reshape(len(ids), n) + 1
            B = len(ids)
            rows = np.arange(B)
            dp = np.full((B, 1 << n), -np.inf)
            dp[:, 0] = 0.0
            for mask in range(1 << n):
                chosen = [j for j in range(n) if mask & (1 << j)]
                u = base[ids].copy()
                allowed = np.ones((B, J), bool)
                if chosen:
                    selected = items[:, chosen]
                    u += W[selected].mean(1)
                    allowed[rows[:, None], selected] = False
                else:
                    allowed[:, 0] = False
                norm = logsumexp(np.where(allowed, u, -np.inf), axis=1)
                if mask == (1 << n) - 1:
                    out[ids] = dp[:, mask] + u[:, 0] - norm
                else:
                    for j in range(n):
                        if not mask & (1 << j):
                            nxt = mask | (1 << j)
                            dp[:, nxt] = np.logaddexp(
                                dp[:, nxt], dp[:, mask] + u[rows, items[:, j]] - norm
                            )
    assert np.isfinite(out).all()
    return out


def make(rank, seed, cfg, train, val):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    m = ObservedShopper(
        price_effects=True,
        seasonal_effects=False,
        think_ahead=False,
        epsilon_price=0.0,
        latent_sizes={"preferences": rank, "price": 4, "season": 2},
        n_negative_samples=8,
        lr=0.003,
        weight_decay=0.001,
        batch_size=32,
    )
    m.instantiate(cfg["C"] * cfg["P"] + 1, cfg["users"])
    rng = np.random.default_rng(seed)
    for v in m.trainable_weights:
        v.assign(rng.normal(0, 0.1, v.shape).astype("float32"))
    frequency = train["member_flat"].mean(0)
    extra = max(float(frequency.sum() - 1), 0.01)
    m.lambda_.assign(
        np.log(np.maximum(frequency / frequency.sum() * extra, 1e-7)).astype("float32")
    )
    combined = {
        k: np.concatenate([train[k], val[k]])
        for k in ("store", "calendar", "display", "advert")
    }
    m.add_observables(
        combined["store"],
        combined["calendar"],
        combined["display"],
        combined["advert"],
        cfg["stores"],
    )
    return m
