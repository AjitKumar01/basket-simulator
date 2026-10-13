"""Portable workflows: support, stochastic law, oracle truth and price timing."""

import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from experiments import common as c
from experiments import evaluate as e
from experiments import optimizer
from experiments.synthetic.worlds import World


def context(n, C=3, P=2):
    return {
        "house": torch.ones(n, dtype=torch.long),
        "store": torch.zeros(n, dtype=torch.long),
        "week": torch.full((n,), 20.0, dtype=torch.float64),
        **{
            k: torch.zeros((n, C, P), dtype=torch.float64)
            for k in ("dlp", "disp", "mail")
        },
    }


def parent():
    y = torch.tensor(
        list(itertools.product((0, 1), repeat=6))[1:], dtype=torch.uint8
    ).reshape(-1, 3, 2)
    return (
        c.independent_start(
            {
                "C": 3,
                "P": 2,
                "R": 2,
                "users": 3,
                "stores": 1,
                "calendar_support": [1, 30],
                "chain_rank": 0,
                "lbfgs_history": 3,
            },
            y,
            11,
        ),
        y,
    )


def test_all_current_samplers_match_normalized_tiny_likelihood():
    initial, y = parent()
    models = []
    independent = c.Independent(initial.config)
    independent.load_state_dict(initial.state_dict())
    structure = c.CachedStructure(
        {
            **initial.config,
            "independent_only": False,
            "trainable_parameters": ["rho", "kappa", "V"],
        }
    )
    structure.load_state_dict(initial.state_dict())
    with torch.no_grad():
        structure.rho.fill_(0.3)
        structure.V.fill_(0.2)
    models += [
        independent,
        structure,
        c.SizeESP(initial, y),
        c.MissionESP(initial, 12),
        c.Neural(initial, 13),
        c.ProductInteractionModel(structure, 2, 14),
    ]
    for model in models:
        with torch.no_grad():
            probability = model.log_probability(context(len(y)), y).exp().numpy()
        np.testing.assert_allclose(probability.sum(), 1, atol=1e-10)
        adapter = e.Adapter(model)
        cache = adapter.cache(context(1))
        draw = adapter.sample(
            {k: v.repeat_interleave(12000, 0) for k, v in cache.items()}, 19
        )
        assert draw.any((1, 2)).all()
        code = (draw.numpy().reshape(-1, 6) @ 2 ** np.arange(6)).astype(int)
        # Enumeration is in lexicographic order, so index by code rather than row.
        expected = np.zeros(64)
        expected[(y.numpy().reshape(-1, 6) @ 2 ** np.arange(6)).astype(int)] = (
            probability
        )
        frequencies = np.bincount(code, minlength=64) / len(draw)
        np.testing.assert_allclose(frequencies, expected, atol=0.012)


def test_oracle_worlds_normalize_and_zero_price_is_invariant():
    h = np.array([1, 2, 3])
    price = np.zeros((3, 3, 4))
    for family in (
        "latent_missions_gumbel",
        "product_ising",
        "latent_missions_gumbel_zero_price",
    ):
        world = World({"seed": 942000, "family": family})
        mass = world.probability(h, price)
        np.testing.assert_allclose(mass.sum(-1), 1, atol=1e-12)
        draw = world.sample(100, 942101)
        assert draw["member"].any((1, 2)).all()
        if "zero_price" in family:
            np.testing.assert_allclose(
                mass, world.probability(h, price + 0.2), atol=1e-12
            )


def test_packed_loader_discards_padding_bits_and_preserves_membership(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(c, "OUT", tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    (data / "config.json").write_text(
        json.dumps({"C": 3, "P": 2, "R": 2, "users": 3, "stores": 1})
    )
    y = np.array([[1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0]], dtype=np.uint8)
    np.savez(
        data / "train.npz",
        packed_member=np.packbits(y, axis=1),
        house=[1, 2],
        store=[0, 0],
        week=[1, 2],
        context=[0, 0],
    )
    for field in ("dlp", "disp", "mail"):
        np.save(data / (field + ".npy"), np.zeros((1, 3, 2)))
    loaded = c.Data("train")
    np.testing.assert_array_equal(loaded.y.numpy().reshape(2, 6), y)
    assert loaded.batch(slice(0, 2))[0]["dlp"].shape == (2, 3, 2)


def test_scaled_optimizer_reports_best_and_final_separately(tmp_path):
    initial, y = parent()
    model = c.Independent(initial.config)
    model.load_state_dict(initial.state_dict())
    model.config["trainable_parameters"] = ["lam"]
    for name, p in model.named_parameters():
        p.requires_grad_(name == "lam")
    with torch.no_grad():
        model.lam.add_(0.5)
    row = optimizer.fit(
        model,
        (context(len(y)), y),
        (context(len(y)), y),
        tmp_path / "fit",
        maxiter=50,
        maxcalls=100,
        time_limit=10,
        ridge=0,
        scales=np.full(6, 5.0),
    )
    assert row["final_stationarity_passed"]
    assert row["final_projected_gradient"] <= 2e-4
    assert row["final_scaled_projected_gradient"] <= 2e-4
    with torch.no_grad():
        score = float(model.log_probability(context(len(y)), y).mean())
    np.testing.assert_allclose(score, row["best_validation_ll"], atol=1e-12)


def test_erim_prepare_uses_training_catalogue_and_retrospective_contract(tmp_path):
    source = tmp_path / "canonical"
    source.mkdir()
    transactions = []
    prices = []
    for category in ("a", "b", "c"):
        for j in range(4):
            product = f"{category}:{j}"
            for period in (1, 31, 41):
                transactions.append(
                    {
                        "category": category,
                        "product_id": product,
                        "customer_id": 1,
                        "store_id": 1,
                        "basket_id": period,
                        "period": period,
                    }
                )
            for period, price in ((1, 2.0), (31, 7.0), (41, 9.0)):
                prices.append(
                    {
                        "category": category,
                        "product_id": product,
                        "store_id": 1,
                        "period": period,
                        "price_source": "retail_aggregate",
                        "average_price": price,
                        **{
                            k: "N"
                            for k in (
                                "end_display",
                                "front_display",
                                "aisle_display",
                                "other_display",
                                "advertised",
                            )
                        },
                    }
                )
    # A later-only popular product must not enter the training catalogue.
    transactions.append(
        {
            "category": "a",
            "product_id": "a:later",
            "customer_id": 99,
            "store_id": 1,
            "basket_id": 42,
            "period": 42,
        }
    )
    pd.DataFrame(transactions).to_parquet(source / "transactions.parquet")
    pd.DataFrame(prices).to_parquet(source / "store_week_prices.parquet")
    run = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments",
            "prepare",
            "--dataset",
            "erim",
            "--source",
            str(source),
            "--out",
            str(run),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    cfg = json.loads((run / "data/config.json").read_text())
    assert "a:later" not in cfg["products"]
    with np.load(run / "data/validation.npz") as data:
        np.testing.assert_allclose(data["dlp"], np.log(7 / 2))
        assert data["member"].shape == (1, 3, 4)


def test_dunnhumby_prepare_uses_strictly_earlier_prices(tmp_path):
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    products = [
        {"PRODUCT_ID": i + 1, "COMMODITY_DESC": f"category{i//16:02}"}
        for i in range(1024)
    ]
    pd.DataFrame(products).to_csv(raw / "product.csv", index=False)
    pd.DataFrame(
        [
            {
                "PRODUCT_ID": 1,
                "STORE_ID": 1,
                "WEEK_NO": 9,
                "display": "0",
                "mailer": "A",
            },
            {
                "PRODUCT_ID": 1,
                "STORE_ID": 1,
                "WEEK_NO": 101,
                "display": "1",
                "mailer": "0",
            },
        ]
    ).to_csv(raw / "causal_data.csv", index=False)
    tx = [
        {
            "household_key": 1,
            "BASKET_ID": w,
            "WEEK_NO": w,
            "PRODUCT_ID": p["PRODUCT_ID"],
            "STORE_ID": 1,
            "QUANTITY": 1,
        }
        for w in (9, 83, 91)
        for p in products
    ]
    pd.DataFrame(tx).to_parquet(source / "data/tx.parquet")
    national = [
        {"PRODUCT_ID": p["PRODUCT_ID"], "WEEK_NO": w, "price": 2.0}
        for p in products
        for w in (9, 82)
    ]
    pd.DataFrame(national).to_parquet(source / "data/price_week.parquet")
    store = [
        {"PRODUCT_ID": p["PRODUCT_ID"], "STORE_ID": 1, "WEEK_NO": w, "price": price}
        for p in products
        for w, price in ((1, 1.0), (9, 3.0), (83, 100.0))
    ]
    pd.DataFrame(store).to_parquet(source / "data/price_store_week.parquet")
    run = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments",
            "prepare",
            "--dataset",
            "dunnhumby",
            "--source",
            str(source / "data"),
            "--raw-source",
            str(raw),
            "--out",
            str(run),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    rows = np.load(run / "data/dlp.npy")
    contexts = pd.read_csv(run / "data/contexts.csv")
    np.testing.assert_allclose(rows[contexts.WEEK_NO == 83], np.log(3 / 2), atol=1e-7)
    np.testing.assert_allclose(rows[contexts.WEEK_NO == 91], np.log(100 / 2), atol=1e-7)
