"""Durable full-data fits. A failed prerequisite stops dependent fits."""

import copy
import fcntl
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from . import common as c
import numpy as np
import torch


def selected(m, name, row):
    """Save the selected model and update the run's model-selection ledger."""
    c.save_model(m, c.OUT / "selected" / name)
    ledger = (
        json.loads((c.OUT / "fit_records.json").read_text())
        if (c.OUT / "fit_records.json").exists()
        else {}
    )
    ledger[name] = row
    c.write("fit_records.json", ledger)


def full_score(m, data):
    """Score every basket in a split with bounded memory and no ridge penalty."""
    with torch.no_grad():
        return float(
            torch.cat(
                [
                    m.log_probability(*data.batch(slice(i, i + 512)))
                    for i in range(0, len(data), 512)
                ]
            ).mean()
        )


def gated(row, name):
    """Block dependent training when a required stage fails final stationarity."""
    if not row["final_stationarity_passed"]:
        raise RuntimeError("Unresolved final gradient failure in " + name)


def create_structure(parent, blocks):
    """Copy the independent parent and activate only the requested structural blocks."""
    cfg = {**parent.config, "independent_only": False, "trainable_parameters": blocks}
    m = c.CachedStructure(cfg)
    m.load_state_dict(parent.state_dict())
    return m


def size_cache(model, data, name):
    """Cache frozen within-subset likelihoods for total-size baseline fitting."""
    cache_directory = c.OUT / "cache" / name
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = cache_directory / "within.npy"
    within_likelihood = np.lib.format.open_memmap(
        path, mode="w+", dtype="float64", shape=(len(data),)
    )
    for batch_start in range(0, len(data), 256):
        context, baskets = data.batch(slice(batch_start, batch_start + 256))
        with torch.no_grad():
            utilities, _ = model.terms(context)
            sizes = baskets.sum((1, 2)).long()
            log_normalizers = c.log_esp(utilities, int(sizes.max()))
            within_likelihood[batch_start : batch_start + 256] = (
                (utilities * baskets.reshape(-1, model.J)).sum(-1)
                - log_normalizers.gather(1, sizes[:, None]).ravel()
            ).numpy()
        if batch_start % 8192 == 0:
            c.status("size_cache", rows=batch_start, total=len(data))
    within_likelihood.flush()
    return {
        **data.ctx,
        "_size_within": c.Shared(np.arange(len(data)), np.load(path, mmap_mode="r")),
    }, data.y


def run():
    """Fit preferences, structural warmups, the basic model, extension and baselines.

    Select only on validation; fit/test manifests freeze source and data identity.
    Required failures stop dependent fits. Rejected optional extensions retain
    the basic parent instead of relabeling the failed attempt as converged.
    """
    start = time.monotonic()
    config = json.loads((c.OUT / "data/config.json").read_text())
    training, validation = c.Data("train"), c.Data("validation")
    # Freeze scientific source/data identity before fitting; validate at resume.
    from .provenance import freeze

    freeze(c.OUT, "fit_manifest.json", include_data=True)
    assert json.loads((c.OUT / "preflight.json").read_text())["state"] == "passed"
    pooled_start = time.monotonic()
    pooled = c.independent_start(config, training.y, c.SEED)
    pooled_seconds = time.monotonic() - pooled_start
    pooled_model = c.Independent(pooled.config)
    pooled_model.load_state_dict(pooled.state_dict())
    selected(
        pooled_model,
        "popularity",
        {
            "label": "Training-only pooled conditional independent MLE",
            "final_stationarity_passed": True,
            "validation_ll": full_score(pooled_model, validation),
            "seconds": pooled_seconds,
        },
    )
    # All data used; same direct tastes and validation-only ridge selection.
    preferences = []
    for name in ("no_price", "independent"):
        candidates = []
        for ridge in (1.0, 10.0):
            model = c.Independent(pooled_model.config)
            model.load_state_dict(pooled_model.state_dict())
            if name == "no_price":
                model.config["trainable_parameters"] = [
                    n
                    for n, parameter in model.named_parameters()
                    if parameter.requires_grad and n != "choice_price"
                ]
                model.choice_price.requires_grad_(False)
            model, row = c.run_fit(
                model,
                f"{name}_ridge{ridge:g}",
                training.pair(),
                validation.pair(),
                ridge,
            )
            candidates.append((model, row))
        model, row = max(candidates, key=lambda x: x[1]["best_validation_ll"])
        gated(row, name)
        selected(
            model,
            name,
            {
                "fits": [fit_record for _, fit_record in candidates],
                "selected": row,
                "validation_ll": row["best_validation_ll"],
                "final_stationarity_passed": row["final_stationarity_passed"],
            },
        )
        if name == "independent":
            parent = model
    # Shared frozen ESP cache, exact mathematical equivalence checked preflight.
    category_blocks = ["linear_p", "linear_q"] if config["C"] > 10 else ["V"]
    stage = create_structure(parent, ["rho", "kappa"] + category_blocks)
    training_cache = c.materialize_cache(
        stage, training, "structure_train", "structure"
    )
    validation_cache = c.materialize_cache(
        stage, validation, "structure_validation", "structure"
    )
    for name, blocks in [
        ("independent_categories", ["rho", "kappa"]),
        ("basic", ["rho", "kappa"] + category_blocks),
    ]:
        model = create_structure(parent, blocks)
        warm = []
        for warmup_blocks, label in [
            (["rho"], "count_warmup"),
            ([n for n in blocks if n != "rho"], "category_warmup"),
        ]:
            warmup_model = create_structure(parent, warmup_blocks)
            warmup_model.load_state_dict(model.state_dict())
            warmup_model, row = c.run_fit(
                warmup_model, name + "_" + label, training_cache, validation_cache
            )
            gated(row, name + "_" + label)
            warmup_model.load_state_dict(
                torch.load(
                    c.OUT
                    / "fits"
                    / (name + "_" + label)
                    / (
                        "continuation/final.pt"
                        if "initial_attempt" in row
                        else "final.pt"
                    ),
                    weights_only=True,
                )
            )
            model.load_state_dict(warmup_model.state_dict())
            warm.append(row)
        model, row = c.run_fit(model, name, training_cache, validation_cache)
        gated(row, name)
        parent_validation_ll = full_score(parent, validation)
        if row["best_validation_ll"] < parent_validation_ll:
            model = create_structure(parent, blocks)
            row["parent_retained"] = True
            row["effective_validation_ll"] = parent_validation_ll
        else:
            row["effective_validation_ll"] = row["best_validation_ll"]
        selected(
            model,
            name,
            {
                "fit": row,
                "warmups": warm,
                "validation_ll": row["effective_validation_ll"],
                "final_stationarity_passed": True,
            },
        )
        if name == "basic":
            basic = model
    del training_cache, validation_cache
    # Projected independent reference is explicit in the released product head.
    head = c.ProductInteractionModel(basic, 4, c.SEED)
    product_training_cache = c.materialize_cache(
        head, training, "product_train", "product"
    )
    product_validation_cache = c.materialize_cache(
        head, validation, "product_validation", "product"
    )
    head, row = c.run_fit(
        head, "product", product_training_cache, product_validation_cache
    )
    parent_validation_ll = full_score(basic, validation)
    row["original_chain_parent_validation_ll"] = parent_validation_ll
    row["accepted"] = row["best_validation_ll"] > parent_validation_ll + 1e-10
    if row["accepted"]:
        gated(row, "product")
        selected(
            head,
            "product",
            {
                "fit": row,
                "validation_ll": row["best_validation_ll"],
                "final_stationarity_passed": True,
            },
        )
    else:
        selected(
            basic,
            "product_fallback",
            {
                "fit": row,
                "validation_ll": parent_validation_ll,
                "label": "Optional head rejected; converged basic parent retained",
                "final_stationarity_passed": True,
            },
        )
    del product_training_cache, product_validation_cache, head
    for name in ("mission", "neural", "size_aware"):
        if time.monotonic() - start > 14400:
            c.write(
                "budget_stop.json",
                {"next_model": name, "elapsed_seconds": time.monotonic() - start},
            )
            break
        model = (
            c.MissionESP(parent, c.SEED)
            if name == "mission"
            else (
                c.Neural(parent, c.SEED)
                if name == "neural"
                else c.SizeESP(parent, training.y)
            )
        )
        tp, vp = training.pair(), validation.pair()
        if name == "size_aware":
            tp, vp = size_cache(model, training, "size_train"), size_cache(
                model, validation, "size_validation"
            )
        model, row = c.run_fit(model, name, tp, vp)
        selected(
            model,
            name,
            {
                "fit": row,
                "validation_ll": row["best_validation_ll"],
                "final_stationarity_passed": row["final_stationarity_passed"],
            },
        )
    import platform

    c.write(
        "fits_complete.json",
        {
            "state": (
                "complete"
                if not (c.OUT / "budget_stop.json").exists()
                else "budget_stop"
            ),
            "seconds": time.monotonic() - start,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (2**20 if platform.system() == "Darwin" else 1024),
        },
    )
    c.status("fits_complete", seconds=time.monotonic() - start)
    # Launch scoring only after fitting and its gating have completed.
    from . import evaluate

    evaluate.main()
