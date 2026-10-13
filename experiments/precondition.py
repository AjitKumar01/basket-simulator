"""Fixed diagonal Fisher coordinates. Objective, bounds and raw gradient unchanged."""

import numpy as np
import torch
from . import common as c
from .optimizer import fit


def scales(model, data, ridge):
    """Build fixed diagonal Fisher-like scales for active preference parameters.

    Use product frequency variance, ID frequencies and covariate second moments.
    Shared contexts are weighted by their training-basket multiplicities. Add
    the ridge curvature and bound the scale to avoid extreme coordinates.
    """
    context, baskets = data
    basket_count = len(baskets)
    product_count = model.C * model.P
    frequencies = baskets.numpy().mean(0).ravel()
    product_variance = np.clip(frequencies * (1 - frequencies), 1e-7, None)
    household_frequency = (
        np.bincount(context["house"].numpy(), minlength=model.config["users"])
        / basket_count
    )
    store_frequency = (
        np.bincount(context["store"].numpy(), minlength=model.config["stores"])
        / basket_count
    )
    weeks = context["week"].numpy().clip(*model.config["calendar_support"])
    phase = (weeks - 1) * 2 * np.pi / 52
    calendar = np.stack(
        [np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase)], -1
    )
    curvature = {
        "lam": product_variance,
        "house_effect": household_frequency[:, None] * product_variance,
        "store_effect": store_frequency[:, None] * product_variance,
        "mu": product_variance[:, None] * (calendar**2).mean(0),
    }
    for name, field in [("choice_price", "dlp"), ("w_dsp", "disp"), ("w_mlr", "mail")]:
        covariates = context[field]
        second_moment = np.zeros(product_count)
        if isinstance(covariates, torch.Tensor):
            curvature[name] = product_variance * covariates.numpy().reshape(
                basket_count, product_count
            ).__pow__(2).mean(0)
            continue
        context_counts = np.bincount(covariates.rows, minlength=len(covariates.values))
        for start in range(0, len(context_counts), 1024):
            batch_values = np.asarray(
                covariates.values[start : start + 1024], dtype=float
            ).reshape(-1, product_count)
            second_moment += (
                context_counts[start : start + len(batch_values)] @ batch_values**2
            )
        curvature[name] = product_variance * second_moment / basket_count
    rows = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        diagonal = curvature[name] + (0 if name == "lam" else ridge / basket_count)
        coordinate_scale = np.clip(1 / np.sqrt(np.maximum(diagonal, 1e-8)), 0.05, 10000)
        rows.append(np.broadcast_to(coordinate_scale, parameter.shape).ravel())
    return np.concatenate(rows)
