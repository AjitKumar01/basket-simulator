"""Algebraically identical scaled-positive ESP; log-space fallback for wide ranges."""

import math
import time
import json
import numpy as np
import torch


def vector_log_esp(u, maximum):
    """Return log subset-weight sums for degrees 0 through maximum.

    For every item, update degree n by adding the choices that include this item
    to those that omit it. Vectorize over subset degrees and batch dimensions;
    loop only over products. This is the stable fallback for extreme utilities.
    """
    t = torch.cat(
        [
            torch.zeros_like(u[..., :1]),
            torch.full((*u.shape[:-1], maximum), -1e30, dtype=u.dtype, device=u.device),
        ],
        -1,
    )
    for j in range(u.shape[-1]):
        degree = min(maximum, j + 1)
        t = torch.cat(
            [
                t[..., :1],
                torch.logaddexp(
                    t[..., 1 : degree + 1], u[..., j, None] + t[..., :degree]
                ),
                t[..., degree + 1 :],
            ],
            -1,
        )
    return t


def fast_log_esp(u, maximum):
    """Compute the same subset sums in shifted positive-coefficient space.

    Use cheaper multiplication/addition when the utility range is safe. Fall
    back to log-space recursion before positive coefficients can underflow.
    The degree-dependent shift is restored before returning log normalizers.
    """
    # Detaching the common shift preserves the exact derivatives: the law is
    # invariant to adding n*shift back to each degree-n coefficient.
    shift = u.detach().amax(-1, keepdim=True)
    spread = shift - u.detach().amin(-1, keepdim=True)
    if float(spread.max()) * maximum > 0.8 * abs(math.log(torch.finfo(u.dtype).tiny)):
        return vector_log_esp(u, maximum)
    weights = (u - shift).exp()
    terms = torch.cat(
        [
            torch.ones_like(u[..., :1]),
            torch.zeros((*u.shape[:-1], maximum), dtype=u.dtype, device=u.device),
        ],
        -1,
    )
    for j in range(u.shape[-1]):
        degree = min(maximum, j + 1)
        terms = torch.cat(
            [
                terms[..., :1],
                terms[..., 1 : degree + 1]
                + weights[..., j, None] * terms[..., :degree],
                terms[..., degree + 1 :],
            ],
            -1,
        )
    out = terms.clamp(min=torch.finfo(u.dtype).tiny).log() + shift * torch.arange(
        maximum + 1, dtype=u.dtype, device=u.device
    )
    return torch.where(
        torch.arange(maximum + 1, device=u.device) <= u.shape[-1],
        out,
        torch.full_like(out, -1e30),
    )
