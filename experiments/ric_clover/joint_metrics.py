from __future__ import annotations

import numpy as np
import torch

# Proper scores of the joint bottom-level forecast distribution. sCRPS scores every
# series' marginal separately and only sees cross-series dependence through the
# aggregates; these two score the dependence directly. Every bottom series is divided
# by its mean absolute value over the training rows first, so a large series does not
# dominate the Euclidean geometry.


def energy_score(y: torch.Tensor, x: torch.Tensor) -> float:
    """ES(y, X) = E|X - y| - 0.5 E|X - X'|, Euclidean, for y [d] and samples X [N, d].

    The second term uses every distinct pair (unbiased U-statistic).
    """
    n = x.shape[0]
    term1 = torch.linalg.vector_norm(x - y, dim=1).mean()
    pair = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
    term2 = pair.sum() / (n * (n - 1))
    return float(term1 - 0.5 * term2)


def variogram_score(y: torch.Tensor, x: torch.Tensor, p: float = 0.5) -> float:
    """VS_p(y, X) = sum_{i<j} (|y_i - y_j|^p - E|X_i - X_j|^p)^2 (Scheuerer & Hamill).

    Unit weights; the pair sum is over distinct series.
    """
    obs = (y[:, None] - y[None, :]).abs().pow(p)  # [d, d]
    pred = (x[:, :, None] - x[:, None, :]).abs().pow(p).mean(0)  # [d, d]
    d = y.shape[0]
    upper = torch.triu(torch.ones(d, d, dtype=torch.bool), diagonal=1)
    return float((obs - pred).pow(2)[upper].sum())


def joint_scores(
    samples: np.ndarray, y_true: np.ndarray, n_bottom: int, scale: np.ndarray
) -> dict[str, float]:
    """Mean energy and variogram score over forecast steps, bottom level, scaled.

    `samples` is [Na+Nb, T, N] and `y_true` [Na+Nb, T] as `RunResult` stores them;
    `scale` is [Nb], each bottom series' mean |y| over the training rows.
    """
    scale_t = torch.as_tensor(np.maximum(scale, 1e-8), dtype=torch.float64)
    xs = torch.as_tensor(samples[-n_bottom:], dtype=torch.float64)  # [Nb, T, N]
    ys = torch.as_tensor(y_true[-n_bottom:], dtype=torch.float64)  # [Nb, T]
    es, vs = [], []
    for t in range(ys.shape[1]):
        y = ys[:, t] / scale_t
        x = (xs[:, t, :] / scale_t[:, None]).T.contiguous()  # [N, Nb]
        es.append(energy_score(y, x))
        vs.append(variogram_score(y, x))
    return {"energy": float(np.mean(es)), "variogram": float(np.mean(vs))}
