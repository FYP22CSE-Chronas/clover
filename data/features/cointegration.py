from __future__ import annotations

import numpy as np

from data.dataset import HierarchicalDataset
from data.features.base import FeatureBlock, register_feature

EQUILIBRIUM_FEATURE = "equilibrium_error"


def johansen_beta(
    levels: np.ndarray,
    rank: int,
    k_ar_diff: int = 1,
    deterministic: str = "const",
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Johansen's reduced-rank estimate of the cointegrating space of `levels`.

    `levels` is `[T, m]` in raw units. Returns `(beta, eigenvalues)`: `beta` is
    `[m, rank]` with unit-norm columns, `eigenvalues` the `rank` largest canonical
    correlations in descending order. Columns of `beta` span the directions along which
    the levels are most strongly pulled back after a shock, i.e. the stationary linear
    combinations if any exist.

    The estimator regresses `delta y_t` and `y_{t-1}` on `k_ar_diff` lags of
    `delta y` (plus an intercept when `deterministic == "const"`), then solves the
    generalized eigenproblem `S10 S00^-1 S01 v = lambda S11 v` on the residual moment
    matrices. `ridge` adds `ridge * mean(diag)` to `S00` and `S11` before solving;
    with more series than usable rows (traffic's 200 lanes over 120 training rows)
    `S11` is singular and some shrinkage is the only way to get an estimate at all.
    """
    if levels.ndim != 2:
        raise ValueError(f"levels must be [T, m], got shape {levels.shape}")
    n_steps, m = levels.shape
    if not 1 <= rank <= m:
        raise ValueError(f"rank must lie in [1, {m}], got {rank}")
    if k_ar_diff < 0:
        raise ValueError(f"k_ar_diff must be >= 0, got {k_ar_diff}")
    if deterministic not in ("none", "const"):
        raise ValueError(
            f"deterministic must be 'none' or 'const', got {deterministic!r}"
        )
    if ridge < 0:
        raise ValueError(f"ridge must be >= 0, got {ridge}")
    y = np.asarray(levels, dtype=np.float64)
    dy = np.diff(y, axis=0)  # [T-1, m]
    p = k_ar_diff
    n = dy.shape[0] - p
    if n < 2:
        raise ValueError(
            f"need more than k_ar_diff + 2 = {p + 2} rows for Johansen, got {n_steps}"
        )
    r0 = dy[p:]  # delta y_t
    r1 = y[p:-1]  # y_{t-1}
    regressors = [dy[p - j - 1 : n + p - j - 1] for j in range(p)]
    if deterministic == "const":
        regressors.append(np.ones((n, 1)))
    if regressors:
        z = np.concatenate(regressors, axis=1)
        r0 = r0 - z @ np.linalg.lstsq(z, r0, rcond=None)[0]
        r1 = r1 - z @ np.linalg.lstsq(z, r1, rcond=None)[0]

    s00 = r0.T @ r0 / n
    s11 = r1.T @ r1 / n
    s01 = r0.T @ r1 / n
    if ridge > 0:
        s00 = s00 + ridge * np.trace(s00) / m * np.eye(m)
        s11 = s11 + ridge * np.trace(s11) / m * np.eye(m)
    try:
        chol = np.linalg.cholesky(s11)
        s00_inv = np.linalg.inv(s00)
    except np.linalg.LinAlgError:
        raise ValueError(
            f"Johansen moment matrices are singular for {m} series over {n} usable "
            "rows; raise ridge, lower rank, or estimate within smaller groups "
            "(the level= option)"
        ) from None
    # Whiten with S11 so the problem is a symmetric eigenproblem; the eigenvectors
    # come back in the whitened coordinates and are mapped back through L^-T.
    inner = np.linalg.solve(chol, s01.T @ s00_inv @ s01)
    inner = np.linalg.solve(chol, inner.T).T
    eigenvalues, vectors = np.linalg.eigh((inner + inner.T) / 2)
    order = np.argsort(eigenvalues)[::-1][:rank]
    beta = np.linalg.solve(chol.T, vectors[:, order])
    beta = beta / np.linalg.norm(beta, axis=0, keepdims=True)
    # Sign is not identified; fix it so the largest-magnitude loading is positive.
    signs = np.sign(beta[np.abs(beta).argmax(axis=0), np.arange(beta.shape[1])])
    signs[signs == 0] = 1.0
    return (beta * signs).astype(np.float32), eigenvalues[order].astype(np.float32)


def bottom_groups(dataset: HierarchicalDataset, level: str | None) -> list[np.ndarray]:
    """Index sets of bottom series to test jointly: all of them, or siblings per node.

    With `level` given, every aggregate node at that level yields one group of the
    bottom series it sums; the groups must partition the bottom level, which holds
    for any level of a strict or grouped hierarchy.
    """
    n_bottom = dataset.n_bottom
    if level is None:
        return [np.arange(n_bottom)]
    try:
        start, stop = dataset.level_slices[level]
    except KeyError:
        raise KeyError(
            f"unknown level {level!r}; available: {sorted(dataset.level_slices)}"
        ) from None
    groups = [np.flatnonzero(dataset.S[row] > 0) for row in range(start, stop)]
    counts = np.zeros(n_bottom, dtype=int)
    for g in groups:
        counts[g] += 1
    if (counts != 1).any():
        raise ValueError(
            f"level {level!r} does not partition the bottom series; "
            f"{int((counts != 1).sum())} series fall in zero or several nodes"
        )
    return groups


def share_deviation(
    levels: np.ndarray, fit_end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each series' deviation from its long-run share of the group total.

    `levels` is `[T, m]`. The share `w_j` is the OLS slope of series `j` on the group
    total over rows `[:fit_end]`, so `sum_j w_j = 1` exactly and the deviations
    `y_j - w_j * total - c_j` sum to zero within the group: they span the `m - 1`
    cointegrating relations of a group driven by one common trend. Unlike Johansen,
    this uses the hierarchy itself as the identifying restriction, and the slope on a
    total dominated by the trend is well determined even when the adjustment is slow.

    Returns `(deviations [T, m], shares [m], beta [m, m-1])`, `beta` an orthonormal
    basis of the share vector's orthogonal complement.
    """
    total = levels.sum(axis=1)
    design = np.column_stack([total[:fit_end], np.ones(fit_end)])
    coef = np.linalg.lstsq(design, levels[:fit_end], rcond=None)[0]  # [2, m]
    shares, intercepts = coef[0], coef[1]
    deviations = levels - np.outer(total, shares) - intercepts
    _, _, vt = np.linalg.svd(shares[None, :])
    beta = vt[1:].T  # null space of the share vector
    return (
        deviations.astype(np.float32),
        shares.astype(np.float32),
        beta.astype(np.float32),
    )


@register_feature(EQUILIBRIUM_FEATURE)
def equilibrium_error(
    dataset: HierarchicalDataset,
    rank: int = 1,
    level: str | None = None,
    method: str = "johansen",
    k_ar_diff: int = 1,
    deterministic: str = "const",
    ridge: float = 0.0,
    fit_end: int | None = None,
) -> FeatureBlock:
    """[T, Nb, X] equilibrium errors as a history feature, `X = rank` or 1.

    `method="johansen"` estimates `rank` cointegrating vectors per group and emits
    `beta^T y_t`, the same channels for every member of a group. `method="share"`
    needs a `level` and emits one channel per series: its deviation from its long-run
    share of the group total (see `share_deviation`), with the group's full `m - 1`
    relations handed over as `beta`. Share identification is the better default on
    hierarchies: with slow adjustment Johansen's small-sample bias leaks the common
    trend into the "error", which then drifts out of sample.

    `beta` is estimated once on rows `[:fit_end]` -- the training span, injected by
    `build_feature` -- and applied to the whole series, so a window only ever sees a
    fixed linear combination of observations it already contains. The pipeline then
    z-scores each channel with training-row statistics; the data layer leaves the
    errors in level units.

    Series `n` carries the errors of the group it was tested in. With `level=None`
    every bottom series is in one group and all `Nb` series share the same `rank`
    channels; with `level="Country/Purpose/State"` (tourism_small) each pair of
    City/NonCity siblings gets its own relation, which keeps the Johansen problem
    small enough to be estimable on short series. A group with fewer than
    `rank + 1` members gets as many relations as it can support and zeros elsewhere.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if method not in ("johansen", "share"):
        raise ValueError(f"method must be 'johansen' or 'share', got {method!r}")
    if method == "share" and level is None:
        raise ValueError("method='share' needs a level whose nodes define the groups")
    n_steps = dataset.n_steps
    end = n_steps if fit_end is None else int(fit_end)
    if not 2 < end <= n_steps:
        raise ValueError(f"fit_end must lie in (2, {n_steps}], got {fit_end}")
    n_channels = 1 if method == "share" else rank
    out = np.zeros((n_steps, dataset.n_bottom, n_channels), dtype=np.float32)
    columns: list[np.ndarray] = []  # each a full-width [Nb] cointegrating vector
    for group in bottom_groups(dataset, level):
        if len(group) < 2:
            continue
        levels = dataset.bottom[:, group]
        if method == "share":
            deviations, _, beta = share_deviation(levels, end)
            out[:, group, 0] = deviations
        else:
            n_relations = min(rank, len(group) - 1)
            beta, _ = johansen_beta(
                levels[:end],
                n_relations,
                k_ar_diff=k_ar_diff,
                deterministic=deterministic,
                ridge=ridge,
            )
            out[:, group, :n_relations] = (levels @ beta)[:, None, :]
        for col in beta.T:
            full = np.zeros(dataset.n_bottom, dtype=np.float32)
            full[group] = col
            columns.append(full)
    if method == "share":
        names = ["share_deviation"]
    else:
        names = [f"eq_error_{k}" for k in range(rank)]
    return FeatureBlock(
        kind="hist_series",
        values=out,
        names=names,
        beta=np.stack(columns, axis=1) if columns else None,
    )
