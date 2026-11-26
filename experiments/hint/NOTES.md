# HINT and HINT + DSF

A PyTorch port of HINT (Olivares et al., *HINT: Hierarchical Mixture Networks for
Coherent Probabilistic Forecasting*, 2023; arXiv 2305.07089) as implemented in
neuralforecast (`HINT` wrapping `NHITS` with a `GMM` loss), plus a variant whose
mixture kernel is a deep sigmoidal flow (DSF). Chosen over the HierE2E port
(`experiments/hiere2e/`) because NHITS trains in minutes on CPU where HierE2E's
LSTM took hours.

```
python -m experiments.hint.bench select   # tune on validation (seed 0)
python -m experiments.hint.bench run      # 5 seeds x {gaussian, dsf}, test window
python -m experiments.hint.bench report   # mean +/- 95% CI, wins, Wilcoxon
```

## What HINT is, as ported

- **Backbone (NHITS).** neuralforecast defaults: three identity stacks of one block
  each, max-pooling kernels [2, 2, 1], frequency downsampling [4, 2, 1], 512-unit
  MLPs, linear interpolation of the forecast knots, residual backcasts. Channel
  independent: one network shared by every hierarchy series, one window at a time.
- **Scaling.** Robust (median / MAD) per window; parameters go back to raw units as
  mu * scale + loc and (softplus(sigma) + 0.2) * scale (neuralforecast's GMM
  `scale_decouple`).
- **Multivariate mixture.** K = 10 Gaussian components with equal weights. Training
  sums each component's log-likelihood over the series of each origin before the
  log-sum-exp over components (neuralforecast's `batch_correlation=True`), so a
  component must explain all of them at once -- the paper's Eq. 7/28 composite
  likelihood. Sampling draws one component per (origin, step, path), shared by all
  series.
- **Coherence.** Reconciliation of samples at prediction only: y -> S P y with
  BottomUp, MinTrace-OLS or MinTrace-WLS, ported from neuralforecast's
  `get_bottomup_P` / `get_mintrace_*_P`.
- **Training.** Adam, 1024 windows per step (32 series x 32 origins), learning rate
  halved 3 times over the run, up to 3000 steps, validation every 100 steps, stop
  after 10 checks without improvement, restore the best.

## HINT + DSF

`kernel="dsf"` gives every component the density of
`z = DSF((y - mu) / sigma)`, z ~ N(0, 1): an exact change of variables, so the
mixture likelihood -- HINT's objective -- is unchanged in form. The flow (8 sigmoid
units, the repo's `_flow_shape`) is per (series, step) and shared by the
components, which keep their own mu and sigma. Sampling inverts the monotone flow
(bracket, bisection, safeguarded Newton). Backbone, scaling, mixture coupling,
reconciliation, schedule and the Gaussian-selected hyperparameters are identical.

The flow runs in the *normalizing* direction because HINT trains on likelihood;
the CLOVER flow head runs in the sampling direction because CLOVER trains on CRPS.
In this direction the slope floor `a_min` bounds how much the flow can widen a
component (tails up to sigma / a_min). With the repo's 0.1, likelihood training on
tourism_small learned near-floor slopes and ~6x too-wide forecasts (validation
sCRPS 0.37 vs 0.10 for Gaussian) while reaching a *lower* training NLL -- the
likelihood/CRPS mismatch. The floor is therefore the one DSF-specific
hyperparameter, chosen on validation from {0.1, 1.0}; at 1.0 the tails can be no
heavier than the Gaussian kernel's.

## Protocol

Seed 0, validation window only: Gaussian HINT over input length (m * h, m in
{2, 3, 4}, plus CLOVER's window; for h = 1, CLOVER's window and its half and
quarter) x learning rate {1e-3, 5e-4, 1e-4}, each run early-stopped; the best
(input length, learning rate, reconciler) wins. Then the DSF floor at those
settings. Then 5 seeds of each, test window, the repo's sCRPS (1% quantile grid,
1000 samples).

## Where the port departs from neuralforecast

1. **Naive level on means only.** NHITS starts its forecast from the last observed
   value and adds it to every output channel; here it is added to the means only, not
   to the raw scales.
2. **Batch layout.** neuralforecast samples 1024 windows from 32 series at mixed
   times and sums the likelihood over all of them. Here the 1024 windows are 32
   series x 32 origins and the sum runs over the series of one origin, which is the
   paper's "series at the same time" coupling.
3. **Joint sampling.** neuralforecast's current `GMM.sample` draws a component per
   element and then shuffles samples to "break quantile correlations", which makes
   series independent before reconciliation. Here components are shared across
   series per (origin, step, path), as in the paper's Eq. 7.
4. **Reconciler chosen on validation**, among the three; which one the paper's
   coherent column used is not stated in the text read.
5. **Validation samples** are capped at 2M draws per check (traffic: ~80 per
   window); the test uses 1000.

## Published reference (HINT paper, Table 1, overall sCRPS)

| dataset | HINT (coherent) | HierE2E | PERMBU-MinT |
|---|---|---|---|
| Labour | 0.0067 +/- 0.0001 | 0.0171 | 0.0069 |
| Traffic | 0.0589 +/- 0.0004 | 0.0426 | 0.0651 |
| Tourism | 0.0536 +/- 0.0004 | 0.0761 | 0.0680 |
| Wiki2 | 0.2447 +/- 0.0007 | 0.2592 | 0.3821 |

Horizons and splits there follow Rangapuram et al. (Tourism h = 8); the repo's
`tourism_small` uses h = 4 and the repo's traffic split. Same ballpark is the check,
not the same digits.
