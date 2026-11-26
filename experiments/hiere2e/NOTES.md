# HierE2E and HierE2E + DSF

A PyTorch port of HierE2E (Rangapuram et al., *End-to-End Learning of Coherent
Probabilistic Forecasts for Hierarchical Time Series*, ICML 2021), plus a variant
whose per-series marginal is a deep sigmoidal flow (DSF). Built because the official
code (GluonTS `DeepVARHierarchicalEstimator`) needs MXNet, which has no build for the
Python on this machine.

```
python -m experiments.hiere2e.bench select   # tune on validation (Gaussian, seed 0)
python -m experiments.hiere2e.bench run      # 5 seeds x {gaussian, dsf}, test window
python -m experiments.hiere2e.bench report   # mean +/- 95% CI, wins, Wilcoxon
```

## What HierE2E is, as ported

- **Backbone (DeepVAR).** One 2-layer LSTM (40 cells, zoneout 0.1 on the states,
  residual from layer 2) reads all N = Na+Nb series together. Its input at each step
  is every series' lags (divided by a per-window mean-absolute scale), one learned
  value per series, and calendar features.
- **Marginal.** Per step and series, an independent Gaussian: GluonTS's
  `DeepVARHierarchicalEstimator` hard-codes `rank=0`, so there is no low-rank factor.
- **Coherence.** Samples are projected, y -> M y with M = S (S^T S)^+ S^T, at every
  step. The projection is the only source of cross-series dependence.
- **Training.** Teacher forcing over context + horizon, loss = CRPS of 200 projected
  samples at every step. Adam 1e-3, weight decay 1e-8, 50 batches per epoch, batch 32,
  element-wise gradient clipping at 10, LR halved after 10 flat epochs (floor 5e-5).
- **Forecasting.** Autoregressive: one draw per path per step, projected, fed back as
  the next step's lag.
- **Lags / calendar.** Daily `[1, 7, 14]` with day of week; monthly `[1, 12]`;
  quarterly `[1]`.

## HierE2E + DSF

`marginal="dsf"` draws `mu + sigma * DSF(z)` instead of `mu + sigma * z`, with the
repo's `_dsf_transform` and `_flow_shape` (8 sigmoid units, slope floor 0.1). Nothing
else changes: same backbone, projection, loss, schedule, and the hyperparameters
selected for plain HierE2E. This is the substitution the CLOVER flow head makes.

## Protocol

Follows HierE2E's supplementary: tune context-length factor ({2,3,4} for long
horizons, {15,25,40,60} for short) and epochs ({10,25,50,100}) on the validation
window, choosing by lowest CRPS. Then run 5 seeds. Splits, validation and test origins
are the repo's own, so the scores sit next to CLOVER's. They are computed with the
repo's sCRPS (1% quantile grid, 1000 samples).

## Where the port departs from GluonTS

1. **Loss estimator (CPU budget).** GluonTS scores CRPS at every context step.
   Here every horizon step is scored, plus 8 random context steps reweighted by
   context/8, and CRPS uses a sort-free paired estimator. Both are unbiased for the
   same loss; they add gradient noise. `FitConfig(loss_context_steps=None,
   crps_estimator="sorted")` restores the exact GluonTS loss.
2. **No warm start in the grid.** The paper also tuned `warmstart_frac` in {0, 0.1};
   here it is fixed at GluonTS's default, 0.
3. **One loss form.** The paper also tried a likelihood-on-fitted-samples loss with 50
   samples for large datasets; only the 200-sample CRPS loss is used here.
4. **Calendar features.** Monthly uses month of year (GluonTS: week of year), quarterly
   quarter of year (GluonTS has no default). Fixed periods, not GluonTS's
   data-dependent max+1.
5. **Prediction samples.** 1000 for the test score (the paper used 200), to match how
   CLOVER is scored here; validation uses 200.
6. **Epochs share one run.** Each context factor trains once for 100 epochs and is
   scored at 10/25/50/100. Exact, because nothing in the schedule depends on the total.
7. **Output layer.** Variance = softplus(raw + b) + 1e-6 with b set so the initial
   sigma is 1 -- GluonTS's `LowrankMultivariateGaussianOutput` defaults as understood;
   that class was not read line by line.
8. **Datasets.** Tourism here is the repo's `tourism_small` with h = 4, as in the
   CLOVER paper; HierE2E's own Tourism results use h = 8, so those numbers are not
   directly comparable.

## Published reference (HierE2E supplementary, mean CRPS over all series, 5 runs)

| dataset | HierE2E | DeepVAR (no projection) |
|---|---|---|
| Labour | 0.0340 +/- 0.0088 | 0.0382 +/- 0.0045 |
| Traffic | 0.0376 +/- 0.0060 | 0.0400 +/- 0.0026 |
| Tourism (h = 8) | 0.0834 +/- 0.0052 | 0.0925 +/- 0.0022 |
| Wiki | 0.2038 +/- 0.0110 | 0.2294 +/- 0.0158 |

Their metric averages over series at each level with a 5% quantile grid; the repo's
"Overall" pools the whole hierarchy with a 1% grid. Expect the same ballpark, not the
same digits.
