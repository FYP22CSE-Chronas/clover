# EC-CLOVER: implementation note

Traced from the baseline repository before any code was written here (see
[IMPLEMENTATION.md](../../IMPLEMENTATION.md) for the repo-wide rules this experiment
follows). This is the actual tensor flow found in `model/network.py`, not a
specification.

## Baseline forward pass (`CLOVER.forward`, `model/network.py:91-114`)

```
WindowBatch.insample_y            [B, L, Nb]                 normalized, raw target
        |  permute(0,2,1)
        v  [B, Nb, L]
   einsum("ij,bjl->bil", S, .)     S: [Na+Nb, Nb] = [A; I]
        v
y_hier                             [B, Na+Nb, L]              hierarchy-level history
        |
        v
encoder(y_hier, hist_channels)     ENCODERS["dilated_conv"|"mlp"]
        v
h_enc                              [B, Na+Nb, C]              C = temp_conv_channels
        |
        v
mixer(h_enc)                       MIXERS["cross_series_mlp"|"identity"]
        v
h_hist                              [B, Nb, C]                 <- cross-series repr.
        |
        +-- static_mlp(stat_exog)  [B, Nb, static_dim]   (optional, concatenated)
        +-- future_mlp(futr_h)     [B, Nb, future_dim]   (optional, concatenated)
        v
context = cat(...)                  [B, Nb, context_dim]
        |
        v
decoder(context, futr_h)            DECODERS["two_stage"]
        v
z                                    [B, Nb, H, decoder.out_dim]
        |
        v
head(z)                             HEADS["factor_model"|"skew_t"|"gmm"]
        v
FactorParams(mu, sigma, F)          mu/sigma: [B, H, Nb], F: [B, H, Nb, K]
```

Downstream (`training/trainer.py`, `model/normalization.py`, `model/distribution.py`):
`denormalize_params` pushes `(loc, scale)` from `ScaleStats` back onto `mu/sigma/F`,
`sample_coherent` draws `mu + sigma*z + F@eps` at the bottom level and aggregates
through `S` via `coherent_aggregate` (clip-then-aggregate), and `crps` (or `energy`)
in `training/losses.py` scores the aggregated samples against `aggregate_targets`.

Key shape facts that drove the design below:

- `S = [A; I]` (`data/hierarchy.py` convention, asserted in `CLOVER.__init__`), so the
  trailing `Nb` rows of `y_hier` are exactly the raw bottom-level history, unencoded.
  Both mixers (`cross_series_mlp`, `identity`) select those same trailing `Nb` rows as
  their residual path (`model/mixers.py:37,58`).
- `batch.insample_y` is **already** `[B, L, Nb]` — the exact orientation the task
  spec asks for (`Y ∈ R^(B×L×Nb)`); no transpose bookkeeping is needed to hand it to
  an EC branch.
- `h_hist` (`[B, Nb, C]`) is the first point in the forward pass that is (a) already
  reduced to the bottom-level axis and (b) has not yet been concatenated with
  static/future context or handed to the decoder. That makes it the exact insertion
  point the task's §13 asks for: "Temporal/Cross-Series representation + EC
  representation -> Fusion -> existing decoder."
- `Encoder` implementations (`model/encoders.py`) already have the right contract for
  an EC encoder: `[B, N, L] (+ optional [B, N, X, L]) -> [B, N, C]`, treating the
  second axis as an independent-series axis with shared weights. Feeding an `[B, Nb,
  L]` EC sequence through one of these unchanged reuses the existing registry instead
  of inventing a new temporal block.

## Insertion point chosen

Immediately after `h_hist = self.mixer(self.encoder(y_hier, hist_channels))`, replace
`h_hist` with `fusion(h_hist, ec_branch(batch.insample_y))`, where `fusion` maps
`[B, Nb, C] x [B, Nb, C_ec] -> [B, Nb, C]` (same `C`, so nothing downstream —
static/future MLPs, `context_dim`, the decoder, the head — needs to change shape).

## Two defects found in the first implementation

Both were diagnosed after the first round of experiments returned parity-with-baseline
on `labour` and `tourism_small`, and both are fixed inside the branch (no CLOVER module
is touched). The fixes are config options that default to the old behaviour.

**1. `concat` fusion discards the baseline representation at initialization.**
`FusionMLP` sends `[h_clover; h_ec]` through a freshly initialized MLP, so at step 0 the
encoder/mixer output the baseline depends on is scrambled rather than passed through.
EC-CLOVER therefore starts *worse* than baseline and has to spend training re-learning
an approximate identity before the EC signal can pay for itself.
`fusion="gated_residual"` returns `h_clover + sigmoid(gate) * MLP([h_clover; h_ec])`,
which preserves the baseline path exactly and starts the EC contribution small
(`gate_init=-2.0` -> ~0.12), so the branch has to earn influence through the CRPS
gradient.

**2. Per-window normalization destroys the level information cointegration lives on.**
`WindowBatch.insample_y` is standardized per window per series (`window_stats`, the
`standard` scaler), but cointegration is a property of *raw levels*. With
`y_norm = (y - mu_w)/sigma_w`,

```
beta^T y_norm = sum_i (beta_i / sigma_i) y_i - sum_i (beta_i mu_i / sigma_i)
```

so (a) the effective cointegrating vector is rescaled per window by `1/sigma_i`, and
(b) a window-dependent constant is added. No single fixed `beta` reproduces the raw
linear combination across windows with different `sigma_w`, which is why a
Johansen-estimated `beta` never transferred: the model never sees the series the
Johansen test was run on. `sequence_norm="center"` removes the offset so the branch
models equilibrium *deviation dynamics* — the part that survives normalization —
rather than a corrupted level. Fully undoing the rescaling would need `(loc, scale)`
inside `WindowBatch`, which is a core contract change and deliberately not done here.

**Also added** (branch-internal, VECM-motivated): `include_diff` appends the sequence's
first difference as a second encoder channel, supplying the short-run `Gamma delta y`
half of a VECM that the first implementation omitted (it modelled only the long-run
`alpha beta^T y` term); `ec_input="e"` encodes the `r` equilibrium errors directly
instead of the `Nb` redundant channels of their rank-`r` image; `encoder_dilations`
lets the EC encoder look further back than CLOVER's; `beta_normalize="columns"` pins
the scale half of the identification problem.

## Why `forward` is overridden

This requires overriding `forward`, not just adding a registry component: no existing
registry protocol (`Encoder`, `Mixer`, `Decoder`, `Head`) is handed the raw
`batch.insample_y` at the point the EC branch needs it (`Mixer.__call__(self, h)` only
sees post-encoder embeddings). `experiments/ec_clover/network.py` subclasses `CLOVER`,
builds all baseline submodules via `super().__init__()` (so encoder/mixer/decoder/head
are byte-for-byte the registry components the baseline would build), and overrides
`forward` only to splice in the two extra lines. With `ec.enabled=False` the subclass
never builds the EC modules and `forward` delegates straight to `CLOVER.forward`, so
baseline behaviour is exact, not approximate.
