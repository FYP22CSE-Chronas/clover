# Implementation guide

How this codebase is put together and the rules to follow when extending it. If a
change would break a rule here, change the rule deliberately rather than working around
it.

## 1. The layering rule

One direction only. [tests/test_layering.py](tests/test_layering.py) parses every
import and fails the build on a violation, so this is checked, not merely documented.

| layer | may import | must never import |
|---|---|---|
| `config.py`, `registry.py`, `contracts.py` | stdlib, torch, yaml | any subpackage |
| `model/` | the leaf modules | `data`, `pipeline`, `pandas` |
| `training/` | leaf modules, `model` | `data`, `pipeline` |
| `data/` | the leaf modules | `model`, `training`, `pipeline`, `torch` |
| `pipeline/` | everything | `experiments` |
| `experiments/` | everything | — |

`experiments/` does not exist yet; its rows are the standing convention for when it
does (§3). They are the one pair the test does not yet check — to enforce them, add
`"experiments"` to each layer's tuple in `FORBIDDEN` in `tests/test_layering.py`.

What this buys:

- `model/` can be reused on any hierarchy without dragging in CSV parsing.
- `data/` can be tested and reasoned about with no torch installed in the loop.
- Every dataset-specific decision has exactly one place it can live.

**Consequence to remember:** `data/` returns numpy and returns *raw units*. Anything
that turns numpy into a tensor, or normalizes, belongs to `pipeline/`.

`data/` holds both halves of the data concern: the loading code at its top level, and
the datasets it reads under `data/raw/<name>/` and `data/favorita/`. One folder, not
two.

**Flat-layout naming constraint.** The repository root is on `sys.path`, so any
top-level module shadows a stdlib module of the same name for the whole process. This
is why the tensor contracts live in `contracts.py` and not `types.py` — the latter
would shadow stdlib `types` and break anything importing it. Before adding a top-level
module, check the name against the standard library. Names *inside* a package are safe:
`data/features/calendar.py` is only reachable as `data.features.calendar` and does not
shadow stdlib `calendar`.

## 2. Where a new thing goes

| you are adding | put it in | register it with |
|---|---|---|
| a history encoder | `model/encoders.py` | `@ENCODERS.register("name")` |
| a cross-series mixer | `model/mixers.py` | `@MIXERS.register("name")` |
| a decoder | `model/decoders.py` | `@DECODERS.register("name")` |
| a distribution head | `model/heads.py` | `@HEADS.register("name")` |
| a window scaler | `model/normalization.py` | `@SCALERS.register("name")` |
| a training objective | `training/losses.py` | `@LOSSES.register("name")` |
| an exogenous feature | `data/features/` | `@register_feature("name")` |
| a train/val/test split | `data/windows.py` | `@register_split("name")` |
| a dataset | `configs/datasets/<name>.yaml` | nothing — config only |
| a one-off data prep job | `scripts/` | nothing — a `main()` |
| an **experimental model** | `experiments/` | see §3 |

There is no `if dataset_name == ...` anywhere in the codebase, and adding one is the
signal that something belongs in a registry instead.

## 3. Experimental models go in `experiments/`

`model/` holds the one architecture this project trains and ships. **Any experimental
model implementation belongs in `experiments/`, never in `model/`** — a new
architecture, a variant of the existing one, a reimplementation to compare against, a
half-finished idea.

```
experiments/
  <experiment-name>/     one folder per experiment, self-contained
    __init__.py            imports the modules below so their decorators run
    ...                    whatever that experiment needs
```

Rules:

- **`experiments/` may import the core; the core may never import `experiments/`.** The
  dependency runs one way, exactly as it does for the layers in §1. If a shipped module
  needs something from an experiment, that thing has stopped being an experiment —
  promote it into `model/` and delete the copy.
- **Reuse rather than fork.** An experiment that only swaps one component registers just
  that component against the existing registry (`@ENCODERS.register("my_encoder")`) and
  reuses the rest of the model unchanged. Copying `network.py` to change ten lines
  guarantees the copy rots.
- **Register under a distinct name**, then select it from config —
  `--set model.encoder=my_encoder`, or an `encoder:` key in a dataset config. Registries
  raise on a duplicate name, so a collision fails loudly at import.
- **Import the experiment to register it.** Decorators only run when the module is
  imported; do that in the experiment's own `__init__.py`, and import the experiment
  from wherever you launch it. Do not add experiment imports to `cli.py`, `model/` or
  `pipeline/` — that is the core depending on an experiment.
- **Experiment configs live beside the experiment**, not in `configs/datasets/`, which
  is reserved for the shipped datasets.
- **An experiment that is abandoned gets deleted**, not commented out. Git remembers it.

Anything selectable but genuinely part of the shipped model — the `crps` objective, the
scalers, the default encoder — stays in its home module. The line is whether the code is
something you are *evaluating* or something the project *runs*.

## 4. Adding a dataset

Six of the seven single hierarchies needed nothing but a config file.

1. Put `data.csv` and `agg_mat.csv` in `data/raw/<name>/`. The columns of `data.csv`
   must equal the rows of `agg_mat.csv` **in order** — that ordering is what makes row
   `i` of `S` correspond to column `i` of the data, and the loader asserts it. The
   trailing `Nb` rows of `S` must be an identity block (`S = [A; I]`).
2. Write `configs/datasets/<name>.yaml`:
   - `data.dir` relative to the project root, `h`, `input_size`, `split`,
   - `data.levels`: explicit `[start, stop)` ranges partitioning `[0, Na+Nb)` with no
     gap or overlap. Read them off the actual file; a mistyped boundary fails at load.
   - `features.static` / `features.future`: builder names plus their params.
   - `model` / `train` blocks for anything that differs from `configs/default.yaml`.
3. Only if the bottom-series names are *not* a separator-joined path, add a module in
   `data/features/` with a `@register_feature` builder, and import it from that
   package's `__init__.py` so the decorator runs at import time.
4. Add the name to the expected set in `tests/test_config.py`. The parametrized config
   tests then cover it automatically.

`grouped_static_dummies(keys, sep, positions)` already handles every separator-joined
naming scheme; reach for a new module only when parsing genuinely differs.

## 5. Config

Every configuration object is a frozen dataclass in `config.py` with a
`from_dict` that **rejects unknown keys**. A typo is an error, never a silently ignored
setting. Validation that depends on one section lives in `__post_init__`;
cross-section validation lives in `RunConfig.validate`.

`--set section.key=value` is typed against the dataclass fields, so ablations never
need a config edit and a wrong field name fails immediately.

When you add a config field: give it a default that reproduces current behaviour, and
validate it where it is defined.

## 6. Shape conventions

Written once here so the code does not have to repeat them.

```
B    batch (windows; for a panel, (item, date) pairs)
L    input_size, the history length
H    h, the forecast horizon
Nb   bottom-level series
Na   aggregate series; the hierarchy is Na + Nb rows
K    factor count
N    Monte Carlo samples
```

| tensor | shape |
|---|---|
| `WindowBatch.insample_y` | `[B, L, Nb]`, normalized |
| `WindowBatch.futr_exog` | `[B, F, L+H, Nb]`, shared channels first |
| `WindowBatch.hist_exog` | `[B, X, L, Nb]` |
| `WindowBatch.stat_exog` | `[Nb, S]`, or `[B, Nb, S]` for a panel |
| `FactorParams.mu` / `.sigma` | `[B, H, Nb]` |
| `FactorParams.F` | `[B, H, Nb, K]` |
| `ForecastSamples.bottom` | `[B, H, Nb, N]` |
| `ForecastSamples.hierarchy` | `[B, H, Na+Nb, N]` |
| `Window.insample` / `.target` | `[L, Nb]` / `[H, Nb]`, numpy, raw units |

The sample axis is always trailing. Future channels are always ordered shared-first,
per-series-last, because the batcher re-normalizes only the trailing per-series block.

## 7. Invariants that must not be broken

- **Clip before aggregating.** Coherence needs `S @ relu(bottom)`, not `relu(S @ bottom)`.
  The two differ, and `tests/test_distribution.py` pins the difference.
- **Coherence is structural.** Never post-process forecasts to make them add up;
  aggregating clipped bottom-level samples through `S` is what makes them coherent.
- **Scale from the insample block only.** `window_stats` must never see the horizon.
  Leakage here inflates every reported number.
- **Denormalize the parameters, not the samples.** The factor model is affine, so
  pushing `(loc, scale)` onto `mu`/`sigma`/`F` is exact and keeps the loss in raw units.
- **Score in raw units.** The objective and sCRPS are computed after denormalization,
  on aggregated samples.
- **sCRPS is a ratio of sums**, never a mean of per-level ratios. "Overall" is the
  whole-hierarchy ratio. Accumulating numerator and denominator separately keeps
  chunked evaluation exact.
- **Anchors need `lag >= h`.** A shorter lag feeds the model true future observations.
  `check_anchor_lags` refuses it; do not bypass it.

## 8. Code style

- **No module docstring headers.** Files start with imports. A module's name and its
  contents say what it is; a banner comment goes stale and adds nothing.
- **Comment the "why", never the "what".** If a line needs explaining because it is
  subtle (clip order, a left-only pad, a scaler floor), say why in one line. Do not
  narrate what the code plainly does.
- **Docstrings on public functions and classes**, one line where one line suffices.
  Skip them on obvious properties, `__init__`, and private helpers whose name is clear.
- Type-annotate signatures. `from __future__ import annotations` at the top.
- Raise with a message that names the offending value and the expectation. Every error
  path in this codebase tells the reader what to fix.
- Line length 90, enforced by ruff. `python -m ruff format .` before committing.
- **Record non-obvious modelling choices** — a scaler, a lag, a feature encoding, a
  split convention — next to the choice itself: one line at the code site, or a note in
  [docs/datasets.md](docs/datasets.md) when it is dataset-specific. A choice with no
  record is indistinguishable from a bug six months later.

## 9. Tests

One file per concern, mirroring the package. A change to behaviour should land with the
test that pins it. In particular:

- New registry entry → a test that it loads by name and produces the documented shape.
- New dataset → the parametrized config and coherence tests cover it once it is listed.
- New invariant → a test that fails when it is broken, not one that merely exercises
  the happy path.

Keep the suite fast enough to run on every change; the end-to-end tests use a handful
of steps and a small sample count deliberately.
