# CLOVER

Coherent probabilistic forecasting for hierarchical time series.

The network emits `mu`, `sigma` and `K` factor loadings at the bottom level. Samples are
drawn by reparameterization — `mu + sigma * z + F @ eps` — clipped to non-negative, then
aggregated through the fixed aggregation matrix `S`. Every forecast therefore satisfies
the hierarchy constraints by construction, and the CRPS objective is computed on the
aggregated samples, so training optimizes the coherent distribution directly rather than
reconciling afterwards.

## Install

```bash
pip install -r requirements.txt
```

Everything runs from the repository root; there is no package to install.

## Run

```bash
python -m cli --dataset tourism_small --seeds 0 1 2
python -m cli --dataset traffic --seeds 0 --set train.batch_size=32
python -m cli --dataset labour --seeds 0 --set model.sigma_activation=exp
python -m cli --dataset favorita --seeds 0 --threads 4
```

`--set` takes typed dotted paths validated against the dataclass fields, so ablations
never require editing a config file; a path naming no field is an error. Results go to
`--out` (default `results/`): one CSV of mean/std per level, one JSON per seed.

## Structure

```
config.py        every config dataclass + YAML loading
registry.py      name -> factory, one instance per swappable interface
contracts.py     tensor contracts exchanged across layers
cli.py           command line entry point

model/           the model itself: pure torch, no pandas, no I/O
  blocks.py        MLP and dilated causal convolution primitives
  encoders.py      history -> per-series summary
  mixers.py        cross-series mixing over the hierarchy
  decoders.py      context -> one feature vector per horizon
  heads.py         features -> factor-model parameters
  network.py       CLOVER, composing the four above
  distribution.py  reparameterized sampling and coherent aggregation
  normalization.py per-window scaling and its exact inverse

training/        objectives, sCRPS estimators, the training loop
  losses.py  metrics.py  trainer.py

data/            everything data: the loading code and the datasets themselves
  dataset.py       HierarchicalDataset, ExogenousFeatures
  csv_source.py    the data.csv / agg_mat.csv pair loader
  panel.py         PanelDataset, for many hierarchies sharing one S
  favorita.py      the Favorita panel loader
  hierarchy.py     level slices and the masks derived from them
  windows.py       rolling windows and the split strategies
  features/        registered exogenous feature builders
  raw/<name>/      the seven CSV-pair datasets
  favorita/        Favorita CSVs, plus the built panel/

pipeline/        composition root: config -> data -> model -> train -> eval
configs/         default.yaml + one file per dataset
scripts/         one-off data preparation
tests/           124 tests, ~17s, no network
docs/            design-decision records
```

The dependency rule, enforced by [tests/test_layering.py](tests/test_layering.py):

| layer | may import |
|---|---|
| `config`, `registry`, `contracts` | nothing else in the repo |
| `model/` | the leaf modules only |
| `training/` | leaf modules, `model/` |
| `data/` | the leaf modules only |
| `pipeline/` | everything |

`model/` contains no dataset name, column-name parsing, level slice or calendar logic.
`data/` never imports the model and never imports torch. Only `pipeline/` imports both.

Seven of the eight datasets are a single hierarchy over one calendar. Favorita is a
*panel* — 4,036 item hierarchies sharing one 93×54 `S` — handled by
[data/panel.py](data/panel.py) and
[pipeline/panel_runner.py](pipeline/panel_runner.py). The model and
trainer are the same objects either way: a panel window is still one `[L, Nb]` history
and one `[h, Nb]` target, and the batch axis indexes `(item, date)` pairs rather than
dates alone.

## Configure

`configs/datasets/<name>.yaml` merged over `configs/default.yaml`, parsed into frozen
dataclasses. An unknown key raises rather than being silently ignored.

Adding a dataset is one config file, plus one small feature module only if its
bottom-series names need custom parsing — six of the seven single hierarchies need
none. See [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Swap components

Encoder, cross-series mixer, decoder and distribution head each sit behind a small
interface and are selected by config string:

```yaml
model:
  encoder: mlp          # or dilated_conv
  mixer: identity       # or cross_series_mlp
```

Adding an encoder is an `@ENCODERS.register("name")` decorator on an `nn.Module` — no
edit to the model class.

## Test

```bash
python -m pytest
python -m ruff check . && python -m ruff format --check .
```

Covered per layer: the layering rule itself, forward shapes with and without each
exogenous block, exact coherence of sampled forecasts, the disabled mixer as a true
identity passthrough, CRPS against the closed-form Gaussian, normalization round-trip,
split disjointness, the anchor leakage guard, config strictness across all eight
datasets, the panel shape contracts, and end-to-end runs producing a finite sCRPS.

## Data

Code and datasets share one `data/` folder: the loaders sit at its top level, the files
they read sit under `data/raw/<name>/` and `data/favorita/`. Build the Favorita panel
once with:

```bash
python scripts/build_favorita_panel.py
```

See [docs/datasets.md](docs/datasets.md) for what each dataset is and where it came
from.

## Docs

- [IMPLEMENTATION.md](IMPLEMENTATION.md) — conventions to follow when extending this
  codebase.
- [docs/datasets.md](docs/datasets.md) — the eight datasets, their level slices,
  features, splits and provenance.

The model is a plain `torch.nn.Module` with no forecasting-framework dependency.
