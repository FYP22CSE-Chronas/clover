# Datasets

Eight hierarchies. Seven are a single hierarchy over one calendar, stored as a pair of
CSVs under `data/raw/<name>/`; Favorita is a panel and lives under `data/favorita/`.

- `data.csv` — `[T, Na+Nb]`, dated index, one column per hierarchy series.
- `agg_mat.csv` — `[Na+Nb, Nb]`, the aggregation matrix `S`, in `[A; I]` form.

`data/csv_source.py` asserts `data.columns == agg_mat.index`. That ordering is the whole
contract: it is what makes row `i` of `S` correspond to column `i` of the data. It also
checks the trailing `Nb` rows of `S` are an identity block, and `verify_coherence`
confirms `S @ bottom` reproduces every aggregate column.

## Inventory

| dataset | T | Na+Nb | Nb | h | L | split | levels |
|---|---|---|---|---|---|---|---|
| tourism_small | 36 | 89 | 56 | 4 | 8 | last_h | 4 |
| labour | 503¹ | 57 | 32 | 12 | 36 | last_h | 4 |
| tourism_large | 228 | 555 | 304 | 12 | 36 | last_h | 8 |
| wiki2 | 366 | 199 | 150 | 7 | 28 | last_h | 5 |
| traffic² | 267 | 207 | 200 | 1 | 56 | block 120/120/27 | 4 |
| prison | 48 | 45 | 32 | 8 | 8 | last_h | 5 |
| police | 334 | 676 | 500 | 30 | 56 | last_h | 6 |
| favorita³ | 1688 | 93 | 54 | 34 | 96 | block 1620/34/34 | 4 |

¹ after `end_date: 2019-12-31` truncation; the file itself has 514 rows. The trailing
rows are the COVID-19 collapse, excluded so the series stays representative.

² rebuilt from raw UCI PEMS-SF, which holds 267 distinct days. A commonly distributed
version of this file has 366 rows, but the extra 99 are a verbatim replay of days 0..98
dated as if contiguous from 2008-01-01, which gets 11% of the weekdays right. This is
the same 200 lanes and the same `S` with the replay dropped and the real calendar
restored.

³ a *panel*: 4,036 grocery items, each an independent hierarchy over the same 54 stores,
so its row above describes one item's hierarchy. Totals are 217,944 bottom series and
375,348 series overall. See [Favorita](#favorita) below.

## Level slices

Level slices live in each dataset's config, as explicit `[start, stop)` row ranges.
`validate_level_slices` requires them to partition `[0, Na+Nb)` with no gap or overlap,
so a mistyped boundary fails at load time.

They are explicit ranges rather than derived, because deriving them is only reliable for
datasets whose column names encode depth uniformly. `data/hierarchy.py` ships
`levels_from_name_depth` for the cases where it does work; no config uses it, because an
explicit range with a source note is easier to check against the data than a parser is.

Each range was read off the actual file, not assumed.

**tourism_small** — Total (1), Purpose (4), State×Purpose (28), bottom (56). Names like
`nsw-hol-city` encode state and purpose directly.

**labour** — column names are Python-list-like strings encoding each path, e.g.
`['Employed full-time', 'Females', 'New South Wales']`. Depth 0 is Total (1), depth 1
geography (8), depth 2 gender×geography (16), depth 3 bottom (32).

**tourism_large** — the only *grouped* rather than strictly nested hierarchy here:
geography and purpose cross, giving eight level blocks. Codes are a geographic prefix
plus a purpose suffix: `TotalAll`, `AAll` (state), `AAAll` (zone), `AAAAll` (region),
`AAAHol` (a bottom series). Each of the eight blocks is one unbroken run of columns; row
sums per block are 304/56/8/4/76/14/2/1.

**wiki2** — underscore-joined path: `de_AAC_AAG_001` is language, access, agent, article.
Depths 0–4 give 1/6/18/24/150.

**traffic** — verified against `S` itself rather than names: the `Total` row sums to 200,
`y1`/`y2` to 100 each, `y11`..`y22` to 50 each, then a 200×200 identity block.

**prison** — Total (1) → State (8) | Gender (2) | Legal (2) → bottom (32) = 45. The
source ships only bottom series and three grouping keys, so `S` was constructed. Note a
crossed reading that also includes the three pairwise interactions gives 81 series
instead; switching means regenerating `agg_mat.csv` and the config levels together.

**police** — Total (1) → Crime (19) | Beat (79) | Street (10) | ZIP (67) → bottom (500)
= 676, constructed the same way as prison.

## Features

A dataset config lists the builders it wants. There is no dataset name anywhere in the
feature code.

| builder | kind | used by |
|---|---|---|
| `quarter_dummies` | future, shared | tourism_small, prison |
| `month_dummies` | future, shared | tourism_large |
| `dayofweek_dummies` | future, shared | wiki2, police, traffic |
| `weekend_indicator` | future, shared | traffic |
| `saturday_proximity` | future, shared | traffic |
| `seasonal_naive_anchor` | future, per-series | all but labour |
| `grouped_static_dummies` | static | tourism_small, wiki2, prison, police |
| `identity_dummies` | static | traffic |
| `tourism_large_static_dummies` | static | tourism_large |

`grouped_static_dummies(keys, sep, positions)` covers every name layout that is a
separator-joined path:

| dataset | example name | sep | positions |
|---|---|---|---|
| tourism_small | `nsw-hol-city` | `-` | `[0]` (state only) |
| wiki2 | `de_AAC_AAG_001` | `_` | `[0,1,2]` |
| prison | `NSW-Male-Sentenced` | `-` | `[0,1,2]` |
| police | `13A-10H30-AVE-77002` | `-` | `[0,1,2,3]` |

Only **tourism_large** needs a dedicated builder, because its keys are a prefix and a
suffix rather than fields between separators. Categories are always read off the data,
never hard-coded, so a regenerated file with different levels stays correct.

Anchors are produced in raw units and back-filled with each series' first observed value
for out-of-range lags. Zero-filling would read as "this series was at zero a year ago" —
a large fabricated seasonal swing the earliest training windows would have to learn to
ignore. The pipeline re-normalizes the anchor channels with each window's own
`(loc, scale)`; the calendar channels are left untouched, which is why future channels
are ordered shared-first, per-series-last.

**Anchor lags must be `>= h`.** A shorter lag feeds the model true future observations
over the horizon. `check_anchor_lags` refuses it, which is why police uses lag 35
against `h=30` and prison uses 8 and 12 against `h=8`.

## Splits

`last_h` — test is the final `h` steps, validation the `h` before it, training every
window whose target clears both.

`block` — fixed chronological row counts, used by traffic and favorita. Validation is
*many* rolling windows (a single `h=1` window is far too thin an early-stopping signal);
test is the single window ending on the last row.

## <a id="favorita"></a>Favorita

The only panel dataset, and the only one whose raw form is large.

**Shape.** 4,036 items × 54 stores of daily grocery sales, 2013-01-01 to 2017-08-15
(1,688 days). Each item carries its own copy of one store geography — 1 national,
16 states, 22 cities, 54 stores = 93 rows — so `S` is 93×54 and is shared across every
item rather than block-diagonal over all 217,944 bottom series. A dense `S` over the
whole panel would be 375,348 × 217,944; the shared form is 93 × 54.

That shape is why `data/panel.py` exists. `HierarchicalDataset` is `[T, Na+Nb]`, one
hierarchy over one calendar; `PanelDataset` is `[n_groups, T, Nb]`. The model and the
trainer are untouched by the distinction — a panel window is still one `[L, Nb]` history
and one `[h, Nb]` target, and the batch axis simply indexes `(item, date)` pairs.

**Split.** The last 34 days are test (2017-07-13 to 2017-08-15) and the 34 before them
validation, leaving days 96..1586 as training forecast-creation dates — 6,017,676
training windows, of which an 80,000-step run at batch 4 sees about 5%.

**Features.** Item perishability and store state dummies (static), past unit sales and
store transactions (historical), promotions and day of week (future-known). Because
perishability varies by item and state dummies by store, the static block is
`[B, Nb, S]` rather than the shared `[Nb, S]` the other datasets use.

**Layout.**

```
data/favorita/
  train.csv              4.7 GB, streamed, never loaded whole
  stores.csv items.csv transactions.csv holidays_events.csv oil.csv
  panel/                 built artifacts, memmapped by the loader
    sales.npy            [4036, 1688, 54] float32
    promo.npy            [4036, 1688, 54] int8
    transactions.npy     [1688, 54] float32
    meta.npz             S, level bounds, calendar, static blocks
```

`configs/datasets/favorita.yaml` points `data.dir` at `data/favorita/panel`. Build it
once with `python scripts/build_favorita_panel.py`; absent those four files the loader
raises rather than half-working. `train.csv` and `panel/` are gitignored — they are
6.5 GB together — but everything needed to regenerate them is tracked.

Fill conventions: absent sales rows mean no sale (zero-filled); promotions are forward-
then back-filled along the date axis.

## Adding a dataset

1. Drop `data.csv` and `agg_mat.csv` into `data/raw/<name>/`, columns matching rows.
2. Write `configs/datasets/<name>.yaml`: `dir`, `h`, `input_size`, split, level ranges,
   feature list, architecture widths, training schedule.
3. Only if its bottom-series names are not a separator-joined path, add a module under
   `data/features/` registering a static builder, and import it from that package's
   `__init__.py` so the decorator runs.

No change to `model/`, and no `if name == ...` anywhere. See
[IMPLEMENTATION.md](../IMPLEMENTATION.md) for the full recipe.
