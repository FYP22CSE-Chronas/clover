# Experiment 02: Early Error-Correction & Johansen Beta Experiments

Complete per-level results from `results/ec/`, `results/synth/`, and `results/g/`.

## Part A: Labour EC-CLOVER (`results/ec/`)

| Arm | Dataset | Seeds | Overall | Total | Geography | Gender/Geography | Status/Gender/Geography |
|---|---|---:|---:|---:|---:|---:|---:|
| `base` | `labour` | 5 | 0.00813 ± 0.00035 | 0.00446 ± 0.00046 | 0.00622 ± 0.00034 | 0.00823 ± 0.00028 | 0.01362 ± 0.00039 |
| `ec_joh` | `labour` | 5 | 0.00819 ± 0.00041 | 0.00453 ± 0.00049 | 0.00629 ± 0.00040 | 0.00828 ± 0.00034 | 0.01367 ± 0.00045 |
| `ec_rand` | `labour` | 5 | 0.00828 ± 0.00036 | 0.00469 ± 0.00039 | 0.00641 ± 0.00036 | 0.00837 ± 0.00030 | 0.01367 ± 0.00052 |

## Part B: Synthetic EC-CLOVER (`results/synth/`)

| Arm | Dataset | Seeds | Overall | Total | Group | Bottom |
|---|---|---:|---:|---:|---:|---:|
| `base` | `synthetic_coint` | 5 | 0.01205 ± 0.00062 | 0.00531 ± 0.00078 | 0.00871 ± 0.00057 | 0.02214 ± 0.00119 |
| `base_h1` | `synthetic_coint` | 5 | 0.00734 ± 0.00099 | 0.00391 ± 0.00145 | 0.00754 ± 0.00057 | 0.01056 ± 0.00115 |
| `ec_joh` | `synthetic_coint` | 5 | 0.01237 ± 0.00060 | 0.00608 ± 0.00111 | 0.00927 ± 0.00078 | 0.02175 ± 0.00030 |
| `ec_rand` | `synthetic_coint` | 5 | 0.01255 ± 0.00041 | 0.00589 ± 0.00091 | 0.00937 ± 0.00086 | 0.02239 ± 0.00097 |

## Part C: Exploratory Rank Sweep (`results/g/`)

| Arm | Dataset | Seeds | Overall | Total | Geography | Gender/Geography | Status/Gender/Geography |
|---|---|---:|---:|---:|---:|---:|---:|
| `lab_base` | `labour` | 5 | 0.00813 ± 0.00035 | 0.00446 ± 0.00046 | 0.00622 ± 0.00034 | 0.00823 ± 0.00028 | 0.01362 ± 0.00039 |
| `lab_r0` | `labour` | 5 | 0.00781 ± 0.00027 | 0.00389 ± 0.00063 | 0.00590 ± 0.00036 | 0.00802 ± 0.00021 | 0.01345 ± 0.00012 |
| `lab_r4` | `labour` | 5 | 0.00784 ± 0.00032 | 0.00384 ± 0.00054 | 0.00598 ± 0.00035 | 0.00810 ± 0.00024 | 0.01344 ± 0.00021 |

| Arm | Dataset | Seeds | Overall | Total | Group | Bottom |
|---|---|---:|---:|---:|---:|---:|
| `syn_base` | `synthetic_coint` | 5 | 0.01205 ± 0.00062 | 0.00531 ± 0.00078 | 0.00871 ± 0.00057 | 0.02214 ± 0.00119 |
| `syn_r0` | `synthetic_coint` | 5 | 0.01239 ± 0.00109 | 0.00636 ± 0.00154 | 0.00925 ± 0.00133 | 0.02154 ± 0.00068 |
| `syn_r8` | `synthetic_coint` | 5 | 0.01284 ± 0.00112 | 0.00702 ± 0.00154 | 0.00971 ± 0.00137 | 0.02177 ± 0.00066 |
