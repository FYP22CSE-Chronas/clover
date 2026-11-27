# Experiment 04: Pi = -L L^T + HS-VP 9-Dataset Benchmark

Complete per-level results from `results/bench_3ds/` and `results/bench_6ds/` (duplicated `bench6/prison_base` omitted).

## Part A: New & Synthetic Datasets (`results/bench_3ds/`)

| Arm | Dataset | Seeds | Overall | Total | State | Industry/State |
|---|---|---:|---:|---:|---:|---:|
| `aus_retail_base` | `aus_retail` | 5 | 0.01915 ± 0.00126 | 0.01243 ± 0.00124 | 0.01680 ± 0.00145 | 0.02821 ± 0.00145 |
| `aus_retail_pi_crps` | `aus_retail` | 5 | 0.01879 ± 0.00063 | 0.01216 ± 0.00066 | 0.01643 ± 0.00065 | 0.02778 ± 0.00092 |
| `aus_retail_pi_hs_vp` | `aus_retail` | 5 | 0.02152 ± 0.00185 | 0.01337 ± 0.00232 | 0.01835 ± 0.00187 | 0.03284 ± 0.00140 |

| Arm | Dataset | Seeds | Overall | Total | Group | Bottom |
|---|---|---:|---:|---:|---:|---:|
| `sim_groups_base` | `sim_groups` | 5 | 0.01709 ± 0.00062 | 0.00794 ± 0.00087 | 0.01710 ± 0.00081 | 0.02622 ± 0.00085 |
| `sim_groups_pi_hs_vp` | `sim_groups` | 5 | 0.01646 ± 0.00072 | 0.00695 ± 0.00091 | 0.01652 ± 0.00134 | 0.02590 ± 0.00113 |

| Arm | Dataset | Seeds | Overall | Total | Region | Gender/Region | Status/Gender/Region |
|---|---|---:|---:|---:|---:|---:|---:|
| `sim_labour_base` | `sim_labour` | 5 | 0.00504 ± 0.00017 | 0.00212 ± 0.00026 | 0.00280 ± 0.00025 | 0.00345 ± 0.00021 | 0.01181 ± 0.00018 |
| `sim_labour_pi_hs_vp` | `sim_labour` | 5 | 0.00495 ± 0.00018 | 0.00184 ± 0.00015 | 0.00252 ± 0.00027 | 0.00320 ± 0.00026 | 0.01224 ± 0.00024 |

| Arm | Dataset | Seeds | Overall | Total | Sex | AgeGroup |
|---|---|---:|---:|---:|---:|---:|
| `us_deaths_base` | `us_deaths` | 5 | 0.01968 ± 0.00253 | 0.01806 ± 0.00259 | 0.01850 ± 0.00253 | 0.02247 ± 0.00252 |
| `us_deaths_pi_crps` | `us_deaths` | 5 | 0.01939 ± 0.00262 | 0.01760 ± 0.00246 | 0.01813 ± 0.00263 | 0.02245 ± 0.00280 |
| `us_deaths_pi_hs_vp` | `us_deaths` | 5 | 0.02382 ± 0.00422 | 0.02096 ± 0.00440 | 0.02168 ± 0.00441 | 0.02881 ± 0.00392 |

## Part B: Public CLOVER Benchmark Datasets (`results/bench_6ds/`)

| Arm | Dataset | Seeds | Overall | Total | Crime | Beat | Street | ZIP | Crime/Beat/Street/ZIP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `police_base` | `police` | 5 | 0.34695 ± 0.00488 | 0.08521 ± 0.01083 | 0.20540 ± 0.00476 | 0.41960 ± 0.00262 | 0.15094 ± 0.00790 | 0.39217 ± 0.00275 | 0.82840 ± 0.00096 |
| `police_pi_crps` | `police` | 5 | 0.34836 ± 0.00488 | 0.08710 ± 0.01121 | 0.20718 ± 0.00537 | 0.42055 ± 0.00206 | 0.15300 ± 0.00832 | 0.39314 ± 0.00227 | 0.82920 ± 0.00071 |
| `police_pi_hs_vp` | `police` | 5 | 0.37453 ± 0.00427 | 0.07737 ± 0.00222 | 0.21203 ± 0.00291 | 0.45134 ± 0.00391 | 0.14712 ± 0.00144 | 0.42116 ± 0.00303 | 0.93820 ± 0.01431 |

| Arm | Dataset | Seeds | Overall | Total | State | Gender | Legal | State/Gender/Legal |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `prison_base` | `prison` | 5 | 0.04560 ± 0.00266 | 0.02653 ± 0.00294 | 0.05072 ± 0.00365 | 0.02704 ± 0.00285 | 0.05090 ± 0.00157 | 0.07280 ± 0.00274 |
| `prison_pi_hs_vp` | `prison` | 5 | 0.04379 ± 0.00319 | 0.02236 ± 0.00214 | 0.04701 ± 0.00416 | 0.02316 ± 0.00216 | 0.05201 ± 0.00333 | 0.07439 ± 0.00639 |

| Arm | Dataset | Seeds | Overall | Country | State | Zone | Region | Country/Purpose | State/Purpose | Zone/Purpose | Region/Purpose |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `tourism_large_base` | `tourism_large` | 5 | 0.12858 ± 0.00254 | 0.05320 ± 0.00595 | 0.07405 ± 0.00394 | 0.11446 ± 0.00324 | 0.15634 ± 0.00140 | 0.07504 ± 0.00477 | 0.11450 ± 0.00262 | 0.18568 ± 0.00191 | 0.25540 ± 0.00257 |
| `tourism_large_pi_hs_vp` | `tourism_large` | 5 | 0.13636 ± 0.00263 | 0.04360 ± 0.00522 | 0.07626 ± 0.00471 | 0.12392 ± 0.00486 | 0.16873 ± 0.00392 | 0.07190 ± 0.00322 | 0.12435 ± 0.00395 | 0.20419 ± 0.00446 | 0.27793 ± 0.00918 |

| Arm | Dataset | Seeds | Overall | Country | Country/Purpose | Country/Purpose/State | Country/Purpose/State/CityNonCity |
|---|---|---:|---:|---:|---:|---:|---:|
| `tourism_small_base` | `tourism_small` | 5 | 0.06687 ± 0.00267 | 0.02678 ± 0.00358 | 0.04865 ± 0.00590 | 0.08327 ± 0.00188 | 0.10879 ± 0.00257 |
| `tourism_small_pi_crps` | `tourism_small` | 5 | 0.06672 ± 0.00294 | 0.02497 ± 0.00421 | 0.04861 ± 0.00414 | 0.08353 ± 0.00210 | 0.10978 ± 0.00398 |
| `tourism_small_pi_hs_vp` | `tourism_small` | 5 | 0.07163 ± 0.00422 | 0.02384 ± 0.00523 | 0.05576 ± 0.00359 | 0.09095 ± 0.00434 | 0.11597 ± 0.00466 |

| Arm | Dataset | Seeds | Overall | Total | Halves | Quarters | Lanes |
|---|---|---:|---:|---:|---:|---:|---:|
| `traffic_base` | `traffic` | 5 | 0.02251 ± 0.00346 | 0.01011 ± 0.00484 | 0.01195 ± 0.00436 | 0.01539 ± 0.00353 | 0.05261 ± 0.00166 |
| `traffic_pi_hs_vp` | `traffic` | 5 | 0.02001 ± 0.00134 | 0.00484 ± 0.00124 | 0.00663 ± 0.00131 | 0.01164 ± 0.00158 | 0.05693 ± 0.00201 |

| Arm | Dataset | Seeds | Overall | Total | Language | Language/Access | Language/Access/Agent | Article |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `wiki2_base` | `wiki2` | 5 | 0.24832 ± 0.01362 | 0.09913 ± 0.02713 | 0.18897 ± 0.01177 | 0.26605 ± 0.01229 | 0.27235 ± 0.01188 | 0.41513 ± 0.01251 |
| `wiki2_pi_hs_vp` | `wiki2` | 5 | 0.25344 ± 0.00873 | 0.09296 ± 0.00706 | 0.18141 ± 0.00775 | 0.27465 ± 0.01019 | 0.28171 ± 0.01046 | 0.43647 ± 0.01919 |
