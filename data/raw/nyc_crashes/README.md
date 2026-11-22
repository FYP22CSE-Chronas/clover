# NYC Motor Vehicle Collisions

Daily motor vehicle collisions by borough and ZIP code, 8 Oct 2022 – 15 Nov 2026.

Source: https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95

## Hierarchy

Total → Borough (5) → ZIP Code (228)

- Total: 1
- Borough: 5
- ZIP Code: 228
- Series: 234

## Mapping

### Bronx — 25 ZIP codes

| ZIP Codes |
|---|
| 10451, 10452, 10453, 10454, 10455, 10456, 10457, 10458, 10459, 10460, 10461, 10462, 10463, 10464, 10465, 10466, 10467, 10468, 10469, 10470, 10471, 10472, 10473, 10474, 10475 |

### Brooklyn — 39 ZIP codes

| ZIP Codes |
|---|
| 11201, 11203, 11204, 11205, 11206, 11207, 11208, 11209, 11210, 11211, 11212, 11213, 11214, 11215, 11216, 11217, 11218, 11219, 11220, 11221, 11222, 11223, 11224, 11225, 11226, 11228, 11229, 11230, 11231, 11232, 11233, 11234, 11235, 11236, 11237, 11238, 11239, 11249, 11251 |

### Manhattan — 88 ZIP codes

| ZIP Codes |
|---|
| 10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007, 10009, 10010, 10011, 10012, 10013, 10014, 10016, 10017, 10018, 10019, 10020, 10021, 10022, 10023, 10024, 10025, 10026, 10027, 10028, 10029, 10030, 10031, 10032, 10033, 10034, 10035, 10036, 10037, 10038, 10039, 10040, 10041, 10044, 10045, 10048, 10055, 10065, 10069, 10075, 10080, 10103, 10105, 10106, 10107, 10110, 10111, 10112, 10115, 10118, 10119, 10120, 10121, 10122, 10123, 10128, 10152, 10153, 10154, 10155, 10158, 10162, 10165, 10167, 10168, 10169, 10170, 10171, 10172, 10173, 10174, 10176, 10177, 10178, 10179, 10271, 10278, 10279, 10280, 10281, 10282 |

### Queens — 65 ZIP codes

| ZIP Codes |
|---|
| 11001, 11004, 11005, 11040, 11101, 11102, 11103, 11104, 11105, 11106, 11109, 11354, 11355, 11356, 11357, 11358, 11359, 11360, 11361, 11362, 11363, 11364, 11365, 11366, 11367, 11368, 11369, 11370, 11371, 11372, 11373, 11374, 11375, 11377, 11378, 11379, 11385, 11411, 11412, 11413, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11422, 11423, 11426, 11427, 11428, 11429, 11430, 11432, 11433, 11434, 11435, 11436, 11691, 11692, 11693, 11694, 11697 |

### Staten Island — 11 ZIP codes

| ZIP Codes |
|---|
| 10301, 10302, 10304, 10305, 10306, 10307, 10308, 10309, 10310, 10312, 10314 |

## Key Events & Data Notes

- **Post-COVID Traffic Shifts:** Commuter travel adjustments and hybrid work policies established new baseline crash volumes.
- **Severe Weather Dips:** Winter blizzards and extreme rainfall produce sharp single-day collision drops citywide.
- **Micro-ZIP Sparsity:** Low-traffic commercial ZIPs (e.g., individual office towers in Manhattan) naturally have zero-crash days.

## Files

`data.csv` — 1500 × 234
- Rows: days, date index (`YYYY-MM-DD`), 2022-10-08 to 2026-11-15
- Columns: stringified paths, e.g. `['10001', 'Manhattan']`, `['Manhattan']`, `['Total']`

`agg_mat.csv` — 234 × 228
- Rows = `data.csv` columns, same order
- Columns = 228 ZIP code series
- Total row: all ones; borough rows: 1 for member ZIP codes; identity block for ZIP codes
