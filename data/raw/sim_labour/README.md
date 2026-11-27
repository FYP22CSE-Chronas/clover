# Simulated Labour (`sim_labour`)

Artificially generated monthly employment hierarchy mirroring the Australian Labour structure, Jan 1980 – Dec 2019.

## Hierarchy

Total → Region (4) → Gender × Region (8) → Status × Gender × Region (16)

- Total: 1
- Region: 4 (`Region0`–`Region3`)
- Gender/Region: 8 (`Females`, `Males` per region)
- Status/Gender/Region: 16 (`Employed full-time`, `Employed part-time` per gender/region)
- Series: 29

## Data Notes

- **Artificially Generated:** Synthetic cointegrated hierarchy with log random-walk group totals and AR(1) logit full-time/part-time shares, producing negative innovation correlation between full-time and part-time siblings while keeping group sums steady.

## Files

`data.csv` — 480 × 29
- Rows: months, date index (`YYYY-MM-DD`), 1980-01-01 to 2019-12-01
- Columns: stringified paths, e.g. `['Employed full-time', 'Females', 'Region0']`, `['Region0']`, `['Total']`

`agg_mat.csv` — 29 × 16
- Rows = `data.csv` columns, same order
- Columns = 16 bottom status/gender/region series
- Total row: all ones; region and gender/region rows: 1 for member series; identity block for bottom series
