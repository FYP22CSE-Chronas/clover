# Simulated Groups (`sim_groups`)

Artificially generated daily cointegrated hierarchy with 4 groups and 4 bottom series per group, Jan 2021 – Sep 2023.

## Hierarchy

Total → Group (4) → Bottom (16)

- Total: 1
- Group: 4 (`G1`–`G4`)
- Bottom: 16 (`G1/S1`–`G4/S4`)
- Series: 21

## Data Notes

- **Artificially Generated:** Synthetic daily cointegrated system designed to test long-run equilibrium and cross-series dependence modeling across groups and bottom series.

## Files

`data.csv` — 1000 × 21
- Rows: days, `date` column (`YYYY-MM-DD`), 2021-01-01 to 2023-09-27
- Columns: `Total`, `G1`–`G4`, `G1/S1`–`G4/S4`

`agg_mat.csv` — 21 × 16
- Rows = `data.csv` columns, same order
- Columns = 16 bottom series (`G1/S1`–`G4/S4`)
- Total row: all ones; group rows: 1 for member series; identity block for bottom series
