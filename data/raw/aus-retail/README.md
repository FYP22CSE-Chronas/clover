# Australian Retail Turnover (`aus_retail`)

Monthly retail turnover (in AUD millions) across 6 Australian states/territories and 6 major non-overlapping retail industry groups, 1982-04-01 – 2018-12-01.

Source: Australian Bureau of Statistics (ABS) via `tsibbledata::aus_retail`

## Hierarchy

Total → State (6) → Major Industry (6 per State)

- Total: 1
- State: 6 (`Australian Capital Territory`, `New South Wales`, `Queensland`, `South Australia`, `Victoria`, `Western Australia`)
- Bottom Series (Industry × State): 36
- Total Series: 43

## Mapping

| Level | Categories |
|---|---|
| **States (6)** | Australian Capital Territory, New South Wales, Queensland, South Australia, Victoria, Western Australia |
| **Major Industries (6)** | Food retailing; Household goods retailing; Clothing, footwear and personal accessory retailing; Department stores; Other retailing; Cafes, restaurants and takeaway food services |

*(Note: `Northern Territory` and `Tasmania` are excluded because they lack complete 1982–2018 coverage across all 6 major industry groups in ABS; sub-industries are excluded to avoid double-counting.)*

## Key Events & Data Notes

- **Exact Coherence:** Reconstructed directly via $y = S \cdot b$. Maximum absolute difference equals `0.00e+00`. Zero missing timestamps, zero NaNs.
- **Cointegration ($I(1)$):** All **36 / 36** bottom series are $I(1)$ (ADF $p > 0.05$ in levels, $p < 0.05$ in first differences), with **66 / 630** pairwise Engle–Granger cointegrated pairs ($p < 0.05$) and strict partial Johansen cointegration ($0 < r < 6$) at the 6-State level (rank 1–2) and within every State's 6-Industry group (rank 1–5).

## Files

`data.csv` — 441 × 43
- Rows: months, date index (`YYYY-MM-DD`), 1982-04-01 to 2018-12-01
- Columns: stringified paths, e.g. `['Total']`, `['New South Wales']`, `['Food retailing', 'New South Wales']`

`agg_mat.csv` — 43 × 36
- Rows = `data.csv` columns, same order
- Columns = 36 bottom `[Industry, State]` series
- Top row: all ones; state rows: 1 for member industries; bottom 36 rows: 36 × 36 identity block
