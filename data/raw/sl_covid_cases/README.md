# COVID-19 Cases — Sri Lanka

Daily new confirmed COVID-19 cases by district, 2 Apr 2020 – 18 May 2021.

Source: https://www.kaggle.com/datasets/asheniranga/sri-lanka-covid19-dataset

## Hierarchy

Total → Province (9) → District (25)

- Total: 1
- Province: 9
- District: 25
- Series: 35

## Mapping

| Province | Districts |
|---|---|
| Western Province | Colombo, Gampaha, Kalutara |
| Central Province | Kandy, Matale, Nuwara Eliya |
| Southern Province | Galle, Matara, Hambantota |
| Northern Province | Jaffna, Kilinochchi, Mannar, Vavuniya, Mullaitivu |
| Eastern Province | Trincomalee, Batticaloa, Ampara |
| North Western Province | Kurunegala, Puttalam |
| North Central Province | Anuradhapura, Polonnaruwa |
| Uva Province | Badulla, Monaragala |
| Sabaragamuwa Province | Ratnapura, Kegalle |

## Key Events & Data Notes

- **Wave 1 & Spread Lag (Apr – Jul 2020):** Cases centered in Colombo/Gampaha; rural northern districts stayed at 0 until late 2020.
- **Minuwangoda / Peliyagoda Cluster (Oct 2020):** Factory outbreak ignited nationwide community transmission.
- **New Year Surge (Apr – May 2021):** Alpha/Delta variant spread caused exponential daily growth (>2,000 cases/day).
- **Temporal Disaggregation (Apr – 14 Nov 2020):** Pre-November gap reconstructed using Johns Hopkins University (JHU) national daily series apportioned by confirmed district outbreak shares.
- **Incident Daily Cases:** Monotonically reconciled daily counts; 0 NaNs, 0 negative values.

## Files

`data.csv` — 411 × 35
- Rows: days, date index (`YYYY-MM-DD`), 2020-04-02 to 2021-05-18
- Columns: stringified paths, e.g. `['Colombo', 'Western Province']`, `['Total']`

`agg_mat.csv` — 35 × 25
- Rows = `data.csv` columns, same order
- Columns = 25 districts
- Total row, one row per province, then identity block
