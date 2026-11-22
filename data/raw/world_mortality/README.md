# World Mortality — All-Cause Deaths

Monthly all-cause mortality across 128 countries, Jan 2015 – Dec 2024.

Source: https://github.com/akarlinsky/world_mortality

## Hierarchy

Total → Continent (6) → Country (128)

- Total: 1
- Continent: 6
- Country: 128
- Series: 135

## Mapping

### Africa — 11 countries

| Countries |
|---|
| Algeria, Botswana, Cabo Verde, Egypt, Mauritius, Mayotte, Namibia, Réunion, Seychelles, South Africa, Tunisia |

### Asia — 29 countries

| Countries |
|---|
| Azerbaijan, Bhutan, Brunei, Cyprus, Hong Kong, Iran, Israel, Japan, Jordan, Kazakhstan, Kuwait, Kyrgyzstan, Lebanon, Macao, Malaysia, Maldives, Mongolia, Oman, Palestine, Philippines, Qatar, Singapore, South Korea, Taiwan, Tajikistan, Thailand, Turkey, United Arab Emirates, Uzbekistan |

### Europe — 49 countries

| Countries |
|---|
| Albania, Andorra, Armenia, Austria, Belarus, Belgium, Bosnia, Bulgaria, Croatia, Czechia, Denmark, Estonia, Faroe Islands, Finland, France, Georgia, Germany, Gibraltar, Greece, Hungary, Iceland, Ireland, Italy, Kosovo, Latvia, Liechtenstein, Lithuania, Luxembourg, Malta, Moldova, Monaco, Montenegro, Netherlands, North Macedonia, Norway, Poland, Portugal, Romania, Russia, San Marino, Serbia, Slovakia, Slovenia, Spain, Sweden, Switzerland, Transnistria, Ukraine, United Kingdom |

### North America — 23 countries

| Countries |
|---|
| Antigua and Barbuda, Aruba, Bahamas, Barbados, Belize, Bermuda, Canada, Costa Rica, Cuba, Dominican Republic, El Salvador, Greenland, Guadeloupe, Guatemala, Jamaica, Martinique, Mexico, Nicaragua, Panama, Puerto Rico, Saint Kitts and Nevis, Saint Vincent and the Grenadines, United States |

### Oceania — 5 countries

| Countries |
|---|
| Australia, Fiji, French Polynesia, New Caledonia, New Zealand |

### South America — 11 countries

| Countries |
|---|
| Argentina, Bolivia, Brazil, Chile, Colombia, Ecuador, French Guiana, Paraguay, Peru, Suriname, Uruguay |

## Key Events & Data Notes

- **COVID-19 Pandemic (2020–2022):** Global mortality surges with peak excess deaths in 2020–2021 (Alpha/Delta waves).
- **Post-Pandemic Normalization (2023–2024):** Stabilized back towards pre-pandemic demographic levels.
- **Weekly-to-Monthly:** 52 ISO weekly countries converted via daily apportionment ($1/7$ per day).
- **Seasonal Imputation:** Missing periods across 38 countries filled via seasonal median decomposition.
- **Boundary Cutoffs:** Cutoff months with < 20 days imputed to prevent partial-month dips.

## Files

`data.csv` — 120 × 135
- Rows: months, date index (YYYY-MM-DD), 2015-01-01 to 2024-12-01
- Columns: stringified paths, e.g. `['Japan', 'Asia']`, `['Asia']`, `['Total']`

`agg_mat.csv` — 135 × 128
- Rows = `data.csv` columns, same order
- Columns = 128 country series
- Total row: all ones; continent rows: 1 for member countries; identity block for countries
