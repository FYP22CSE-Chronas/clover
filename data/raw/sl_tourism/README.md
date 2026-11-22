# Sri Lanka Tourism Arrivals

Monthly tourist arrivals, Jan 2018 – May 2026.

Source: https://www.kaggle.com/datasets

## Hierarchy

Total → Continent (6) → Region (16) → Country (200)

- Total: 1
- Continent: 6
- Region: 16
- Country: 200
- Series: 223

## Mapping

### Asia — 5 regions, 47 countries

| Region | Countries |
|---|---|
| South Asia (7) | Afghanistan, Bangladesh, Bhutan, India, Maldives, Nepal, Pakistan |
| Southeast Asia (11) | Brunei Darussalam, Cambodia, Indonesia, Laos, Malaysia, Myanmar, Philippines, Singapore, Thailand, Timor-Leste, Vietnam |
| East Asia (6) | China, Hong Kong, Japan, Mongolia, South Korea, Taiwan |
| Middle East (15) | Bahrain, Iran, Iraq, Israel, Jordan, Kuwait, Lebanon, Oman, Palestinian Territories, Qatar, Saudi Arabia, Syrian Arab Republic, Turkey, United Arab Emirates, Yemen |
| Central Asia & Caucasus (8) | Armenia, Azerbaijan, Georgia, Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, Uzbekistan |

### Europe — 4 regions, 47 countries

| Region | Countries |
|---|---|
| Western Europe (9) | Austria, Belgium, France, Germany, Liechtenstein, Luxembourg, Monaco, Netherlands, Switzerland |
| Northern Europe (11) | Denmark, Estonia, Finland, Iceland, Ireland, Isle Of Man, Latvia, Lithuania, Norway, Sweden, United Kingdom |
| Southern Europe (17) | Albania, Andorra, Bosnia And Herzegovina, Croatia, Cyprus, Greece, Italy, Kosovo, Malta, Montenegro, North Macedonia, Portugal, San Marino, Serbia, Slovenia, Spain, Vatican City |
| Eastern Europe & CIS (10) | Belarus, Bulgaria, Czech Republic, Hungary, Moldova, Poland, Romania, Russian Federation, Slovakia, Ukraine |

### Africa — 2 regions, 53 countries

| Region | Countries |
|---|---|
| North Africa (6) | Algeria, Egypt, Libya, Morocco, Sudan, Tunisia |
| Sub-Saharan Africa (47) | Angola, Benin, Botswana, Burkina Faso, Burundi, Cameroon, Cape Verde, Central African Republic, Chad, Comoros, Cote D'Ivoire, Democratic Republic Of Congo, Djibouti, Equatorial Guinea, Eritrea, Eswatini, Ethiopia, Gabon, Gambia, Ghana, Guinea-Bissau, Kenya, Lesotho, Liberia, Madagascar, Malawi, Mali, Mauritania, Mauritius, Mozambique, Namibia, Niger, Nigeria, Republic Of Congo, Rwanda, Sao Tome And Principe, Senegal, Seychelles, Sierra Leone, Somalia, South Africa, South Sudan, Tanzania, Togo, Uganda, Zambia, Zimbabwe |

### North America — 3 regions, 26 countries

| Region | Countries |
|---|---|
| Northern America (3) | Canada, Mexico, United States |
| Central America (7) | Belize, Costa Rica, El Salvador, Guatemala, Honduras, Nicaragua, Panama |
| Caribbean (16) | Antigua And Barbuda, Aruba, Bahamas, Barbados, Cuba, Dominica, Dominican Republic, Grenada, Haiti, Jamaica, Netherlands Antilles, Saint Barthelemy, Saint Kitts And Nevis, Saint Lucia, Saint Vincent And The Grenadines, Trinidad And Tobago |

### South America — 1 region, 12 countries

| Region | Countries |
|---|---|
| South America (12) | Argentina, Bolivia, Brazil, Chile, Colombia, Ecuador, Guyana, Paraguay, Peru, Suriname, Uruguay, Venezuela |

### Oceania — 1 region, 15 countries

| Region | Countries |
|---|---|
| Oceania (15) | Australia, Fiji, Guam, Kiribati, Marshall Islands, Micronesia, Nauru, New Zealand, Palau, Papua New Guinea, Samoa, Solomon Islands, Tonga, Tuvalu, Vanuatu |

## Key Events & Data Notes

- **Easter Attacks (Apr 2019):** Severe arrivals slump following hotel and church bombings.
- **COVID-19 Border Closure (Apr – Nov 2020):** Total arrivals = 0 due to nationwide commercial airport shutdown.
- **Pilot Bio-Bubble (Dec 2020):** Limited reopening with 393 Ukrainian charter tourist arrivals.
- **Economic Crisis (2022):** Fuel and power shortages caused a temporary recovery dip.
- **Micro-Market Zeros:** Remote island states (e.g. Nauru, Vatican) legitimately record 0 arrivals in most months.

## Files

`data.csv` — 101 × 223
- Rows: months, date index (`YYYY-MM-DD`), Jan 2018 – May 2026
- Columns: stringified paths, e.g. `['India', 'South Asia', 'Asia']`, `['Total']`

`agg_mat.csv` — 223 × 200
- Rows = `data.csv` columns, same order
- Columns = 200 countries
- Total row, one row per continent, one per region, then identity block
