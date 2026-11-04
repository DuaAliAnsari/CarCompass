# Data quality report

| Step | Rows |
|---|---|
| Rows read | 151875 |
| Dropped: missing required values | 4763 |
| Dropped: out of range | 140 |
| Dropped: duplicates | 3780 |
| Dropped: low mileage | 9589 |
| Dropped: rare make/model | 466 |
| City grouped as 'Other' | 425 |
| Engine set to missing | 230 |
| **Rows kept** | **133137** |
| Kept: pre-2010 cars | 42431 |
| Kept: 2010 and later | 90706 |
| Train rows | 106596 |
| Test rows | 26541 |

Seed: 42
Pandera schema check: passed

## Sources

| Source | Rows read | Rows kept | Test rows |
|---|---|---|---|
| pakwheels_2022 | 89956 | 74552 | 14982 |
| pakistan_used_cars | 61919 | 58585 | 11559 |

## Price agreement: pakwheels_2022 vs pakistan_used_cars

- make/model/year groups compared: 1145
- median price ratio (pakwheels_2022 / pakistan_used_cars): 0.813
- groups within 15%: 33%
- sources agree: **no**

## Price rescaling

Prices of later sources were multiplied by these factors (measured on train rows only) to match the first source's price level:

- pakwheels_2022: x1.2346
- after rescaling, pakwheels_2022: median ratio 1.0, 79% of groups within 15%
Train file hash: `d3a726348e1c`
Test file hash: `6a75712d005b`
