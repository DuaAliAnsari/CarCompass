# Model comparison: five price families

Shared dataset: 97786 listings, test rows: 19525 (A 11556, B 7969). B prices are scaled to A's price level.
Champions are chosen on validation rows; test rows are only reported.

## Family champions (each on its own rows)

| Family | Champion | Train rows | Val MAE | Test MAE | Test MdAPE | Range coverage |
|---|---|---|---|---|---|---|
| `price_global_ab` | `lgbm_900_lr02` | 62573 | 333,743 | 370,313 | 6.5% | 80% |
| `price_dataset_a` | `lgbm_900_lr02` | 37590 | 347,817 | 387,479 | 6.1% | 80% |
| `price_dataset_b` | `lgbm_900_lr02` | 24983 | 292,036 | 319,170 | 5.8% | 80% |
| `price_new_cars_2010plus` | `lgbm_900_lr02` | 41467 | 376,902 | 413,633 | 5.1% | 80% |
| `price_old_cars_pre2010` | `lgbm_600_lr03` | 21106 | 240,116 | 263,931 | 9.9% | 80% |

## Same test rows for every model (MAE in PKR)

| Model | All test | A rows | B rows | Pre-2010 | 2010+ |
|---|---|---|---|---|---|
| `price_global_ab` | 370,313 | 389,722 | 342,167 | 257,490 | 428,281 |
| `price_dataset_a` | 413,223 | 387,479 | 450,554 | 271,935 | 485,816 |
| `price_dataset_b` | 636,365 | 855,102 | 319,170 | 274,047 | 822,524 |
| `price_new_cars_2010plus` | 695,380 | 643,021 | 771,307 | 1,243,740 | 413,633 |
| `price_old_cars_pre2010` | 1,601,675 | 1,901,993 | 1,166,177 | 263,931 | 2,289,008 |
| **served system** | 360,636 | 376,881 | 337,079 | 257,490 | 413,633 |

Era specialists only make sense on their own years, so read them in the Pre-2010 or 2010+ column.

## Routing decision (validation rows)

| Default candidate | Val MAE (all val rows) |
|---|---|
| `price_global_ab` ← default | 333,743 |
| `price_dataset_a` | 379,527 |
| `price_dataset_b` | 560,441 |

A specialist is switched on if it beats the default by at least 1% on its own years.

| Specialist | Its val MAE | Default on same rows | Gain | Switched on |
|---|---|---|---|---|
| `price_new_cars_2010plus` | 376,902 | 382,254 | +1.4% | yes |
| `price_old_cars_pre2010` | 240,116 | 237,237 | -1.2% | no |

## Does the assembly column help?

`price_global_ab` champion settings (`lgbm_900_lr02`): val MAE 333,743 with assembly, 334,935 without -> assembly helps.
