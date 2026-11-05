# Data quality report: shared A+B dataset

| Step | Dataset A | Dataset B |
|---|---|---|
| Rows read | 61919 | 55675 |
| Dropped: missing required values | 853 | 0 |
| Dropped: out of range | 140 | 191 |
| Dropped: duplicates (within source) | 826 | 3662 |
| Dropped: low mileage | 1308 | 3803 |
| Clean rows | 58792 | 48019 |
| Dropped: same car already in A |  | 8558 |

Rules on A+B together: dropped 467 rows of rare make/model pairs, grouped 504 rows of rare cities as 'Other'.

| Split | Dataset A | Dataset B | Total | Used for |
|---|---|---|---|---|
| fit | 37590 | 24983 | 62573 | training |
| val | 4698 | 3167 | 7865 | choosing champions and routing |
| cal | 4709 | 3114 | 7823 | price range |
| test | 11556 | 7969 | 19525 | final scores only |
| **all** | | | **97786** | |

Pre-2010 cars: 32961, 2010 and later: 64825.
B prices scaled by 1.4213 (A's prices are this many times B's 2021 prices for the same car), from 642 make/model/year groups in the fit rows.
Seed: 42. Pandera schema check: passed.
File hash: `2a4b7626af05`
