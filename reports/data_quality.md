# Data quality report

| Step | Rows |
|---|---|
| Rows read | 61919 |
| Dropped: missing required values | 853 |
| Dropped: out of range | 140 |
| Dropped: duplicates | 826 |
| Dropped: low mileage | 1308 |
| Dropped: rare make/model | 411 |
| City grouped as 'Other' | 350 |
| Engine set to missing | 157 |
| **Rows kept** | **58381** |
| Kept: pre-2010 cars | 18051 |
| Kept: 2010 and later | 40330 |
| Train rows | 92654 |
| Test rows | 11701 |

Seed: 42
Pandera schema check: passed
Train file hash: `68d3a93fd824`
Test file hash: `6b3d8898d652`

## Dataset B (added to the training data only)

| Step | Rows |
|---|---|
| Rows read | 55675 |
| Dropped: missing required values | 0 |
| Dropped: out of range | 191 |
| Dropped: duplicates | 3662 |
| Dropped: low mileage | 3803 |
| Dropped: rare make/model | 334 |
| Dropped: listing group also in the test set | 1711 |
| City set to 'Other' | 614 |
| **Rows added to training** | **45974** |

Price factor applied to B (A's prices are this many times B's for the same car):
1.4286, from 762 make/model/year groups.
Train rows: 46680 from A + 45974 from B = 92654.
Test rows: 11701 (A only). Pandera schema check of the combined data:
passed
