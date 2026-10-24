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
| Train rows | 46680 |
| Test rows | 11701 |

Seed: 42
Pandera schema check: passed
Train file hash: `751022ac900d`
Test file hash: `1175ae431d0a`
