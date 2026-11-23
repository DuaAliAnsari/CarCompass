# CarCompass – Known Limitations (to fix in Milestone 2)

_Noted 2026-11-05. Not committed; personal working note._

## 1. Cross-source duplicate removal is too loose

**Where:** `src/prepare_multi.py:61` `drop_cross_source_duplicates()` (commit `0f1d2baf`, feat/multi-model).

**What it does:** deletes every Dataset B row whose key
`make, model, year, mileage, city` (`GROUP_COLS`, `src/prepare_data.py:39`) also
appears anywhere in Dataset A. Result: **8,558 B rows removed**.

**Why it was done:** A and B are both PakWheels scrapes (B is older), so the same car
can be in both. If one copy is in train and the other in test, the test score is
inflated (leakage), and the car is counted twice in training.

**Why it is a problem:** the key says "these listings look similar", not "this is the
same physical car". Audit of the 8,558 matches (re-run on raw data):

| Check on matched pairs | Result |
|---|---|
| Transmission agrees | 78.5% |
| Engine size agrees | 71.1% |
| Fuel agrees | 92.3% |
| Assembly agrees | 88.9% |
| Mileage is a multiple of 10,000 | 62.2% (vs 32.8% of all B rows) |
| Matches at exactly 100,000 km | 1,525 |
| A rows matched (B rows match several A rows) | 16,616 pairs for 8,558 B rows |
| A rows sharing their key with another A row | 13,382 |
| Still matched with a strict key (same transmission, fuel, assembly, engine within 50 cc) | 5,489 |
| Strict key + non-round mileage | 1,982 |
| A/B price ratio of matches (median) | 1.40 (same as the 1.42 price factor; does not prove same car) |

So roughly **3,000–6,500 of the removed rows were probably different cars**.

**Impact:** errs on the safe side: no leakage, so reported test results
(served system MAE 360,636 PKR) remain valid. The cost is lost training data, mostly
hurting the B-only and pre-2010 families.

**Not covered (opposite risk):** the same car relisted later with higher mileage does
not match at all (mileage is in the key), so it could still leak. Likely small.

**Fix for M2:**
1. Keep all B rows; give matching A and B rows the **same `listing_group`** so the
   group-aware split keeps them on one side (prevents leakage without deleting data).
2. Delete only **strict-key** duplicates (same transmission, fuel, assembly, engine
   within 50 cc, ideally non-round mileage) to avoid double counting.
3. Re-run `make train-multi`, report test MAE before vs after in
   `reports/model_comparison.md`.
4. Fix the docstring in `src/prepare_multi.py` (line 12) that claims a matching key
   "is the same car".

**Viva answer in one line:** "The cross-source dedup key over-matched; it was
conservative (no leakage, lost some B data). We measured it, and in M2 we switched to
grouping plus strict-key removal and compared MAE."

## 2. Other open items (from earlier)

- PR #7 (README) has factual errors: `src/prepare_b.py` should be
  `src/prepare_combined.py`; `prepare_data.py` *is* used by the A+B path; Docker
  section conflicts with PR #6.
- PR #6 still targets `feat/multi-model`; retarget to `main` after PR #7.
- Tag `v1.0-milestone1` not created yet.
