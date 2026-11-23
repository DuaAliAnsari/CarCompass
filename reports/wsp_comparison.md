# Weight-space programming — beat attempt (v3)

This run adds LightGBM seed soups (prediction-space model soup), a
teacher-distilled MLP with richer features, same-basin WSP, and
validation-selected prediction blends.

Elapsed training time: 11.2 minutes.

## Existing LightGBM champions (same test rows)

| Model | Test MAE (PKR) |
|---|---:|
| `price_global_ab` | 370,313 |
| `price_new_cars_2010plus` | 695,380 |
| `price_old_cars_pre2010` | 1,601,675 |

## Programs (sorted by validation MAE)

| Program | Val MAE (PKR) | Test MAE (PKR) | Notes |
|---|---:|---:|---|
| `blend_3way` **best** | 325,025 | 361,474 | 0.80·lgbm_greedy + 0.00·lgbm_uniform + 0.20·WSP |
| `blend_lgbm_greedy_x_wsp_a0.2` | 325,025 | 361,474 | (1-α)·lgbm_greedy + α·WSP, α=0.2 on val |
| `blend_lgbm_uniform_x_wsp_a0.2` | 326,119 | 361,896 | (1-α)·lgbm_uniform + α·WSP, α=0.2 on val |
| `blend_teacher_x_wsp_a0.3` | 328,513 | 364,189 | (1-α)·teacher + α·WSP, α=0.3 on val |
| `lgbm_champ_x_greedy_a0.95` | 328,744 | 365,310 | disk champion + 0.95·greedy soup (α on val) |
| `lgbm_soup_greedy` | 328,755 | 365,297 | Greedy LGBM soup members [7, 777, 99, 11, 'disk'] |
| `lgbm_champ_x_uniform_a0.75` | 329,535 | 365,549 | disk champion + 0.75·uniform soup (α on val) |
| `lgbm_soup_uniform` | 329,774 | 365,486 | Mean of 10 LGBM variants (+disk) |
| `lgbm_seed_7` | 330,549 | 365,589 | LGBM variant seed=7 leaves=127 |
| `lgbm_seed_777` | 330,627 | 366,196 | LGBM variant seed=777 leaves=127 |
| `lgbm_seed_99` | 330,831 | 368,029 | LGBM variant seed=99 leaves=127 |
| `lgbm_seed_123` | 331,026 | 367,372 | LGBM variant seed=123 leaves=127 |
| `lgbm_seed_11` | 331,965 | 368,365 | LGBM variant seed=11 leaves=127 |
| `lgbm_seed_404` | 332,212 | 366,697 | LGBM variant seed=404 leaves=159 |
| `lgbm_champion_disk` | 333,743 | 370,313 | Served price_global_ab on disk |
| `lgbm_seed_202` | 334,084 | 371,395 | LGBM variant seed=202 leaves=95 |
| `lgbm_seed_42` | 334,181 | 366,825 | LGBM variant seed=42 leaves=127 |
| `lgbm_seed_22` | 338,899 | 369,140 | LGBM variant seed=22 leaves=63 |
| `wsp_soup_greedy` | 361,827 | 402,585 | Greedy WSP soup {mlp_base, mlp_basin_7} |
| `wsp_wise_distill_a0.25` | 362,003 | 402,254 | WiSE-FT soft→hard α=0.25 |
| `wsp_soup_basin` | 363,276 | 403,875 | Uniform soup of distilled base + forks |
| `mlp_distill_soft` | 364,384 | 404,821 | MLP distilled on 0.65·teacher + 0.35·hard |
| `wsp_wise_distill_a0` | 364,384 | 404,821 | WiSE-FT soft→hard α=0 |
| `mlp_basin_7` | 364,384 | 404,821 | Basin fork seed 7 from distilled WiSE |
| `mlp_basin_99` | 364,384 | 404,821 | Basin fork seed 99 from distilled WiSE |
| `mlp_basin_123` | 364,384 | 404,821 | Basin fork seed 123 from distilled WiSE |
| `mlp_basin_202` | 364,384 | 404,821 | Basin fork seed 202 from distilled WiSE |
| `mlp_basin_404` | 364,384 | 404,821 | Basin fork seed 404 from distilled WiSE |
| `wsp_wise_distill_a0.5` | 368,950 | 408,814 | WiSE-FT soft→hard α=0.5 |
| `wsp_wise_distill_a0.75` | 383,963 | 423,204 | WiSE-FT soft→hard α=0.75 |
| `mlp_distill_hard` | 405,910 | 445,773 | Soft MLP fine-tuned on hard labels |
| `wsp_wise_distill_a1` | 405,910 | 445,773 | WiSE-FT soft→hard α=1 |

## Does it beat `price_global_ab`?

**Yes.** Best program `blend_3way` is 8,839 PKR better on test (361,474 vs 370,313).

## vs previous WSP best

Previous best test MAE 387,149 → this run 361,474 (better by 25,676).

## Method (v3)

1. Retrain champion LightGBM config (`lgbm_900_lr02`) on several seeds; soup log-price predictions.
2. Distill MLP `(384,192,96)` on soft teacher targets mixed with hard labels; richer features (age, log mileage, make_model).
3. Same-basin forks + greedy weight soup + WiSE-FT toward the teacher.
4. Prediction blends of teacher / LGBM soup / WSP; α chosen on validation.
5. Best program = lowest validation MAE; test MAE reported once.

