# S6E11 experiments

Loop: `experiments/<idea>.py screen` → gate → `experiments/<idea>.py full` → gate → library. Folds: `folds.npy` = `StratifiedKFold(5, shuffle=True, random_state=17)` over train row order, shared by every member. Members live in `oof/<mode>/<name>.npz` (gitignored).

Gate (decided 2026-11-27): ≥ +5e-5 mean AUC **and** ≥ 4/5 folds up. A feature change that replaces a model (raw → TE) is gated on single-model CV; a new model family is gated on the nested hill-climb of library + candidate vs library alone (`experiments/gate.py`).

Honest trees: members with early stopping run one pass with `early_stopping_holdout=0.2` (inner tail of each training fold), then refit on whole training folds with 1.25 × mean best trees and no early stopping. Screen mode stops after the holdout pass.

## Screening calibration

`experiments/calibrate.py`, 2026-11-27. Four variants with known full CV (baseline 0.94666, te_xgb 0.94940, lgbm4_te 0.95012, lgbm31_raw 0.94503; 20.8 min of full runs in total). Screen = stratified `frac` of every (fold, label) group of `folds.npy`; only folds below `n_folds` are scored, the rest only train; early stopping on the inner holdout, no refit.

| screen | baseline / te_xgb / lgbm4_te / lgbm31_raw | Kendall τ vs full | same order | runtime (4 variants) | speed-up |
|---|---|---|---|---|---|
| **20% rows × 3 folds (chosen)** | 0.94416 / 0.94503 / 0.94771 / 0.94321 | 1.00 | yes | 1.1 min | 18.8× |
| 20% rows × 5 folds | 0.94469 / 0.94504 / 0.94718 / 0.94249 | 1.00 | yes | 2.0 min | 10.5× |
| 35% rows × 3 folds | 0.94413 / 0.94625 / 0.94659 / 0.94199 | 1.00 | yes | 1.6 min | 12.8× |
| 35% rows × 5 folds | 0.94294 / 0.94505 / 0.94568 / 0.94066 | 1.00 | yes | 2.5 min | 8.3× |

Over all 7 full members the chosen screen has τ = 0.71. It undervalues the GLM (screen 0.94307, below the baseline; full 0.94928): the one-hots of exact values need rows. It also swaps xgb_d2_te and lgbm4_te, which are 9e-5 apart on full CV. It shrinks TE gains (te_xgb − baseline: +0.00087 screen vs +0.00274 full). **Use:** the screen kills an idea only when it trails the reference by > 1e-3; new model families always get a full run; gates are decided on full runs.

## Ladder

All members on `folds.npy`, honest trees, full mode. Gate order followed completion order (GLM finished before LGBM-4), and the order matters: LGBM-4 failed only because XGB-d2 was already in.

| # | member | single CV | per-fold CV | Δ (gate) | folds up | gate | runtime | in library |
|---|---|---|---|---|---|---|---|---|
| 0 | baseline (notebook XGB, raw) | 0.94666 | 0.94463 0.94751 0.94256 0.94681 0.95180 | — | — | reference (notebook 0.9468) | 2.2 min | yes (step 6) |
| 1 | te_xgb (raw + TE of values, digit keys) | 0.94940 | 0.94786 0.94985 0.94557 0.94929 0.95444 | +0.00274 single vs baseline | 5/5 | PASS | 6.1 min | yes |
| 2 | glm (L2 logistic C=0.01) | 0.94928 | 0.94722 0.94976 0.94590 0.94904 0.95447 | +0.00074 blend (0.94940 → 0.95014) | 5/5 | PASS | 3.2 min | yes |
| 3 | xgb_d2_te (depth 2, lr 0.03) | 0.95021 | 0.94850 0.95063 0.94634 0.95014 0.95544 | +0.00028 blend (→ 0.95042) | 5/5 | PASS | 4.0 min | yes |
| 4 | lgbm4_te (4 leaves, lr 0.03) | 0.95012 | 0.94841 0.95045 0.94625 0.95006 0.95541 | +0.00002 blend | 4/5 | FAIL (ρ 0.9967 with xgb_d2_te) | 10.8 min | no |
| 5 | glm_boost (xgb_d2 + cross-fitted GLM base_margin) | 0.95026 | 0.94823 0.95038 0.94689 0.95025 0.95555 | +0.00014 blend (→ 0.95056) | 4/5 | PASS | 7.6 min | yes |
| 6 | baseline as a member | 0.94666 | (above) | +0.00009 blend (→ 0.95064) | 4/5 | PASS (ρ 0.963, raw view) | — | yes |
| — | lgbm31_raw (calibration contrast) | 0.94503 | 0.94286 0.94630 0.94066 0.94524 0.95012 | — | — | not gated | 1.7 min | no |

Canaries (shuffled labels, `test_plumbing.py`): TE pipeline AUC ≈ 0.5; GLM mean of seeds 2–4 = 0.504. Seed 1 gave 0.526 both with and without TE (the no-TE GLM has no label-derived inputs, so it is permutation noise, sd ≈ 0.009). GLM-boost offsets: a training row's offset does not change when its own label flips.

## Stack and leaderboard

`experiments/stack.py`, library = `experiments/library.txt` (te_xgb, glm, xgb_d2_te, glm_boost, baseline). Nested CV on the same folds.

| blender | nested CV | per-fold | vs best single (0.95026) | weights |
|---|---|---|---|---|
| hill climb (rank) | 0.95064 | 0.94870 0.95096 0.94703 0.95066 0.95588 | +0.00039 | te_xgb 0.045, glm 0.000, xgb_d2_te 0.318, glm_boost 0.500, baseline 0.136 |
| **logistic stack** | **0.95066** | 0.94874 0.95096 0.94705 0.95065 0.95589 | +0.00040 | coefs te_xgb 0.042, glm 0.034, xgb_d2_te 0.452, glm_boost 0.687, baseline 0.146 |

Submission file: `submissions/1127_1048_logistic_stack_5m.csv` (rank-scaled, test = 5-fold-averaged member predictions).

| date | file | CV | public LB | rank | note |
|---|---|---|---|---|---|
| 2026-11-02 | baseline notebook (OOF approach) | 0.9468 | 0.94677 | 2343/3274 (on 11-26) | user's own |

## Run log

Appended by `common.run_member` and `gate.py`. Trees: refit tree count (full) or best trees per fold (screen).

| time | member | mode | mean AUC | fold AUCs | runtime | trees |
|---|---|---|---|---|---|---|
| 11-27 10:07 | baseline | screen | 0.94416 | 0.93737 0.94825 0.94686 | 0.2 min | 10000 |
| 11-27 10:09 | lgbm31_raw | screen | 0.94321 | 0.93713 0.94626 0.94623 | 0.1 min | [136, 158, 183] |
| 11-27 10:09 | lgbm4_te | screen | 0.94771 | 0.93949 0.95200 0.95165 | 0.6 min | [1869, 818, 2080] |
| 11-27 10:10 | baseline | full | 0.94666 | 0.94463 0.94751 0.94256 0.94681 0.95180 | 2.2 min | 1937 |
| 11-27 10:11 | te_xgb | screen | 0.94503 | 0.93774 0.94825 0.94911 | 0.5 min | [1479, 828, 1652] |
| 11-27 10:16 | glm | screen | 0.94307 | 0.93580 0.94895 0.94448 | 0.6 min | - |
| 11-27 10:17 | te_xgb | full | 0.94940 | 0.94786 0.94985 0.94557 0.94929 0.95444 | 6.1 min | 1909 |
| 11-27 10:17 | GATE single te_xgb vs baseline | full | Δ +0.00274 | Δ per fold +0.00323 +0.00235 +0.00301 +0.00248 +0.00263 | 5/5 up | PASS |
| 11-27 10:18 | xgb_d2_te | screen | 0.94760 | 0.93943 0.95123 0.95214 | 0.4 min | [1841, 761, 1380] |
| 11-27 10:19 | lgbm31_raw | full | 0.94503 | 0.94286 0.94630 0.94066 0.94524 0.95012 | 1.7 min | 286 |
| 11-27 10:20 | glm | full | 0.94928 | 0.94722 0.94976 0.94590 0.94904 0.95447 | 3.2 min | - |
| 11-27 10:21 | GATE blend glm vs [te_xgb] | full | 0.94940 → 0.95014, Δ +0.00074 | Δ per fold +0.00047 +0.00076 +0.00096 +0.00068 +0.00082 | 5/5 up, weight 0.474, max ρ 0.97560 | PASS |
| 11-27 10:22 | xgb_d2_te | full | 0.95021 | 0.94850 0.95063 0.94634 0.95014 0.95544 | 4.0 min | 2067 |
| 11-27 10:22 | GATE blend xgb_d2_te vs [te_xgb glm] | full | 0.95014 → 0.95042, Δ +0.00028 | Δ per fold +0.00028 +0.00025 +0.00015 +0.00032 +0.00038 | 5/5 up, weight 0.590, max ρ 0.99192 | PASS |
| 11-27 10:24 | glm_boost | screen | 0.94876 | 0.94289 0.95303 0.95037 | 1.1 min | [294, 162, 337] |
| 11-27 10:29 | lgbm4_te | full | 0.95012 | 0.94841 0.95045 0.94625 0.95006 0.95541 | 10.8 min | 1981 |
| 11-27 10:30 | GATE blend lgbm4_te vs [te_xgb glm xgb_d2_te] | full | 0.95042 → 0.95044, Δ +0.00002 | Δ per fold +0.00003 -0.00001 +0.00001 +0.00004 +0.00004 | 4/5 up, weight 0.277, max ρ 0.99672 | FAIL |
| 11-27 10:32 | glm_boost | full | 0.95026 | 0.94823 0.95038 0.94689 0.95025 0.95555 | 7.6 min | 741 |
| 11-27 10:32 | GATE blend glm_boost vs [te_xgb glm xgb_d2_te] | full | 0.95042 → 0.95056, Δ +0.00014 | Δ per fold +0.00009 -0.00005 +0.00025 +0.00023 +0.00017 | 4/5 up, weight 0.500, max ρ 0.99576 | PASS |
| 11-27 10:32 | baseline@0.2x3 | screen | 0.94416 | 0.93737 0.94825 0.94686 | 0.2 min | [1109, 1113, 979] |
| 11-27 10:33 | te_xgb@0.2x3 | screen | 0.94503 | 0.93774 0.94825 0.94911 | 0.4 min | [1479, 828, 1652] |
| 11-27 10:33 | lgbm4_te@0.2x3 | screen | 0.94771 | 0.93949 0.95200 0.95165 | 0.4 min | [1869, 818, 2080] |
| 11-27 10:33 | lgbm31_raw@0.2x3 | screen | 0.94321 | 0.93713 0.94626 0.94623 | 0.1 min | [136, 158, 183] |
| 11-27 10:34 | baseline@0.2x5 | screen | 0.94469 | 0.93737 0.94825 0.94686 0.94576 0.94523 | 0.3 min | [1109, 1113, 979, 1081, 1051] |
| 11-27 10:34 | te_xgb@0.2x5 | screen | 0.94504 | 0.93774 0.94825 0.94911 0.94640 0.94372 | 0.7 min | [1479, 828, 1652, 1493, 1463] |
| 11-27 10:35 | lgbm4_te@0.2x5 | screen | 0.94718 | 0.93949 0.95200 0.95165 0.94684 0.94593 | 0.7 min | [1869, 818, 2080, 1971, 1217] |
| 11-27 10:35 | lgbm31_raw@0.2x5 | screen | 0.94249 | 0.93713 0.94626 0.94623 0.94127 0.94157 | 0.2 min | [136, 158, 183, 192, 166] |
| 11-27 10:36 | baseline@0.35x3 | screen | 0.94413 | 0.94496 0.94756 0.93987 | 0.2 min | [942, 861, 878] |
| 11-27 10:36 | te_xgb@0.35x3 | screen | 0.94625 | 0.94767 0.94879 0.94228 | 0.5 min | [1361, 1153, 1149] |
| 11-27 10:37 | lgbm4_te@0.35x3 | screen | 0.94659 | 0.94818 0.94982 0.94179 | 0.7 min | [1952, 1111, 1393] |
| 11-27 10:37 | lgbm31_raw@0.35x3 | screen | 0.94199 | 0.94130 0.94594 0.93872 | 0.1 min | [216, 180, 219] |
| 11-27 10:37 | baseline@0.35x5 | screen | 0.94294 | 0.94496 0.94756 0.93987 0.93852 0.94379 | 0.3 min | [942, 861, 878, 960, 958] |
| 11-27 10:38 | te_xgb@0.35x5 | screen | 0.94505 | 0.94767 0.94879 0.94228 0.94038 0.94612 | 0.8 min | [1361, 1153, 1149, 975, 1250] |
| 11-27 10:39 | lgbm4_te@0.35x5 | screen | 0.94568 | 0.94818 0.94982 0.94179 0.94237 0.94625 | 1.1 min | [1952, 1111, 1393, 833, 1866] |
| 11-27 10:40 | lgbm31_raw@0.35x5 | screen | 0.94066 | 0.94130 0.94594 0.93872 0.93476 0.94255 | 0.2 min | [216, 180, 219, 95, 149] |
| 11-27 10:41 | GATE blend baseline vs [te_xgb glm xgb_d2_te glm_boost] | full | 0.95056 → 0.95064, Δ +0.00009 | Δ per fold -0.00000 +0.00014 +0.00009 +0.00015 +0.00008 | 4/5 up, weight 0.136, max ρ 0.96301 | PASS |
| 11-27 10:44 | STACK hill_climb [4 members] | full | 0.95056 | 0.94870 0.95082 0.94694 0.95052 0.95580 | vs best single +0.00030 | te_xgb=0.130 glm=0.019 xgb_d2_te=0.352 glm_boost=0.500 |
| 11-27 10:44 | STACK logistic_stack [4 members] | full | 0.95058 | 0.94871 0.95083 0.94698 0.95052 0.95584 | vs best single +0.00032 | te_xgb=0.140 glm=0.084 xgb_d2_te=0.491 glm_boost=0.645 |
| 11-27 10:48 | STACK hill_climb [5 members] | full | 0.95064 | 0.94870 0.95096 0.94703 0.95066 0.95588 | vs best single +0.00039 | te_xgb=0.045 glm=0.000 xgb_d2_te=0.318 glm_boost=0.500 baseline=0.136 |
| 11-27 10:48 | STACK logistic_stack [5 members] | full | 0.95066 | 0.94874 0.95096 0.94705 0.95065 0.95589 | vs best single +0.00040 | te_xgb=0.042 glm=0.034 xgb_d2_te=0.452 glm_boost=0.687 baseline=0.146 |
