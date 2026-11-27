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
| 2026-11-27 10:51 | `1127_1048_logistic_stack_5m.csv` | 0.95066 | **0.95170** | ~1039/3322 | approved by user; LB − CV = +0.00104 (the baseline had ≈ 0); public top-10% cutoff 0.95533 |
| 2026-11-27 13:46 | `1127_1244_logistic_stack_3m.csv` (ref 59001979) | 0.95067 | 0.95167 | — | Phase 3 A, approved by user: pruned 3-member stack, test = fold average |
| 2026-11-27 13:46 | `1127_1342_logistic_stack_3m_refit-glm_boost-xgb_d2_te.csv` (ref 59001983) | 0.95067 (same OOF) | **0.95173** | — | Phase 3 B, approved by user: same stack, glm_boost + xgb_d2_te test from one all-rows refit (lever 5). B − A = +0.00006 with file ρ 0.9988 (gap noise SD ≈ 3e-4·√(1−ρ) ≈ 1e-5) |

## Phase 3: queue runner (2026-11-27)

`experiments/run_queue.py <ideas.txt>` runs each line (`<idea> blend`, `<idea> single <ref>`, `prune`): screen → kill check → full → gate → library → re-stack, with each step recorded in `<ideas>.state.json` (resume). Kill rule (user, 2026-11-27): only `single` variants can be killed, iff the screen trails the reference by > 2% of (1 − ref AUC) **and** every scored fold is down; families always get a full run. Hours guard: a step starts only if its estimate (full = 18 × screen runtime) ends by 20:00. Pruning (user, 2026-11-27): drop the weakest leave-one-out member while it fails the gate, one per round.

| # | member | single CV | per-fold CV | Δ blend (gate) | folds up | gate | runtime (full) | est. |
|---|---|---|---|---|---|---|---|---|
| check | lgbm31_raw via queue (single vs baseline) | 0.94503 | identical to Phase 2 (max fold diff 0.0) | −0.00163 single | 0/5 | FAIL (expected) | 1.3 min | 1.8 min |
| 7 | catboost_cat (GPU, raw + string copies of every numeric/flag) | 0.94637 | 0.94442 0.94665 0.94198 0.94664 0.95218 | −0.00001 (5 members); −0.00001 (3) | 1/5; 2/5 | FAIL (ρ 0.963) | 24.6 min | 23.2 min |
| 8 | glm_boost_lgbm4 (LGBM 4 leaves + GLM init_score) | **0.95029** | 0.94825 0.95042 0.94691 0.95031 0.95558 | +0.00002 (5); +0.00002 (3) | 5/5 | FAIL (ρ 0.9995 with glm_boost) | 8.8 min | 19.1 min |
| prune | leave-one-out, 5 → 3 members | — | — | te_xgb −0.00000, then glm +0.00000 | 1/5, 3/5 | DROP both | 1 min | — |
| 9 | realmlp (pytabkit, S6E10 config, holdout epoch + n_refit=1) | 0.94993 | 0.94801 0.95069 0.94595 0.95017 0.95481 | +0.00001 (3) | 3/5 | FAIL (ρ 0.986, weight 0.125) | 9.0 min | 19.0 min |
| 10 | tabm (tabm-mini k=32, pwl; no refit in pytabkit → 80% of fold) | 0.94932 | 0.94712 0.95008 0.94564 0.94896 0.95478 | −0.00001 (3) | 2/5 | FAIL (weight 0.000) | 33.0 min | 77.1 min |

| 11 | xgb_d2_te_bag (TE training rows averaged over seeds 17/18/19) | — | screen 0.93648 0.94715 0.94958 | screen −0.00320 vs xgb_d2_te (threshold −0.00105) | 0/3 | KILL | 0.4 min (screen) | — |
| 12 | xgb_d2_te_gp (+ deviations of 6 numerics from their per-category means, train-fold stats) | 0.95023 | 0.94838 0.95046 0.94657 0.95025 0.95549 | +0.00002 single vs xgb_d2_te | 3/5 | FAIL | 3.6 min | 4.6 min |

The bagging kill is not a wrapper bug: `BaggedTargetEncoder` with the single seed 17 reproduces the xgb_d2_te screen exactly (0.93943 0.95123 0.95214). A likely cause: averaging lowers the noise of the training-row encodings, so the trees over-trust TE relative to the whole-fold valid/test encodings. Canaries (shuffled labels, seeds 2–4) pass for both new builders.

Library after Phase 3 batches: `xgb_d2_te glm_boost baseline`. Logistic stack **0.95067** (hill climb 0.95065), up from 0.95066 with 5 members. NN canary (TE pipeline + integer categories, shuffled labels, seeds 2–4): 0.4916, same as the TE pipeline alone (0.4914); seed 0 gave 0.475, below chance, so not leakage.

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
| 11-27 12:00 | lgbm31_raw | screen | 0.94321 | 0.93713 0.94626 0.94623 | 0.1 min | [136, 158, 183] |
| 11-27 12:00 | QUEUE kill-check lgbm31_raw vs baseline | screen | 0.94321 vs 0.94416, Δ -0.00095, threshold -0.00112 | 3/3 folds down | GO |
| 11-27 12:00 | QUEUE lgbm31_raw | full | est 1.8 min | started | - | - |
| 11-27 12:02 | lgbm31_raw | full | 0.94503 | 0.94286 0.94630 0.94066 0.94524 0.95012 | 1.3 min | 286 |
| 11-27 12:02 | GATE single lgbm31_raw vs baseline | full | Δ -0.00163 | Δ per fold -0.00178 -0.00120 -0.00191 -0.00157 -0.00168 | 0/5 up | FAIL |
| 11-27 12:02 | QUEUE done queue_check.txt | - | - | - | - | - |
| 11-27 12:04 | catboost_cat | screen | 0.94445 | 0.93853 0.94793 0.94690 | 1.3 min | [680, 1039, 973] |
| 11-27 12:04 | QUEUE catboost_cat | full | est 23.2 min | started | - | - |
| 11-27 12:29 | catboost_cat | full | 0.94637 | 0.94442 0.94665 0.94198 0.94664 0.95218 | 24.6 min | 1957 |
| 11-27 12:29 | GATE blend catboost_cat vs [te_xgb glm xgb_d2_te glm_boost baseline] | full | 0.95064 → 0.95064, Δ -0.00001 | Δ per fold -0.00000 -0.00003 -0.00001 +0.00001 -0.00000 | 1/5 up, weight 0.037, max ρ 0.96315 | FAIL |
| 11-27 12:30 | glm_boost_lgbm4 | screen | 0.94869 | 0.94223 0.95317 0.95066 | 1.1 min | [125, 135, 416] |
| 11-27 12:30 | QUEUE glm_boost_lgbm4 | full | est 19.1 min | started | - | - |
| 11-27 12:39 | glm_boost_lgbm4 | full | 0.95029 | 0.94825 0.95042 0.94691 0.95031 0.95558 | 8.8 min | 816 |
| 11-27 12:40 | GATE blend glm_boost_lgbm4 vs [te_xgb glm xgb_d2_te glm_boost baseline] | full | 0.95064 → 0.95066, Δ +0.00002 | Δ per fold +0.00001 +0.00001 +0.00001 +0.00003 +0.00002 | 5/5 up, weight 0.517, max ρ 0.99949 | FAIL |
| 11-27 12:41 | PRUNE loo te_xgb [5 members] | full | Δ -0.00000 | Δ per fold -0.00000 -0.00000 +0.00000 -0.00000 -0.00001 | 1/5 up | fail | DROP |
| 11-27 12:41 | PRUNE loo glm [5 members] | full | Δ -0.00000 | Δ per fold -0.00000 -0.00000 +0.00000 -0.00000 +0.00000 | 0/5 up | fail | - |
| 11-27 12:41 | PRUNE loo xgb_d2_te [5 members] | full | Δ +0.00005 | Δ per fold +0.00008 +0.00010 -0.00004 +0.00005 +0.00009 | 4/5 up | keep | - |
| 11-27 12:41 | PRUNE loo glm_boost [5 members] | full | Δ +0.00017 | Δ per fold +0.00007 -0.00001 +0.00030 +0.00027 +0.00020 | 4/5 up | keep | - |
| 11-27 12:41 | PRUNE loo baseline [5 members] | full | Δ +0.00009 | Δ per fold -0.00000 +0.00014 +0.00009 +0.00015 +0.00008 | 4/5 up | keep | - |
| 11-27 12:41 | PRUNE loo glm [4 members] | full | Δ +0.00000 | Δ per fold +0.00000 -0.00000 -0.00000 +0.00000 +0.00000 | 3/5 up | fail | DROP |
| 11-27 12:41 | PRUNE loo xgb_d2_te [4 members] | full | Δ +0.00011 | Δ per fold +0.00023 +0.00018 -0.00005 +0.00009 +0.00012 | 4/5 up | keep | - |
| 11-27 12:41 | PRUNE loo glm_boost [4 members] | full | Δ +0.00017 | Δ per fold +0.00008 -0.00001 +0.00030 +0.00027 +0.00019 | 4/5 up | keep | - |
| 11-27 12:41 | PRUNE loo baseline [4 members] | full | Δ +0.00011 | Δ per fold +0.00004 +0.00016 +0.00011 +0.00017 +0.00007 | 5/5 up | keep | - |
| 11-27 12:42 | PRUNE loo xgb_d2_te [3 members] | full | Δ +0.00011 | Δ per fold +0.00023 +0.00018 -0.00005 +0.00009 +0.00012 | 4/5 up | keep | - |
| 11-27 12:42 | PRUNE loo glm_boost [3 members] | full | Δ +0.00032 | Δ per fold +0.00014 +0.00015 +0.00058 +0.00035 +0.00036 | 5/5 up | keep | - |
| 11-27 12:42 | PRUNE loo baseline [3 members] | full | Δ +0.00011 | Δ per fold +0.00004 +0.00016 +0.00011 +0.00016 +0.00007 | 5/5 up | keep | - |
| 11-27 12:44 | STACK hill_climb [3 members] | full | 0.95065 | 0.94870 0.95096 0.94702 0.95067 0.95589 | vs best single +0.00039 | xgb_d2_te=0.341 glm_boost=0.512 baseline=0.146 |
| 11-27 12:44 | STACK logistic_stack [3 members] | full | 0.95067 | 0.94874 0.95098 0.94703 0.95068 0.95589 | vs best single +0.00041 | xgb_d2_te=0.504 glm_boost=0.680 baseline=0.176 |
| 11-27 12:44 | QUEUE done batch1.txt | - | - | - | - | - |
| 11-27 12:45 | GATE blend glm_boost_lgbm4 vs [xgb_d2_te glm_boost baseline] | full | 0.95065 → 0.95066, Δ +0.00002 | Δ per fold +0.00001 +0.00001 +0.00002 +0.00003 +0.00002 | 5/5 up, weight 0.524, max ρ 0.99949 | FAIL |
| 11-27 12:45 | GATE blend catboost_cat vs [xgb_d2_te glm_boost baseline] | full | 0.95065 → 0.95064, Δ -0.00001 | Δ per fold -0.00000 -0.00002 -0.00001 +0.00001 +0.00000 | 2/5 up, weight 0.025, max ρ 0.96315 | FAIL |
| 11-27 12:51 | realmlp | screen | 0.94804 | 0.94148 0.95163 0.95100 | 1.1 min | - |
| 11-27 12:51 | QUEUE realmlp | full | est 19.0 min | started | - | - |
| 11-27 13:00 | realmlp | full | 0.94993 | 0.94801 0.95069 0.94595 0.95017 0.95481 | 9.0 min | - |
| 11-27 13:00 | GATE blend realmlp vs [xgb_d2_te glm_boost baseline] | full | 0.95065 → 0.95065, Δ +0.00001 | Δ per fold +0.00002 +0.00004 -0.00003 +0.00004 -0.00003 | 3/5 up, weight 0.125, max ρ 0.98617 | FAIL |
| 11-27 13:04 | tabm | screen | 0.94717 | 0.94049 0.95038 0.95065 | 4.3 min | - |
| 11-27 13:04 | QUEUE tabm | full | est 77.1 min | started | - | - |
| 11-27 13:38 | tabm | full | 0.94932 | 0.94712 0.95008 0.94564 0.94896 0.95478 | 33.0 min | - |
| 11-27 13:38 | GATE blend tabm vs [xgb_d2_te glm_boost baseline] | full | 0.95065 → 0.95064, Δ -0.00001 | Δ per fold +0.00000 -0.00000 -0.00002 -0.00001 +0.00000 | 2/5 up, weight 0.000, max ρ 0.97724 | FAIL |
| 11-27 13:38 | QUEUE done batch2.txt | - | - | - | - | - |
| 11-27 13:40 | REFIT glm_boost (all rows) | full | test only | 741 trees | 1.2 min | - |
| 11-27 13:40 | REFIT xgb_d2_te (all rows) | full | test only | 2067 trees | 0.3 min | - |
| 11-27 13:42 | STACK hill_climb_refit-glm_boost-xgb_d2_te [3 members] | full | 0.95065 | 0.94870 0.95096 0.94702 0.95067 0.95589 | vs best single +0.00039 | xgb_d2_te=0.341 glm_boost=0.512 baseline=0.146 |
| 11-27 13:42 | STACK logistic_stack_refit-glm_boost-xgb_d2_te [3 members] | full | 0.95067 | 0.94874 0.95098 0.94703 0.95068 0.95589 | vs best single +0.00041 | xgb_d2_te=0.504 glm_boost=0.680 baseline=0.176 |
| 11-27 13:51 | xgb_d2_te_bag | screen | 0.94440 | 0.93648 0.94715 0.94958 | 0.4 min | [2787, 2102, 1922] |
| 11-27 13:51 | QUEUE kill-check xgb_d2_te_bag vs xgb_d2_te | screen | 0.94440 vs 0.94760, Δ -0.00320, threshold -0.00105 | 3/3 folds down | KILL |
| 11-27 13:51 | xgb_d2_te_gp | screen | 0.94780 | 0.93974 0.95150 0.95215 | 0.3 min | [1611, 922, 1330] |
| 11-27 13:51 | QUEUE kill-check xgb_d2_te_gp vs xgb_d2_te | screen | 0.94780 vs 0.94760, Δ +0.00020, threshold -0.00105 | 0/3 folds down | GO |
| 11-27 13:51 | QUEUE xgb_d2_te_gp | full | est 4.6 min | started | - | - |
| 11-27 13:55 | xgb_d2_te_gp | full | 0.95023 | 0.94838 0.95046 0.94657 0.95025 0.95549 | 3.6 min | 2092 |
| 11-27 13:55 | GATE single xgb_d2_te_gp vs xgb_d2_te | full | Δ +0.00002 | Δ per fold -0.00012 -0.00017 +0.00023 +0.00011 +0.00004 | 3/5 up | FAIL |
| 11-27 13:55 | QUEUE done batch3.txt | - | - | - | - | - |
