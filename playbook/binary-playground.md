# Binary Playground playbook

What has measurably worked for **self-built** solutions on Kaggle Playground binary competitions (ROC AUC, ~700k synthetic rows). Each technique carries its evidence; each competition adds a retro section at the end. Numbers are from authors' own tables or markdown (public notebooks had no saved outputs), so treat them as directional and re-measure here.

Rules of the road (user, 2026-11-26): techniques from public notebooks and discussions plus the original dataset are fine. Never ingest anyone's OOF/submission files, never fit anything to the public LB.

## Ranked levers

### 1. Target encoding of exact values and digit keys (biggest: +0.002 to +0.003)
- Treat every raw column's exact value as a category. For each float column add digit keys: `cents = round(100x) % 100`, `int = floor(x)` (truncate, don't round), `ten = floor(x/10)`, `dec = floor(10x)`, `lead1 = len*10 + first digit`, `lead2 = len*100 + first two digits`, `length`.
- Encode with the m-estimate `(sum_y + 20*prior) / (n + 20)`, **cross-fitted 5 ways inside each training fold**. Validation/test rows use encodings fitted on the whole training fold. Keep raw columns and cents, drop the raw keys.
- sklearn `TargetEncoder(smooth="auto")` and extra clutter columns cost ~0.0009 vs the plain m-estimate at 20 (S6E11, alexchilton).
- Evidence (S6E11): XGB raw 0.94727 → +TE of int/decimal parts 0.94947 → +exact-value TE 0.95018 (evgendvorkin). LGBM raw 0.94639 → all groups 0.94930 (georgymamarin). LB +0.0012 then +0.0009 (alexchilton).
- S6E10 version of the same idea: numerics duplicated as factorized category columns (embedded for NNs, CTR/TE for GBDTs). Your own S6E10 best (CV 0.9608) already used a weaker form.

### 2. Very shallow GBDT, slow learning rate, many trees (+0.0009 to +0.0014)
- LGBM `num_leaves=4, learning_rate=0.03, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1`, ~1600 trees or early stopping on an **inner** split. XGB depth 1–2 as extra views.
- Evidence (S6E11): 63 leaves 0.94701 vs 4 leaves 0.94838 (LB +0.001); 4 leaves/1600 trees vs 15 leaves/600 trees +0.0013 on a holdout. Optuna drifted to 2 leaves with no extra gain.

### 3. GLM family as a second view (+0.0002 to +0.0006)
- Joint GLM: L2 logistic, C=0.01, on standardized log numerics, flags, one-hot categoricals, `SplineTransformer(n_knots=8, degree=3, knots="quantile")`, one-hot of every value/digit key seen ≥5 times (rarer values share a column), plus standardized TE logits.
- GLM-boost: XGB depth 2 or LGBM 4 leaves with `base_margin` / `init_score` = GLM logit (computed out-of-fold).
- Evidence (S6E11): H10+GLM fixed 80/20 blend +0.00022 OOF (5/5 folds), LB +0.00014. GLM-boost +0.00061 OOF (5/5), LB 0.95142.

### 4. Member diversity feeding a stack (+0.0006 to +0.0012 in S6E10)
- Families that added the most: RealMLP, TabM, TFMs (TabPFN/TabICL), CatBoost with all columns categorical. In S6E11 RealMLP alone got weight 0; TabM alone was weak but got a high stack weight.
- Accept a member only if its rank correlation with every existing member is < 0.9999 **and** nested CV gains ≥ +5e-5.
- "A new model family helps once; adding the same family again stops paying."

RealMLP (pytabkit `RealMLP_TD_Classifier`, yekenot S6E10): `n_ens=8, n_epochs=3, batch_size=288, lr=0.053, lr_sched="flat_anneal", wd=0.015, wd_sched="cos_log_15", first_layer_lr_factor=0.25, hidden_sizes=[512,256,128], act="silu", p_drop=0.05, p_drop_sched="pow5", embedding_size=5, max_one_hot_cat_size=2, plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=0.75, ls_eps=0.01, ls_eps_sched="sqrt_cos", tfms=["one_hot","median_center","robust_scale","smooth_clip","embedding","l2_normalize"]`. Inputs: numerics also as factorized category columns, count encodings, TE of a few combos inside folds. The S6E11 variant used 16 epochs, batch 512, lr 0.04, mish. Pass a holdout from the training rows as `X_val`, not the scored fold.

TabM (S6E11): `tabm-mini`, k=32, piecewise-linear embeddings, d_block 512 × 3, dropout 0.1, lr 2e-3, 40 epochs, patience 6.

### 5. Full refit for the test side (LB only, mixed evidence)
- One fit on all rows with `n_iter = 1.25 × mean best_iteration` and encoders refitted on all rows (alexchilton +0.00045 LB together with GLM). georgymamarin's proxy preferred averaging the 5 fold models. Test both ways and keep the CV-backed choice.

### 6. Stacker (small: +1e-5 to +3e-5)
- Logistic regression on logit-ranks `logit((rank - 0.5) / n)`, `tol=1e-6`, scored by **nested** CV over 3 outer seeds. C ≈ 0.05 for ~60 members, 0.01 for wide pools (C must shrink as the pool widens).
- Caruana hill climb ties it when done right: start with the best member, add with replacement, take weights at the **best** iteration, stop after ~20 stalled iterations, score fold-honestly (fit weights on 4 folds, score the 5th).
- Nonlinear, gated or AUC-proportional combiners lost 20–56e-5 in nested CV (S6E11, asteriospanoras).

### 7. Cheap extras
- Encoder seed bagging: +1.4e-5 to +6e-5.
- Label-free group profiles (group means of key numerics by categorical combos, deviation from the group mean, computed on train+test): +0.0001 (S6E11), and the "route ID" (exact Flight Distance) profile in S6E10.
- Zero-rating = "not applicable" in survey data (S6E10: wifi=0 → 88.7% satisfied vs 39.1% at 1).

## Things that did not help
- Original data as extra rows, or a teacher model's probability as a feature (S6E11: −0.0037, 5/5 folds down). Per-value lookup rates from the original are untested on their own. (S6E10 notebook pools did contain original-data teachers; test per competition.)
- All pairwise crosses, deep trees, kNN, GaussianNB/SVM filler models, frequency encoding of categoricals, Optuna on top of a good recipe, BPE tokens, TabPFN/TabICL added to an already wide pool.
- 10-fold encoders mixed with 5-fold members (OOF/test mismatch).
- Calibration for an AUC metric (AUC ignores monotone transforms).

## Validation hygiene (non-negotiable)
- One fixed fold file per competition (`StratifiedKFold(5, shuffle=True, random_state=SEED)` with the notebooks' `SEED`), shared by every member, so OOFs are stackable. Public notebooks mostly use seed 42; that only matters for public OOF banks, which we don't use.
- Never early-stop on the scored fold: it inflates OOF by ~+2–3e-5 per member and the stacker then rewards the most optimistic members. Use an inner split or a fixed tree count.
- Gate for a change: ≥ +5e-5 AUC and ≥ 4/5 folds improved. Canary check: shuffled labels must give AUC ≈ 0.5.
- Never stack a stack's OOF (pure nested-CV optimism).

## Leaderboard noise
- The public LB is ~20% of the test set. Two different files: public-gap SD ≈ 3e-4; for files with rank correlation ρ the SD scales with `sqrt(1 - ρ)`.
- Picking the best of K public probes inflates the score by ~σ·E[max of K normals] (~1.5σ for K=10).
- Board-steered steps: +5.0 public, 0.0 private. CV-steered: +0.8 public, +0.3 private. Across S6E1–S6E9 only 40 of 90 public top-10 spots stayed in the private top 10.
- Keep at least one CV-backed final submission.

## Competition retros

### S6E10 (airline satisfaction), finished 2026-10-31
Own best CV 0.9608 (LGBM-dominated GBDT ensemble on KBins + count + target encoding); rank 1646/3734. Winner 0.96318. Raw columns beat later FE variants; tuning lowered CV to 0.9598 (tuned on a different FE version). Missing: saved OOFs, NN/TFM members, hill climbing.

### S6E11 (credit card fraud), closes 2026-11-30
Start: LB 0.94677 (baseline CV ~0.9468), rank 2343/3274 on 2026-11-26. Best self-built public notebook ≈ LB 0.9515 (rank ~1090); everything ≥ 0.953 used public OOF banks or LB probing. Target: self-built ≥ 0.951. Retro goes here after the private LB.
- Phase 3 stub (2026-11-27), private LB pending. Final picks: pruned 3-member logistic stack (xgb_d2_te, glm_boost, baseline), CV 0.95067. A = fold-average test, public 0.95167; B = top-2 refitted on all rows, public 0.95173.
- What did not add to this library (blend Δ ≤ +2e-5 each): CatBoost with all columns as categories (single 0.94637), RealMLP (0.94993), TabM (0.94932), GLM-boost on LGBM-4 (0.95029, ρ 0.9995 with the XGB version), group-mean deviations (+2e-5 single). TE seed bagging over 3 cross-fit seeds **hurt**: −0.0032 on the screen.
- Lever 5 (full refit) gave +6e-5 public over the fold average at file ρ 0.9988. Pruning 5 → 3 members kept the CV (0.95066 → 0.95067).
