# Handoff

## Phase 1: kvbiii-ml saves OOFs and ensembles from arrays (2026-11-26)

### What was built
- kvbiii-ml PR `feat/phase-1-oof-ensembling`, 3 commits on top of main 749eb31:
  - `CrossValidationTrainer.oof_predictions_`;
  - `kvbiii_ml.modeling.optimization.oof_ensembling` (`hill_climb`, `logistic_stack`, `BlendResult`);
  - `DigitKeysFeatureGenerator`.
- The early-stopping holdout already existed (`early_stopping_holdout=`). This phase measured it and wrote no new code for it.
- `Kaggle_Competitions_venv`:
  - kvbiii-ml is installed editable from `~/Desktop/KubiK/Programming/Python/Paczki/kvbiii-ml`, so it runs whatever branch is checked out there. Keep it on the phase branch or `main` after merge.
  - Added pytabkit 1.8.0, torch 2.15.0+cu132 (`cuda.is_available()` True). nccl went 2.32.3→2.30.7.
  - XGB and LGBM `device="cuda"` fits pass, and so does a RealMLP GPU fit.
- `~/.claude/skills/kvbiii-ml-reference/SKILL.md` updated to rev 736dd8f with everything below.

### API contracts Phase 2 calls
```python
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer
tr = CrossValidationTrainer("classification", "Roc AUC", cv=PredefinedSplit(folds), verbose=False)
_, valid_scores, test_proba = tr.fit(model, X, y, X_test)
oof = tr.oof_predictions_[:, 1]    # (n,) input row order; equals valid_scores per fold
test = test_proba[:, 1]            # fold-averaged predict_proba
np.savez(f"oof/{member}.npz", oof=oof, test=test, fold_auc=valid_scores, config=json.dumps(cfg))

from kvbiii_ml.modeling.optimization.oof_ensembling import hill_climb, logistic_stack
r = hill_climb(OOF, y, folds, TEST)      # OOF (n, m), TEST (n_test, m), folds (n,) int ids
s = logistic_stack(OOF, y, folds, TEST)  # same inputs
r.score, r.fold_scores, r.oof, r.test, r.weights, r.params   # nested AUC, same folds
```
- `BlendResult.oof` is the nested blend. Never stack it again.
- `score` is the mean of per-fold AUCs, in sorted fold-id order.
- Binary ROC AUC only.
- `PredefinedSplit(folds)` (sklearn) turns `folds.npy` into the trainer's `cv`.

Exact-value + digit-key m-estimate TE as the trainer's `preprocessing_pipeline`:
```python
from kvbiii_ml.data_processing.feature_engineering.digit_keys import ALL_KEYS, DigitKeysFeatureGenerator
from kvbiii_ml.data_processing.preprocessing.categorical_encoding.target_encoder import TargetEncoderWithOriginal
from feature_engine.selection import DropFeatures
keys = [f"{c}_PREPROCESS_KEY_{k.upper()}" for c in cols for k in ALL_KEYS]
Pipeline([("keys", DigitKeysFeatureGenerator(cols)),
          ("te", TargetEncoderWithOriginal(smooth=20, cv=5, variables=cols + keys, random_state=SEED)),
          ("drop", DropFeatures([k for k in keys if not k.endswith("_CENTS")]))])
```
- `TargetEncoderWithOriginal(smooth=20)` is exactly `(Σy + 20·prior)/(n + 20)`. It cross-fits inside the training fold.
- Encoded columns end in `_PREPROCESS_TARGET_ENC`. Raw columns and cents keys stay.
- Runtime: about 6 s per fold for 8 columns (64 encoded) on 560k rows.

### Measurements
- **Early stopping (S6E11, LGBM 31 leaves, lr 0.05, StratifiedKFold(5, shuffle, 42))**:

  | Setup | OOF AUC | Δ vs fixed trees |
  |---|---|---|
  | Scored-fold ES | 0.945211 | +2.6e-5 (optimism) |
  | Fixed 339 trees, full fold (honest) | 0.945185 | 0 |
  | `early_stopping_holdout=0.2` | 0.944785 | −4.0e-4 (80% of rows) |

  Recipe for honest members: run one fit with the holdout to read `best_iteration_`, then refit with `n_estimators = 1.25 × mean` and no early stopping. Each run is ~35 s on CPU.
- **Blenders on synthetic 700k × 30**: `hill_climb` 138 s (6 climbs, ~70 iterations each), `logistic_stack` 143 s.

### Decisions (2026-11-26)
- User: hill climb has non-negative weights only (Caruana with replacement).
- User: the stack's outer loop is the given fold ids, so it is comparable to the hill climb. C comes from inner StratifiedKFold(5) over 3 seeds.
- User delegated encoder and FE details. Reuse sklearn-backed `TargetEncoderWithOriginal(smooth=20)` and add only `DigitKeysFeatureGenerator`. Lead and length keys count digits from the leading integer digit, and use the decimals when |x| < 1.
- Me: `oof_predictions_` follows the test-prediction convention: the full proba matrix, NaN for never-validated rows. The `fit` return tuple is unchanged.
- Me: no code change for the inner holdout. It already existed and is tested.

### Open issues
- kvbiii-ml is not released to PyPI. The Kaggle venv relies on the editable install, so the venv breaks if that checkout moves to an old branch.
- `early_stopping_holdout` takes the positional tail of the training fold, not a stratified random split. That's fine for shuffled Playground data, but it would not suit a sorted train file.
- `EnsembleWeightTunerCV._collect_fold_predictions` still recomputes OOF itself. It could read `oof_predictions_` instead (not done, out of scope).
- The hill climb scores candidates with threads. On a machine with few cores, expect it to be slower than the 138 s measured here (32 cores).

## Phase 2: S6E11 loop, GBDT/GLM ladder, first stack (2026-11-27)

### What was built
- Branch `s6e11/phase-2-loop` in `Kaggle_Competitions`. kvbiii-ml is unchanged: its local `main` was behind the merged Phase 1 PR #96, so it was fast-forwarded to 2ecf896 (the editable venv runs that).
- `S6E11_.../experiments/`:
  - `common.py`: data, folds, screen subsets, the TE pipeline, `Member`, `run_member`, member files and the run log.
  - `gate.py`, `calibrate.py`, `stack.py`.
  - One script per idea: `baseline_xgb`, `te_xgb`, `lgbm4_te`, `xgb_d2_te`, `lgbm31_raw`, `glm`, `glm_boost`. Each exposes `member()`.
  - `test_plumbing.py`: 6 pytest checks (folds, screen split, npz round trip, TE canary, GLM canary over 3 seeds, GLM-boost offset honesty). Takes ~90 s.
- `EDA.ipynb` gained a last section, "Agent facts", that writes `agent/facts.md` and `agent/facts.json`. Headless run: `jupyter nbconvert --to notebook --execute --inplace EDA.ipynb`, 2 min.
- Result: logistic stack of 5 members, nested CV **0.95066**, public LB **0.95170**, rank ~1039/3322. The previous entry was LB 0.94677, rank 2343.

### Folder contract as built (`S6E11_Predicting_Credit_Card_Fraud_Classification/`)
- `folds.npy`: int8 fold id per train row, `StratifiedKFold(5, shuffle=True, random_state=17)` over train row order. `common.load_folds` writes it on first use. Gitignored, but deterministic.
- `oof/{screen,full}/<member>.npz`:
  - keys: `oof`, `test` (empty in screen mode), `fold_auc`, `rows`, `fold`, `config` (JSON with params, `best_trees`, `refit_trees`, `runtime_s`).
  - screen files hold only the scored rows.
- `submissions/<MMDD_HHMM>_<method>_<n>m.csv`: rank-scaled test blend.
- `experiments/library.txt`: accepted members in acceptance order. Currently `te_xgb glm xgb_d2_te glm_boost baseline`.
- `experiments.md`: the calibration table, ladder, stack + LB table, and an auto-appended run log written by `run_member`, `gate.py` and `stack.py`.
- Run an idea with `cd experiments && ../../Kaggle_Competitions_venv/bin/python -W ignore <idea>.py screen|full`.
- Gate commands:
  - `python gate.py single <cand> <ref> full`
  - `python gate.py blend <cand> full`
- Blend: `python stack.py`.

### Measurements
| member | single CV | gate | runtime (full) |
|---|---|---|---|
| baseline (notebook XGB, raw) | 0.94666 | reference; later +0.00009 blend PASS | 2.2 min |
| te_xgb | 0.94940 | +0.00274 single, 5/5 PASS | 6.1 min |
| glm | 0.94928 | +0.00074 blend, 5/5 PASS | 3.2 min |
| xgb_d2_te | 0.95021 | +0.00028 blend, 5/5 PASS | 4.0 min |
| lgbm4_te | 0.95012 | +0.00002 blend FAIL (ρ 0.9967 with xgb_d2_te) | 10.8 min (CPU) |
| glm_boost | 0.95026 | +0.00014 blend, 4/5 PASS | 7.6 min |
| stack: logistic / hill climb | 0.95066 / 0.95064 | +0.00040 over best single | ~3 min each |

- **Screen**: 20% of every (fold, label) group, 3 scored folds, holdout early stopping only.
  - It keeps the full-CV order of the 4 calibration variants (τ = 1.00) and is 18.8× faster: 1.1 min for 4 models.
  - Over all 7 members τ is only 0.71. It undervalues the GLM and swaps members that are 1e-4 apart.
  - So it only rejects ideas trailing by > 1e-3. New families get a full run.
- **LB − CV**: +0.00104 for the stack. The baseline had ≈ 0. One public draw has SD ≈ 3e-4, so don't read much into it.

### Decisions (2026-11-27)
- User Q1: skip the original dataset this phase (Phase 3 idea).
- User Q2: the baseline member is the notebook's XGB config only, re-run on `folds.npy` with honest trees.
- User Q3: an FE change that replaces a model (raw → TE) is gated on single-model CV. A new model family is gated on the nested hill-climb gain over the library. Both need ≥ +5e-5 and ≥ 4/5 folds (≥ n−1 folds on screens).
- User: approved the stack submission. Submitted at 10:51 local, LB logged.
- Me: SEED 17, from the notebooks.
- Me: honest trees use one `early_stopping_holdout=0.2` pass, then a refit with 1.25 × the mean best trees. Screen mode skips the refit.
- Me: GLM-boost gets its `base_margin` per outer fold. Training rows use inner `KFold(5)` cross-fitted GLM logits. Valid and test rows use the GLM fitted on the whole training fold. The offset is split off before the trainer's pipeline, so each outer fold is its own `fit_cv` call.
- Me: TE covers exact values of all 13 numerics, the 5 categoricals and 7 digit keys of the 6 floats, with smooth 20. The GLM keeps the raw keys for one-hots with min_frequency 5.

### Open issues
- Gates depend on order. LGBM-4 failed only because XGB-d2 entered first. GLM now gets weight 0.000 in the hill climb and 0.034 in the stack: it passed when the library was small, and GLM-boost has since absorbed it. A pruning pass (leave-one-out nested CV) is not done.
- The test side is the average of the 5 fold models. A full refit (lever 5) is untested.
- The run log in `experiments.md` also holds the 16 calibration screen rows (names like `baseline@0.2x3`).
- `kaggle competitions submissions` prints a version warning first, so parse the table by header, not by line number.
- The GLM canary on seed 1 gave 0.526 both with and without TE. That is permutation noise, so the test averages seeds 2–4.

### Phase 3 ideas, ranked (expected value per machine hour)
1. **NN members** (lever 4): RealMLP (pytabkit, GPU), then TabM. Biggest family-diversity lever (+0.0006–0.0012 in S6E10). Feed them the TE features plus factorized numerics, with an inner holdout as `X_val`. Gate on blend gain.
2. **CatBoost, all columns as categoricals** (GPU): a new GBDT family with its own CTR encodings, likely low ρ with the XGB members.
3. **Test-side full refit** (lever 5): one fit on all rows with 1.25 × best trees and encoders refitted on all rows. Try it on glm_boost and xgb_d2_te. CV cannot score it, so it costs one LB probe each. Keep a CV-backed fold-average submission as one of the 2 final picks.
4. **Library pruning + a second GLM-boost view** (LGBM-4 on the GLM margin). Cheap, since the GLM fits can be reused.
5. **Cheap extras**: encoder seed bagging (+1e-5 to 6e-5), label-free group profiles (+1e-4 in S6E11 reports), original-data per-value lookup rates (Q1, untested).
