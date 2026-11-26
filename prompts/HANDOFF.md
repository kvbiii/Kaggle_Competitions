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
