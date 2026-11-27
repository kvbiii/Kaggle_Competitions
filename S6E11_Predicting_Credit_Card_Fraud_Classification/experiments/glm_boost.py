"""GLM-boost: XGB depth 2 on TE features with the cross-fitted GLM logit as base_margin (lever 3)."""

import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
from common import (
    ES_HOLDOUT,
    FEATURES,
    N_FOLDS,
    REFIT_FACTOR,
    SCREEN,
    SEED,
    TARGET,
    Member,
    best_trees,
    es_param,
    fit_cv,
    load_data,
    load_folds,
    log_line,
    member_path,
    mode_from_argv,
    save_member,
    screen_split,
    te_pipeline,
)
from glm import glm_estimator, glm_pipeline
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgb_d2_te import PARAMS
from xgboost import XGBClassifier

NAME = "glm_boost"
OFFSET = "glm_offset"


def glm_offsets(X: pd.DataFrame, y: pd.Series, others: list[pd.DataFrame]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return inner-KFold cross-fitted GLM logits for the rows of X and full-fit logits for each frame in `others`."""
    model = Pipeline([("pre", glm_pipeline()), ("glm", glm_estimator())])
    own = cross_val_predict(model, X, y, cv=KFold(N_FOLDS, shuffle=True, random_state=SEED), method="decision_function", n_jobs=N_FOLDS)
    if not others:
        return own, []
    fitted = clone(model).fit(X, y)
    return own, [fitted.decision_function(frame) for frame in others]


def member() -> Member:
    """Build the boosting half of GLM-boost; the offset column is added per outer fold."""
    model = XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", random_state=SEED, enable_categorical=True, device="cuda", **PARAMS)
    return Member(NAME, model, pipeline=te_pipeline(), config={"model": "xgb + GLM base_margin", "features": "raw + TE", **PARAMS}, offset_column=OFFSET)


def fold_data(X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame | None, valid: np.ndarray) -> tuple:
    """Attach the outer fold's offsets: cross-fitted on its training rows, full-fit on valid and test rows."""
    others = [X[valid]] if X_test is None else [X[valid], X_test]
    own, applied = glm_offsets(X[~valid], y[~valid], others)
    offsets = np.empty(len(X))
    offsets[~valid], offsets[valid] = own, applied[0]
    test_k = None if X_test is None else X_test.assign(**{OFFSET: applied[1]})
    return X.assign(**{OFFSET: offsets}), y, test_k, np.where(valid, 0, -1)


def run(mode: str, cand: Member | None = None) -> None:
    """Holdout pass per outer fold to read best trees, then (full) a fixed-tree refit; save and log (default member: glm_boost)."""
    train, test = load_data()
    y_all = train[TARGET].to_numpy()
    folds = load_folds(y_all)
    rows, fold_ids = screen_split(folds, y_all, **SCREEN) if mode == "screen" else (np.arange(len(train)), folds)
    X, y = train.iloc[rows][FEATURES], train.iloc[rows][TARGET]
    X_test = test[FEATURES] if mode == "full" else None
    cand, start = cand or member(), time.perf_counter()
    scored = np.unique(fold_ids[fold_ids >= 0])
    datasets = [fold_data(X, y, X_test, fold_ids == k) for k in scored]
    passes = [fit_cv(cand, cand.model, data, ES_HOLDOUT) for data in datasets]
    config = {**cand.config, "mode": mode, "screen": SCREEN if mode == "screen" else None, "best_trees": [best_trees(t.fitted_estimators_[0]) for t, _ in passes]}
    if mode == "full":
        config["refit_trees"] = math.ceil(REFIT_FACTOR * np.mean(config["best_trees"]))
        refit = clone(cand.model).set_params(n_estimators=config["refit_trees"], **{es_param(cand.model): None})
        passes = [fit_cv(cand, refit, data, None) for data in datasets]
    oof = np.full(len(rows), np.nan)
    for k, (trainer, _) in zip(scored, passes, strict=True):
        oof[fold_ids == k] = trainer.oof_predictions_[fold_ids == k, 1]
    fold_auc = np.array([trainer.valid_scores_[0] for trainer, _ in passes])
    test_pred = np.mean([t for _, t in passes], axis=0) if mode == "full" else np.empty(0)
    runtime = time.perf_counter() - start
    config["runtime_s"] = round(runtime, 1)
    keep = fold_ids >= 0
    save_member(member_path(cand.name, mode), {"oof": oof[keep], "test": test_pred, "fold_auc": fold_auc, "rows": rows[keep], "fold": fold_ids[keep]}, config)
    log_line(f"| {datetime.now():%m-%d %H:%M} | {cand.name} | {mode} | {fold_auc.mean():.5f} | {' '.join(f'{a:.5f}' for a in fold_auc)} | {runtime / 60:.1f} min | {config.get('refit_trees', config['best_trees'])} |")


if __name__ == "__main__":
    run(mode_from_argv())
