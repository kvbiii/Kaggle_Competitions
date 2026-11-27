"""
Full refit for the test side (lever 5): one fit on all train rows with the member's refit trees.

Usage:
    python refit.py <member> [<member> ...]     (writes oof/full/<member>_refit.npz, test only)

Encoders are refitted on all rows; GLM-boost members get inner-KFold cross-fitted GLM offsets on
train and a full-fit GLM offset on test. CV cannot score this, only the LB can.
"""

import copy
import sys
import time
from datetime import datetime
from importlib import import_module

import numpy as np
from common import (
    TARGET,
    es_param,
    load_data,
    load_member,
    log_line,
    member_path,
    save_member,
)
from glm_boost import glm_offsets
from kvbiii_ml.modeling.training.base_trainer import (
    BaseTrainer,
    FitRows,
    predict_with_offset,
)
from sklearn.base import clone

SCRIPTS = {"baseline": "baseline_xgb"}


def refit(name: str) -> np.ndarray:
    """Fit member `name` once on all train rows and return its positive-class test probabilities."""
    cand = import_module(SCRIPTS.get(name, name)).member()
    config = load_member(member_path(name, "full"))["config"]
    train, test = load_data()
    X, y, X_test = train[cand.features], train[TARGET], test[cand.features]
    start = time.perf_counter()
    model = copy.deepcopy(cand.model)
    if "refit_trees" in config:
        model.set_params(n_estimators=config["refit_trees"], **{es_param(model): None})
    offset, offset_test = glm_offsets(X, y, [X_test]) if cand.offset_column else (None, [None])
    pipeline = clone(cand.pipeline) if cand.pipeline else None
    X_fit = pipeline.fit_transform(X, y) if pipeline else X
    X_pred = pipeline.transform(X_test) if pipeline else X_test
    fitted = BaseTrainer.fit_estimator(model, FitRows(X_fit, y, offset=offset), None, verbose=False, eval_on_train=False)
    proba = predict_with_offset(fitted, X_pred, offset_test[0], proba=True) if offset is not None else fitted.predict_proba(X_pred)
    runtime = time.perf_counter() - start
    save_member(member_path(f"{name}_refit", "full"), {"test": np.asarray(proba)[:, 1]}, {**config, "refit": "all rows", "runtime_s": round(runtime, 1)})
    log_line(f"| {datetime.now():%m-%d %H:%M} | REFIT {name} (all rows) | full | test only | {config.get('refit_trees', '-')} trees | {runtime / 60:.1f} min | - |")
    return np.asarray(proba)[:, 1]


if __name__ == "__main__":
    for member_name in sys.argv[1:]:
        refit(member_name)
