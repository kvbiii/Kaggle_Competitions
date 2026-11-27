"""Shared plumbing for S6E11 experiments: data, fixed folds, screen subsets, member files, run log."""

import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from feature_engine.selection import DropFeatures
from kvbiii_ml.data_processing.feature_engineering.categorical_aligner import (
    CategoricalAligner,
)
from kvbiii_ml.data_processing.feature_engineering.digit_keys import (
    ALL_KEYS,
    DigitKeysFeatureGenerator,
)
from kvbiii_ml.data_processing.preprocessing.categorical_encoding.target_encoder import (
    TargetEncoderWithOriginal,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import PredefinedSplit, StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OOF_DIR = ROOT / "oof"
SUBMISSION_DIR = ROOT / "submissions"
FOLDS_FILE = ROOT / "folds.npy"
LOG_FILE = ROOT / "experiments.md"

SEED = 17
N_FOLDS = 5
TARGET = "is_fraud"
FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "account_balance_usd", "velocity_score", "merchant_risk_score"]
INTS = ["txn_count_last_24h", "card_age_months", "customer_age", "cvv_retry_count", "time_of_day_hour", "day_of_week", "prior_disputes"]
NUMERIC = FLOATS + INTS
FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
CATEGORICAL = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FEATURES = NUMERIC + FLAGS + CATEGORICAL

SCREEN = {"frac": 0.2, "n_folds": 3}
ES_HOLDOUT = 0.2
REFIT_FACTOR = 1.25
TE_SMOOTH = 20
ES_PARAMS = ("early_stopping_rounds", "early_stopping_round")


@dataclass
class Member:
    """One OOF-library candidate: a model, its input columns and an optional per-fold pipeline."""

    name: str
    model: BaseEstimator
    features: list[str] = field(default_factory=lambda: list(FEATURES))
    pipeline: Pipeline | None = None
    config: dict = field(default_factory=dict)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test with boolean flags as ints and an int target."""
    train = pd.read_csv(DATA_DIR / "train.csv", index_col="id")
    test = pd.read_csv(DATA_DIR / "test.csv", index_col="id")
    train[FLAGS] = train[FLAGS].astype(int)
    test[FLAGS] = test[FLAGS].astype(int)
    train[TARGET] = train[TARGET].astype(int)
    return train, test


def make_folds(y: np.ndarray) -> np.ndarray:
    """Return the fold id of every row from StratifiedKFold(5, shuffle, SEED)."""
    folds = np.empty(len(y), dtype=np.int8)
    for k, (_, valid) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)):
        folds[valid] = k
    return folds


def load_folds(y: np.ndarray) -> np.ndarray:
    """Load `folds.npy`, writing it on first use; fail if it does not match the train rows."""
    if not FOLDS_FILE.exists():
        np.save(FOLDS_FILE, make_folds(y))
    folds = np.load(FOLDS_FILE)
    if len(folds) != len(y):
        raise ValueError(f"folds.npy has {len(folds)} rows, train has {len(y)}.")
    return folds


def screen_split(folds: np.ndarray, y: np.ndarray, frac: float, n_folds: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Pick a screen subset: `frac` of every (fold, label) group, scoring only folds below `n_folds`.

    Returns:
        tuple[np.ndarray, np.ndarray]: Sorted row positions, and their fold ids with -1 for rows
            that only ever train (PredefinedSplit convention).
    """
    rng = np.random.default_rng(SEED)
    groups = folds.astype(np.int64) * 2 + y.astype(np.int64)
    picked = [rng.choice(idx, round(frac * len(idx)), replace=False) for idx in (np.flatnonzero(groups == g) for g in np.unique(groups))]
    rows = np.sort(np.concatenate(picked))
    return rows, np.where(folds[rows] < n_folds, folds[rows], -1)


def te_pipeline(numeric: list[str] = NUMERIC, floats: list[str] = FLOATS) -> Pipeline:
    """Aligned categoricals, digit keys of floats, cross-fitted m-estimate TE of values and keys."""
    keys = [f"{c}_PREPROCESS_KEY_{k.upper()}" for c in floats for k in ALL_KEYS]
    return Pipeline([
        ("align", CategoricalAligner(variables=CATEGORICAL, warn_on_unknown=False)),
        ("keys", DigitKeysFeatureGenerator(floats)),
        ("te", TargetEncoderWithOriginal(smooth=TE_SMOOTH, cv=N_FOLDS, variables=numeric + CATEGORICAL + keys, random_state=SEED)),
        ("drop", DropFeatures([k for k in keys if not k.endswith("_CENTS")])),
    ])


def aligned_pipeline() -> Pipeline:
    """Raw columns with categoricals cast to a shared category set."""
    return Pipeline([("align", CategoricalAligner(variables=CATEGORICAL, warn_on_unknown=False))])


def save_member(path: Path, arrays: dict[str, np.ndarray], config: dict) -> None:
    """Write a member's arrays and JSON config to one `.npz`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays, config=json.dumps(config))


def load_member(path: Path) -> dict:
    """Read a member `.npz` back into a dict with the config decoded."""
    with np.load(path) as data:
        member = {k: data[k] for k in data.files}
    member["config"] = json.loads(str(member["config"]))
    return member


def member_path(name: str, mode: str) -> Path:
    """Return the `.npz` path of a member in `screen` or `full` mode."""
    return OOF_DIR / mode / f"{name}.npz"


def log_line(text: str) -> None:
    """Append one line to the run log at the end of `experiments.md` and echo it."""
    with LOG_FILE.open("a") as handle:
        handle.write(text + "\n")
    print(text)


def _es_param(model: BaseEstimator) -> str | None:
    """Return the name of the model's active early-stopping parameter, if any."""
    params = model.get_params()
    return next((p for p in ES_PARAMS if params.get(p)), None)


def _best_trees(model: BaseEstimator) -> int:
    """Return the number of trees at the best iteration of a fitted XGB or LGBM model."""
    if hasattr(model, "best_iteration_"):
        return int(model.best_iteration_)
    return int(model.best_iteration) + 1


def _fit(member: Member, model: BaseEstimator, data: tuple, holdout: float | None) -> tuple[CrossValidationTrainer, np.ndarray]:
    """Fit one CV pass of `model` on `(X, y, X_test or None, fold ids)`; return it and the test proba."""
    X, y, X_test, folds = data
    trainer = CrossValidationTrainer("classification", "Roc AUC", cv=PredefinedSplit(folds), preprocessing_pipeline=member.pipeline, verbose=False, early_stopping_holdout=holdout)
    _, _, test_proba = trainer.fit(model, X, y, X_test)
    return trainer, np.empty(0) if test_proba is None else np.asarray(test_proba)[:, 1]


def run_member(member: Member, mode: str) -> dict:
    """
    Run a member in `screen` or `full` mode, save its `.npz` and log one line.

    - Screen: SCREEN subset, early stopping on the inner holdout, no refit.
    - Full: all rows and 5 folds; with early stopping configured, one holdout pass reads the best
      tree count, then a refit on whole training folds uses REFIT_FACTOR x the mean, no ES.
    """
    train, test = load_data()
    y_all = train[TARGET].to_numpy()
    folds = load_folds(y_all)
    rows, fold_ids = screen_split(folds, y_all, **SCREEN) if mode == "screen" else (np.arange(len(train)), folds)
    X_test = test[member.features] if mode == "full" else None
    data = (train.iloc[rows][member.features], train.iloc[rows][TARGET], X_test, fold_ids)
    start = time.perf_counter()
    es = _es_param(member.model)
    trainer, test_pred = _fit(member, member.model, data, ES_HOLDOUT if es else None)
    config = {**member.config, "mode": mode, "screen": SCREEN if mode == "screen" else None}
    if es:
        config["best_trees"] = [_best_trees(m) for m in trainer.fitted_estimators_]
    if es and mode == "full":
        n_trees = math.ceil(REFIT_FACTOR * np.mean(config["best_trees"]))
        config["refit_trees"] = n_trees
        trainer, test_pred = _fit(member, clone(member.model).set_params(n_estimators=n_trees, **{es: None}), data, None)
    runtime = time.perf_counter() - start
    scored = fold_ids >= 0
    fold_auc = np.asarray(trainer.valid_scores_)
    config["runtime_s"] = round(runtime, 1)
    arrays = {"oof": trainer.oof_predictions_[scored, 1], "test": test_pred, "fold_auc": fold_auc, "rows": rows[scored], "fold": fold_ids[scored]}
    save_member(member_path(member.name, mode), arrays, config)
    folds_text = " ".join(f"{a:.5f}" for a in fold_auc)
    log_line(f"| {datetime.now():%m-%d %H:%M} | {member.name} | {mode} | {fold_auc.mean():.5f} | {folds_text} | {runtime / 60:.1f} min | {config.get("refit_trees", config.get("best_trees", "-"))} |")
    return {"fold_auc": fold_auc, "config": config}


def mode_from_argv() -> str:
    """Read `screen` or `full` from the command line (default `screen`)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode not in {"screen", "full"}:
        raise SystemExit("usage: python <idea>.py [screen|full]")
    return mode
