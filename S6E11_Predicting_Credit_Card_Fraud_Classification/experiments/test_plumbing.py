"""Plumbing tests for the S6E11 experiment loop: folds, screen subsets, member files, encoder canary."""

import numpy as np
import pandas as pd
from common import (
    CATEGORICAL,
    NUMERIC,
    load_data,
    load_member,
    make_folds,
    save_member,
    screen_split,
    te_pipeline,
)
from kvbiii_ml.modeling.training.cross_validation import CrossValidationTrainer
from lightgbm import LGBMClassifier
from sklearn.model_selection import PredefinedSplit


def test_make_folds_is_deterministic_and_stratified() -> None:
    y = np.r_[np.ones(500), np.zeros(9500)]
    folds = make_folds(y)
    assert np.array_equal(folds, make_folds(y))
    assert sorted(np.unique(folds)) == [0, 1, 2, 3, 4]
    assert all(y[folds == k].sum() == 100 for k in range(5))


def test_screen_split_keeps_fold_ids_and_marks_unscored_folds() -> None:
    y = np.r_[np.ones(1000), np.zeros(19000)]
    folds = make_folds(y)
    rows, screen_folds = screen_split(folds, y, frac=0.25, n_folds=3)
    assert np.array_equal(rows, np.sort(rows))
    assert len(rows) == 5000
    assert y[rows].sum() == 250
    assert np.array_equal(screen_folds >= 0, folds[rows] < 3)
    assert np.array_equal(screen_folds[screen_folds >= 0], folds[rows][folds[rows] < 3])
    assert np.array_equal(rows, screen_split(folds, y, frac=0.25, n_folds=3)[0])


def test_member_round_trip(tmp_path) -> None:
    oof = np.linspace(0, 1, 10)
    arrays = {"oof": oof, "test": oof[:4], "fold_auc": np.array([0.9, 0.8]), "rows": np.arange(10), "fold": np.zeros(10, int)}
    save_member(tmp_path / "m.npz", arrays, config={"a": 1})
    member = load_member(tmp_path / "m.npz")
    assert np.array_equal(member["oof"], oof)
    assert member["config"] == {"a": 1}


def test_te_pipeline_canary_shuffled_labels_give_chance_auc() -> None:
    train, _ = load_data()
    sample = train.sample(60_000, random_state=0)
    X = sample[NUMERIC + CATEGORICAL]
    y = pd.Series(np.random.default_rng(0).permutation(sample["is_fraud"].to_numpy()), index=X.index)
    trainer = CrossValidationTrainer("classification", "Roc AUC", cv=PredefinedSplit(make_folds(y.to_numpy())), preprocessing_pipeline=te_pipeline(), verbose=False)
    _, valid, _ = trainer.fit(LGBMClassifier(n_estimators=100, num_leaves=15, verbosity=-1), X, y)
    assert abs(np.mean(valid) - 0.5) < 0.015
