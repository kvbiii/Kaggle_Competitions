"""CatBoost (GPU) on raw columns plus a string-category copy of every numeric and flag: its own CTR view (lever 4)."""

import pandas as pd
from catboost import CatBoostClassifier
from common import CATEGORICAL, FLAGS, NUMERIC, SEED, Member, mode_from_argv, run_member
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

COPIED = NUMERIC + FLAGS
CAT_SUFFIX = "_CAT"
PARAMS = {"n_estimators": 20_000, "learning_rate": 0.06, "depth": 6, "l2_leaf_reg": 3.0, "early_stopping_rounds": 300}


def add_category_copies(X: pd.DataFrame) -> pd.DataFrame:
    """Append a string copy of every numeric and flag column, so CatBoost builds CTRs of exact values."""
    return X.assign(**{f"{c}{CAT_SUFFIX}": X[c].astype(str) for c in COPIED})


def member() -> Member:
    """Build this idea's member; use_best_model=False keeps the refit from trimming trees on the scored fold."""
    cat_features = CATEGORICAL + [f"{c}{CAT_SUFFIX}" for c in COPIED]
    model = CatBoostClassifier(task_type="GPU", eval_metric="AUC", random_seed=SEED, cat_features=cat_features, use_best_model=False, verbose=0, **PARAMS)
    pipeline = Pipeline([("copies", FunctionTransformer(add_category_copies))])
    return Member("catboost_cat", model, pipeline=pipeline, config={"model": "catboost", "features": "raw + string copies of numerics/flags", **PARAMS})


if __name__ == "__main__":
    run_member(member(), mode_from_argv())
