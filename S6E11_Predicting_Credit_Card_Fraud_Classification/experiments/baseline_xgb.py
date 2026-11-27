"""Baseline member: the XGB config of `Modeling_Baseline.ipynb` on raw columns, honest trees."""

import numpy as np
from common import FEATURES, SEED, Member, aligned_pipeline, mode_from_argv, run_member
from xgboost import XGBClassifier

TRAIN_ROWS = 699_960
POSITIVE_RATE = 0.0167


def xgb_baseline_params(n_rows: int, n_features: int, positive_rate: float) -> dict:
    """Return `xgb_baseline_params` of the baseline notebook for a binary task of this size."""
    hessian = positive_rate * (1 - positive_rate)
    min_child = int(np.clip(np.sqrt(n_rows) / 4, 20, 500))
    leaf_hessian = hessian * min_child
    learning_rate = float(np.clip(0.01 * np.sqrt(n_rows / 1e6), 0.01, 0.05))
    return {
        "n_estimators": int(100 / learning_rate),
        "learning_rate": learning_rate,
        "tree_method": "hist",
        "grow_policy": "depthwise",
        "max_depth": int(np.clip(round(np.log10(n_rows)) + 1, 4, 8)) - 1,
        "min_child_weight": leaf_hessian,
        "gamma": 0.02 * leaf_hessian,
        "subsample": 0.85,
        "colsample_bytree": float(np.clip(1 - 0.1 * np.log2(max(n_features, 8) / 8), 0.4, 1.0)),
        "reg_lambda": leaf_hessian,
        "reg_alpha": 0.10 * leaf_hessian,
        "max_bin": int(np.clip(n_rows // 20, 32, 255)),
        "early_stopping_rounds": int(2 / learning_rate),
    }


def member() -> Member:
    """Build this idea's member."""
    params = xgb_baseline_params(TRAIN_ROWS, len(FEATURES), POSITIVE_RATE)
    model = XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=SEED, enable_categorical=True, device="cuda", **params)
    return Member("baseline", model, pipeline=aligned_pipeline(), config={"model": "xgb", "features": "raw", **params})


if __name__ == "__main__":
    run_member(member(), mode_from_argv())
