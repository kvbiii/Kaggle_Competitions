"""XGB depth 2 on the TE features: a second shallow view (playbook lever 2)."""

from common import SEED, Member, mode_from_argv, run_member, te_pipeline
from xgboost import XGBClassifier

PARAMS = {"n_estimators": 20_000, "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 5.0, "subsample": 0.8, "colsample_bytree": 0.5, "reg_lambda": 1.0, "max_bin": 255, "early_stopping_rounds": 300}

def member() -> Member:
    """Build this idea's member."""
    model = XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", random_state=SEED, enable_categorical=True, device="cuda", **PARAMS)
    return Member("xgb_d2_te", model, pipeline=te_pipeline(), config={"model": "xgb", "features": "raw + TE", **PARAMS})


if __name__ == "__main__":
    run_member(member(), mode_from_argv())
