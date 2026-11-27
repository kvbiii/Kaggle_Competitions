"""TE-XGB: baseline XGB params on raw columns plus exact-value and digit-key m-estimate TE."""

from baseline_xgb import POSITIVE_RATE, TRAIN_ROWS, xgb_baseline_params
from common import FEATURES, SEED, Member, mode_from_argv, run_member, te_pipeline
from xgboost import XGBClassifier


def member() -> Member:
    """Build this idea's member."""
    params = xgb_baseline_params(TRAIN_ROWS, len(FEATURES), POSITIVE_RATE)
    model = XGBClassifier(objective="binary:logistic", eval_metric="auc", random_state=SEED, enable_categorical=True, device="cuda", **params)
    return Member("te_xgb", model, pipeline=te_pipeline(), config={"model": "xgb", "features": "raw + TE(values, digit keys)", **params})


if __name__ == "__main__":
    run_member(member(), mode_from_argv())
