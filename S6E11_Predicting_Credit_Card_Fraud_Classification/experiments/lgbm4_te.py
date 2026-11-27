"""LGBM with 4 leaves, slow learning rate and many trees on the TE features (playbook lever 2)."""

from common import SEED, Member, mode_from_argv, run_member, te_pipeline
from lightgbm import LGBMClassifier

PARAMS = {"n_estimators": 20_000, "learning_rate": 0.03, "num_leaves": 4, "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.5, "reg_lambda": 1.0, "early_stopping_round": 300}

def member() -> Member:
    """Build this idea's member."""
    model = LGBMClassifier(objective="binary", metric="auc", random_state=SEED, n_jobs=32, verbosity=-1, **PARAMS)
    return Member("lgbm4_te", model, pipeline=te_pipeline(), config={"model": "lgbm", "features": "raw + TE", **PARAMS})


if __name__ == "__main__":
    run_member(member(), mode_from_argv())
