"""LGBM 31 leaves on raw columns: a known-weaker contrast for screen calibration."""

from common import SEED, Member, aligned_pipeline, mode_from_argv, run_member
from lightgbm import LGBMClassifier

PARAMS = {"n_estimators": 5_000, "learning_rate": 0.05, "num_leaves": 31, "early_stopping_round": 100}

def member() -> Member:
    """Build this idea's member."""
    model = LGBMClassifier(objective="binary", metric="auc", random_state=SEED, n_jobs=32, verbosity=-1, **PARAMS)
    return Member("lgbm31_raw", model, pipeline=aligned_pipeline(), config={"model": "lgbm", "features": "raw", **PARAMS})


if __name__ == "__main__":
    run_member(member(), mode_from_argv())
