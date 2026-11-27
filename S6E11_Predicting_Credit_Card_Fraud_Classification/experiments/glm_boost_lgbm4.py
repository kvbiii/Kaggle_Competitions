"""GLM-boost, second view: LGBM 4 leaves on the TE features with the cross-fitted GLM logit as init_score."""

from common import SEED, Member, mode_from_argv, te_pipeline
from glm_boost import OFFSET, run
from lgbm4_te import PARAMS
from lightgbm import LGBMClassifier


def member() -> Member:
    """Build the boosting half; glm_boost.run adds the offset column per outer fold."""
    model = LGBMClassifier(objective="binary", metric="auc", random_state=SEED, n_jobs=32, verbosity=-1, **PARAMS)
    return Member("glm_boost_lgbm4", model, pipeline=te_pipeline(), config={"model": "lgbm + GLM init_score", "features": "raw + TE", **PARAMS}, offset_column=OFFSET)


if __name__ == "__main__":
    run(mode_from_argv(), member())
