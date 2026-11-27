"""GLM member: L2 logistic on log numerics, splines, one-hots of frequent values/digit keys and TE logits."""

import numpy as np
from common import (
    CATEGORICAL,
    FEATURES,
    FLAGS,
    FLOATS,
    N_FOLDS,
    NUMERIC,
    SEED,
    TE_SMOOTH,
    Member,
    mode_from_argv,
    run_member,
)
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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)

C = 0.01
MAX_ITER = 3000
MIN_FREQUENCY = 5
N_KNOTS = 8
PROBA_CLIP = 1e-6
LOW_CARD_INTS = ["txn_count_last_24h", "cvv_retry_count", "time_of_day_hour", "day_of_week", "prior_disputes"]
SMOOTH_COLS = [*FLOATS, "card_age_months", "customer_age"]
KEYS = [f"{c}_PREPROCESS_KEY_{k.upper()}" for c in FLOATS for k in ALL_KEYS]
TE_COLS = [f"{c}_PREPROCESS_TARGET_ENC" for c in NUMERIC + CATEGORICAL + KEYS]


def te_logit(p: np.ndarray) -> np.ndarray:
    """Map encoded rates to the logit scale."""
    p = np.clip(p, PROBA_CLIP, 1 - PROBA_CLIP)
    return np.log(p / (1 - p))


def glm_pipeline() -> Pipeline:
    """Per-fold pandas stage: aligned categoricals, digit keys kept, cross-fitted TE of values and keys."""
    return Pipeline([
        ("align", CategoricalAligner(variables=CATEGORICAL, warn_on_unknown=False)),
        ("keys", DigitKeysFeatureGenerator(FLOATS)),
        ("te", TargetEncoderWithOriginal(smooth=TE_SMOOTH, cv=N_FOLDS, variables=NUMERIC + CATEGORICAL + KEYS, random_state=SEED)),
    ])


def glm_estimator() -> Pipeline:
    """Sparse design matrix plus L2 logistic regression."""
    design = ColumnTransformer([
        ("log", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), NUMERIC),
        ("spline", SplineTransformer(n_knots=N_KNOTS, degree=3, knots="quantile"), SMOOTH_COLS),
        ("flags", "passthrough", FLAGS),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL + LOW_CARD_INTS),
        ("values", OneHotEncoder(min_frequency=MIN_FREQUENCY, handle_unknown="infrequent_if_exist"), SMOOTH_COLS + KEYS),
        ("te", make_pipeline(FunctionTransformer(te_logit), StandardScaler()), TE_COLS),
    ], sparse_threshold=1.0)
    return make_pipeline(design, LogisticRegression(C=C, max_iter=MAX_ITER))


def member() -> Member:
    """Build this idea's member."""
    config = {"model": "logistic", "C": C, "features": "log numerics, splines, flags, one-hot cats/low-card ints, one-hot values+keys (>=5), TE logits"}
    return Member("glm", glm_estimator(), features=list(FEATURES), pipeline=glm_pipeline(), config=config)


if __name__ == "__main__":
    run_member(member(), mode_from_argv())
