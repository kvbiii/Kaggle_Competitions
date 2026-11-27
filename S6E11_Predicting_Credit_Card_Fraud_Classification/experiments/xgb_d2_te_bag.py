"""XGB depth 2 with encoder seed bagging: training-row TE averaged over several cross-fit seeds (lever 7)."""

import numpy as np
import pandas as pd
from common import SEED, TE_SMOOTH, Member, mode_from_argv, run_member, te_pipeline
from kvbiii_ml.data_processing.preprocessing.categorical_encoding.target_encoder import (
    TargetEncoderWithOriginal,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from xgb_d2_te import PARAMS
from xgboost import XGBClassifier

SEEDS = (SEED, SEED + 1, SEED + 2)
ENC_SUFFIX = "_PREPROCESS_TARGET_ENC"


class BaggedTargetEncoder(TransformerMixin, BaseEstimator):
    """Cross-fitted m-estimate TE whose training-row encodings are averaged over `seeds`; valid/test use the whole fold."""

    def __init__(self, variables: list[str], seeds: tuple[int, ...] = SEEDS, smooth: float = TE_SMOOTH, cv: int = 5) -> None:
        """Store the encoder settings."""
        self.variables = variables
        self.seeds = seeds
        self.smooth = smooth
        self.cv = cv

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaggedTargetEncoder":
        """Fit the per-seed encoders (fit_transform does the work)."""
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, **_: object) -> pd.DataFrame:
        """Return X plus the seed-averaged cross-fitted encodings of its own rows."""
        encoders = [TargetEncoderWithOriginal(smooth=self.smooth, cv=self.cv, variables=self.variables, random_state=s) for s in self.seeds]
        frames = [encoder.fit_transform(X, y) for encoder in encoders]
        self.encoder_ = encoders[0]
        encoded = [c for c in frames[0].columns if c.endswith(ENC_SUFFIX)]
        out = frames[0].copy()
        out[encoded] = np.mean([frame[encoded].to_numpy() for frame in frames], axis=0)
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode new rows with the whole-fold statistics (identical for every seed)."""
        return self.encoder_.transform(X)


def bagged_pipeline() -> Pipeline:
    """The TE pipeline with its encoder swapped for the seed-bagged one."""
    pipeline = te_pipeline()
    return pipeline.set_params(te=BaggedTargetEncoder(variables=pipeline.named_steps["te"].variables))


def member() -> Member:
    """Build this idea's member."""
    model = XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", random_state=SEED, enable_categorical=True, device="cuda", **PARAMS)
    return Member("xgb_d2_te_bag", model, pipeline=bagged_pipeline(), config={"model": "xgb", "features": f"raw + TE bagged over seeds {SEEDS}", **PARAMS})


if __name__ == "__main__":
    run_member(member(), mode_from_argv())
