"""XGB depth 2 on TE features plus label-free group profiles: deviation of key numerics from their category means (lever 7)."""

import pandas as pd
from common import CATEGORICAL, SEED, Member, mode_from_argv, run_member, te_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from xgb_d2_te import PARAMS
from xgboost import XGBClassifier

PROFILED = ["amount_usd", "distance_from_home_km", "velocity_score", "merchant_risk_score", "hours_since_last_txn", "account_balance_usd"]


class GroupProfiles(TransformerMixin, BaseEstimator):
    """Append `x - mean(x | group)` for every profiled column and group; means come from the training fold only."""

    def __init__(self, groups: list[str], columns: list[str]) -> None:
        """Store the grouping and profiled columns."""
        self.groups = groups
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "GroupProfiles":
        """Learn the per-category means of the profiled columns (labels unused)."""
        self.means_ = {g: X.groupby(X[g].astype(str))[self.columns].mean() for g in self.groups}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X plus the deviation columns; categories unseen in fit get NaN."""
        deviations = {}
        for group, means in self.means_.items():
            aligned = means.reindex(X[group].astype(str)).to_numpy()
            for j, column in enumerate(self.columns):
                deviations[f"{column}_GP_{group}_DEV"] = X[column].to_numpy() - aligned[:, j]
        return X.assign(**deviations)


def gp_pipeline() -> Pipeline:
    """The TE pipeline with group profiles inserted right after category alignment."""
    steps = te_pipeline().steps
    return Pipeline([steps[0], ("profiles", GroupProfiles(CATEGORICAL, PROFILED)), *steps[1:]])


def member() -> Member:
    """Build this idea's member."""
    model = XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", random_state=SEED, enable_categorical=True, device="cuda", **PARAMS)
    return Member("xgb_d2_te_gp", model, pipeline=gp_pipeline(), config={"model": "xgb", "features": "raw + TE + group-mean deviations", **PARAMS})


if __name__ == "__main__":
    run_member(member(), mode_from_argv())
