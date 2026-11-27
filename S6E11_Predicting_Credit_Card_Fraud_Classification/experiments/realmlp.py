"""RealMLP (pytabkit, GPU) on the TE features plus integer columns as categories (lever 4, S6E10 config)."""

import pandas as pd
from common import INTS, SEED, Member, mode_from_argv, run_member, te_pipeline
from kvbiii_ml.data_processing.feature_engineering.categorical_aligner import (
    CategoricalAligner,
)
from pytabkit import RealMLP_TD_Classifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

INT_CATS = [f"{c}_CAT" for c in INTS]
VAL_METRIC = "1-auc_ovr"
PARAMS = {
    "n_ens": 8, "n_epochs": 3, "batch_size": 288, "lr": 0.053, "lr_sched": "flat_anneal", "wd": 0.015, "wd_sched": "cos_log_15",
    "first_layer_lr_factor": 0.25, "hidden_sizes": [512, 256, 128], "act": "silu", "p_drop": 0.05, "p_drop_sched": "pow5",
    "embedding_size": 5, "max_one_hot_cat_size": 2, "plr_hidden_1": 16, "plr_hidden_2": 8, "plr_act_name": "gelu", "plr_lr_factor": 0.1151,
    "plr_sigma": 0.75, "ls_eps": 0.01, "ls_eps_sched": "sqrt_cos", "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
}


def add_int_categories(X: pd.DataFrame) -> pd.DataFrame:
    """Append a string copy of every integer column (label-free), to be embedded as a category."""
    return X.assign(**{f"{c}_CAT": X[c].astype(str) for c in INTS})


def nn_pipeline() -> Pipeline:
    """TE pipeline, then integer copies aligned to the training fold's category set."""
    return Pipeline([*te_pipeline().steps, ("int_copies", FunctionTransformer(add_int_categories)), ("align_copies", CategoricalAligner(variables=INT_CATS, warn_on_unknown=False))])


def member() -> Member:
    """Build this idea's member: the inner holdout picks the epoch, n_refit=1 refits on the whole training fold."""
    model = RealMLP_TD_Classifier(device="cuda:0", random_state=SEED, verbosity=0, n_refit=1, val_metric_name=VAL_METRIC, **PARAMS)
    return Member("realmlp", model, pipeline=nn_pipeline(), config={"model": "realmlp", "features": "raw + TE + int categories", **PARAMS}, val_holdout=True)


if __name__ == "__main__":
    run_member(member(), mode_from_argv())
