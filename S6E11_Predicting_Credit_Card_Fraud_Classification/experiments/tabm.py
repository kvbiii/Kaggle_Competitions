"""TabM-mini (pytabkit, GPU) with piecewise-linear embeddings on the RealMLP inputs (lever 4, S6E11 config)."""

from common import SEED, Member, mode_from_argv, run_member
from pytabkit import TabM_D_Classifier
from realmlp import VAL_METRIC, nn_pipeline

PARAMS = {"arch_type": "tabm-mini", "tabm_k": 32, "num_emb_type": "pwl", "d_block": 512, "n_blocks": 3, "dropout": 0.1, "lr": 2e-3, "n_epochs": 40, "patience": 6, "batch_size": 1024}


def member() -> Member:
    """Build this idea's member: patience on the inner holdout; pytabkit 1.8.0 has no TabM refit, so it trains on 80% of the fold."""
    model = TabM_D_Classifier(device="cuda:0", random_state=SEED, verbosity=0, val_metric_name=VAL_METRIC, **PARAMS)
    return Member("tabm", model, pipeline=nn_pipeline(), config={"model": "tabm", "features": "raw + TE + int categories", **PARAMS}, val_holdout=True)


if __name__ == "__main__":
    run_member(member(), mode_from_argv())
