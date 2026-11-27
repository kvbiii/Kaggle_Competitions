"""Screen calibration: rerun known variants at candidate screen sizes and compare ranks with full CV."""

import baseline_xgb
import common
import lgbm4_te
import lgbm31_raw
import numpy as np
import te_xgb
from common import load_member, member_path, run_member
from scipy.stats import kendalltau

CONFIGS = [(0.2, 3), (0.2, 5), (0.35, 3), (0.35, 5)]
VARIANTS = [baseline_xgb, te_xgb, lgbm4_te, lgbm31_raw]


def screen_name(name: str, frac: float, n_folds: int) -> str:
    """Return the member name used for one calibration screen."""
    return f"{name}@{frac}x{n_folds}"


def main() -> None:
    """Screen every variant at every config, then print one calibration row per config."""
    full = {v.member().name: load_member(member_path(v.member().name, "full")) for v in VARIANTS}
    full_auc = np.array([m["fold_auc"].mean() for m in full.values()])
    full_minutes = sum(m["config"]["runtime_s"] for m in full.values()) / 60
    print(f"full: {' '.join(f'{n}={a:.5f}' for n, a in zip(full, full_auc, strict=True))} | {full_minutes:.1f} min total")
    for frac, n_folds in CONFIGS:
        common.SCREEN = {"frac": frac, "n_folds": n_folds}
        scores, seconds = [], 0.0
        for variant in VARIANTS:
            member = variant.member()
            member.name = screen_name(member.name, frac, n_folds)
            result = run_member(member, "screen")
            scores.append(result["fold_auc"].mean())
            seconds += result["config"]["runtime_s"]
        tau = kendalltau(scores, full_auc).statistic
        same = bool(np.array_equal(np.argsort(scores), np.argsort(full_auc)))
        print(f"| {frac:.0%} rows × {n_folds} folds | {' '.join(f'{s:.5f}' for s in scores)} | τ {tau:.2f} | same order {same} | {seconds / 60:.1f} min | {full_minutes * 60 / seconds:.1f}× faster |")


if __name__ == "__main__":
    main()
