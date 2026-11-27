"""Blend the accepted library: hill climb vs logistic stack (nested, same folds); write the better one."""

from datetime import datetime

import numpy as np
import pandas as pd
from common import (
    DATA_DIR,
    SUBMISSION_DIR,
    TARGET,
    load_data,
    load_member,
    log_line,
    member_path,
)
from gate import library, member_matrix
from kvbiii_ml.modeling.optimization.oof_ensembling import hill_climb, logistic_stack
from scipy.stats import rankdata


def main() -> None:
    """Fit both blenders on the full-mode library, log them, and save the better test blend."""
    names = library()
    oof, test, folds, rows = member_matrix(names, "full")
    y = load_data()[0][TARGET].to_numpy()[rows]
    best_single = max(load_member(member_path(n, "full"))["fold_auc"].mean() for n in names)
    results = {"hill_climb": hill_climb(oof, y, folds, test, space="rank"), "logistic_stack": logistic_stack(oof, y, folds, test)}
    for method, r in results.items():
        weights = " ".join(f"{n}={w:.3f}" for n, w in zip(names, r.weights, strict=True))
        log_line(f"| {datetime.now():%m-%d %H:%M} | STACK {method} [{len(names)} members] | full | {r.score:.5f} | {' '.join(f'{s:.5f}' for s in r.fold_scores)} | vs best single {r.score - best_single:+.5f} | {weights} |")
    method, best = max(results.items(), key=lambda item: item[1].score)
    submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    submission[TARGET] = rankdata(best.test) / len(best.test)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    path = SUBMISSION_DIR / f"{datetime.now():%m%d_%H%M}_{method}_{len(names)}m.csv"
    submission.to_csv(path, index=False)
    print(f"{method} wins ({best.score:.5f}); wrote {path} ({len(submission)} rows, nan={np.isnan(submission[TARGET]).sum()})")


if __name__ == "__main__":
    main()
