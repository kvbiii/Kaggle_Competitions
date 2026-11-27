"""
Prune the library: drop, one at a time, the member whose leave-one-out contribution fails the gate.

Usage:
    python prune.py [screen|full]
"""

import sys
from collections.abc import Callable
from datetime import datetime

import numpy as np
from common import TARGET, load_data, log_line
from gate import LIBRARY_FILE, library, member_matrix, verdict
from kvbiii_ml.modeling.optimization.oof_ensembling import hill_climb


def weakest_failing(contributions: dict[str, np.ndarray]) -> str | None:
    """Return the member with the lowest mean per-fold contribution among those failing the gate, or None."""
    failing = {name: delta.mean() for name, delta in contributions.items() if not verdict(delta)[0]}
    return min(failing, key=failing.__getitem__) if failing else None


def contributions(names: list[str], mode: str, y_all: np.ndarray) -> dict[str, np.ndarray]:
    """Return each member's per-fold nested hill-climb gain: library minus library without it."""
    oof, test, folds, rows = member_matrix(names, mode)
    y = y_all[rows]
    full = hill_climb(oof, y, folds, test).fold_scores
    keep = {name: [j for j in range(len(names)) if j != i] for i, name in enumerate(names)}
    return {name: full - hill_climb(oof[:, cols], y, folds, test[:, cols]).fold_scores for name, cols in keep.items()}


def prune(mode: str = "full", wait: Callable[[int], None] | None = None) -> list[str]:
    """Drop failing members one per round (re-scoring the rest each round); rewrite library.txt; return the dropped."""
    y_all = load_data()[0][TARGET].to_numpy()
    dropped = []
    while len(names := library()) > 1:
        if wait:
            wait(len(names))
        contrib = contributions(names, mode, y_all)
        drop = weakest_failing(contrib)
        for name, delta in contrib.items():
            passed, text = verdict(delta)
            log_line(f"| {datetime.now():%m-%d %H:%M} | PRUNE loo {name} [{len(names)} members] | {mode} | {text} | {'keep' if passed else 'fail'} | {'DROP' if name == drop else '-'} |")
        if drop is None:
            break
        LIBRARY_FILE.write_text("".join(f"{n}\n" for n in names if n != drop))
        dropped.append(drop)
    return dropped


if __name__ == "__main__":
    print("dropped:", prune(sys.argv[1] if len(sys.argv) > 1 else "full"))
