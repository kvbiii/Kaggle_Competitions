"""
Gate a candidate member.

Usage:
    python gate.py single <candidate> <reference> [screen|full]
    python gate.py blend <candidate> [screen|full]     (library = experiments/library.txt)
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from common import TARGET, load_data, load_member, log_line, member_path
from kvbiii_ml.modeling.optimization.oof_ensembling import hill_climb
from scipy.stats import spearmanr

LIBRARY_FILE = Path(__file__).with_name("library.txt")
MIN_GAIN = 5e-5


def library() -> list[str]:
    """Return the accepted member names, in acceptance order."""
    return LIBRARY_FILE.read_text().split() if LIBRARY_FILE.exists() else []


def verdict(delta: np.ndarray) -> tuple[bool, str]:
    """Apply the gate (mean gain >= MIN_GAIN and at most one fold down) to per-fold deltas."""
    up = int((delta > 0).sum())
    passed = delta.mean() >= MIN_GAIN and up >= len(delta) - 1
    return passed, f"Δ {delta.mean():+.5f} | Δ per fold {' '.join(f'{d:+.5f}' for d in delta)} | {up}/{len(delta)} up"


def member_matrix(names: list[str], mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack members into (oof, test, folds, y) after checking they share rows and folds."""
    members = [load_member(member_path(n, mode)) for n in names]
    first = members[0]
    for name, m in zip(names, members, strict=True):
        if not (np.array_equal(m["rows"], first["rows"]) and np.array_equal(m["fold"], first["fold"])):
            raise ValueError(f"{name} does not share rows/folds with {names[0]}.")
    oof = np.column_stack([m["oof"] for m in members])
    test = np.column_stack([m["test"] for m in members]) if first["test"].size else np.zeros((1, len(members)))
    return oof, test, first["fold"], first["rows"]


def single(candidate: str, reference: str, mode: str) -> bool:
    """Gate on single-model per-fold AUC of candidate vs reference."""
    cand, ref = load_member(member_path(candidate, mode)), load_member(member_path(reference, mode))
    passed, text = verdict(cand["fold_auc"] - ref["fold_auc"])
    log_line(f"| {datetime.now():%m-%d %H:%M} | GATE single {candidate} vs {reference} | {mode} | {text} | {'PASS' if passed else 'FAIL'} |")
    return passed


def blend(candidate: str, mode: str, y_all: np.ndarray) -> bool:
    """Gate on the nested hill-climb AUC of library + candidate vs library alone."""
    names = library()
    oof, test, folds, rows = member_matrix([*names, candidate], mode)
    y = y_all[rows]
    base = hill_climb(oof[:, :-1], y, folds, test[:, :-1])
    with_candidate = hill_climb(oof, y, folds, test)
    max_corr = max(spearmanr(oof[:, -1], oof[:, i]).statistic for i in range(len(names)))
    passed, text = verdict(with_candidate.fold_scores - base.fold_scores)
    weight = with_candidate.weights[-1]
    log_line(f"| {datetime.now():%m-%d %H:%M} | GATE blend {candidate} vs [{' '.join(names)}] | {mode} | {base.score:.5f} → {with_candidate.score:.5f}, {text}, weight {weight:.3f}, max ρ {max_corr:.5f} | {'PASS' if passed else 'FAIL'} |")
    return passed


if __name__ == "__main__":
    kind, args = sys.argv[1], sys.argv[2:]
    if kind == "single":
        single(args[0], args[1], args[2] if len(args) > 2 else "full")
    elif kind == "blend":
        blend(args[0], args[1] if len(args) > 1 else "full", load_data()[0][TARGET].to_numpy())
    else:
        raise SystemExit(__doc__)
