"""
Run an idea list unattended: screen, kill check, full run, gate, library update, re-stack.

Usage:
    python run_queue.py <ideas.txt>

Each non-comment line of the ideas file is `<idea> blend` (a new family, gated on nested hill-climb gain),
`<idea> single <ref>` (a variant replacing `ref`, gated on single-model CV) or `prune`. Finished steps are
recorded in `<ideas>.state.json` and skipped on a restart. Every step waits until its estimate fits
08:00-20:00. Shortcut: state keys are `<idea>:<step>`, so an idea (or `prune`) runs once per ideas file.
"""

import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import stack
from common import TARGET, load_data, load_member, log_line, member_path
from gate import LIBRARY_FILE, blend, library, single
from prune import prune

HERE = Path(__file__).resolve().parent
START_HOUR, END_HOUR = 8, 20
KILL_FRACTION = 0.02
FULL_OVER_SCREEN = 18
SECONDS = {"screen": 15 * 60, "kill": 60, "gate": 10 * 60, "library": 0, "stack": 8 * 60, "prune": 0}
PRUNE_SECONDS_PER_MEMBER = 3 * 60
STEPS = {"blend": ("screen", "kill", "full", "gate", "library", "stack"), "single": ("screen", "kill", "full", "gate", "library", "stack"), "prune": ("prune", "stack")}
STOP = {"KILL", "FAIL", "CRASH", "UNCHANGED"}

Entry = tuple[str, str, str | None]


def killed(cand_auc: np.ndarray, ref_auc: np.ndarray) -> bool:
    """Kill iff the screen trails the reference by > KILL_FRACTION of its error (1 - AUC) and every fold is down."""
    gap = cand_auc.mean() - ref_auc.mean()
    return bool(gap < -KILL_FRACTION * (1 - ref_auc.mean()) and (cand_auc < ref_auc).all())


def fits_window(now: datetime, seconds: float) -> bool:
    """Return whether a job of `seconds` started now begins after START_HOUR and ends by END_HOUR today."""
    return now.hour >= START_HOUR and now + timedelta(seconds=seconds) <= now.replace(hour=END_HOUR, minute=0, second=0, microsecond=0)


def next_start(now: datetime) -> datetime:
    """Return the next START_HOUR: today if it is still before it, else tomorrow."""
    today = now.replace(hour=START_HOUR, minute=0, second=0, microsecond=0)
    return today if now < today else today + timedelta(days=1)


def wait_for_window(seconds: float, label: str) -> None:
    """Sleep until a job of `seconds` fits the machine hours, logging each wait."""
    if seconds > (END_HOUR - START_HOUR) * 3600:
        raise ValueError(f"{label}: estimate {seconds / 3600:.1f} h exceeds the daily window.")
    while not fits_window(now := datetime.now(), seconds):
        start = next_start(now)
        log_line(f"| {now:%m-%d %H:%M} | QUEUE WAIT {label} | - | est {seconds / 60:.1f} min | until {start:%m-%d %H:%M} | - | - |")
        time.sleep((start - now).total_seconds())


def parse_ideas(text: str) -> list[Entry]:
    """Parse the ideas file into (idea, kind, ref) entries."""
    entries = []
    for line in text.splitlines():
        parts = line.split("#")[0].split()
        if not parts:
            continue
        if parts == ["prune"]:
            entries.append(("prune", "prune", None))
        elif len(parts) == 2 and parts[1] == "blend" or len(parts) == 3 and parts[1] == "single":
            entries.append((parts[0], parts[1], parts[2] if len(parts) == 3 else None))
        else:
            raise ValueError(f"Bad ideas line: {line!r} (want '<idea> blend', '<idea> single <ref>' or 'prune').")
    return entries


def run_steps(entry: Entry, state_file: Path, step_fn: Callable[[str, Entry], str], wait: Callable[[str, Entry], None]) -> None:
    """Run the entry's steps not yet recorded in `state_file`, saving each result; stop at a STOP result."""
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    for step in STEPS[entry[1]]:
        key = f"{entry[0]}:{step}"
        if key not in state:
            wait(step, entry)
            state[key] = step_fn(step, entry)
            tmp = state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=1))
            tmp.replace(state_file)
        if state[key] in STOP:
            return


def estimate_seconds(step: str, entry: Entry) -> float:
    """Estimate a step's runtime: full = FULL_OVER_SCREEN x the measured screen runtime, others fixed."""
    if step == "full":
        return FULL_OVER_SCREEN * load_member(member_path(entry[0], "screen"))["config"]["runtime_s"]
    return SECONDS[step]


def train(idea: str, mode: str) -> str:
    """Run `<idea>.py <mode>` in a subprocess; return 'done' or 'CRASH' (logged)."""
    code = subprocess.run([sys.executable, "-W", "ignore", f"{idea}.py", mode], cwd=HERE, check=False).returncode
    if code:
        log_line(f"| {datetime.now():%m-%d %H:%M} | QUEUE {idea} | {mode} | exit code {code} | - | - | CRASH |")
        return "CRASH"
    return "done"


def kill_check(idea: str, kind: str, ref: str | None) -> str:
    """Apply the kill rule to a variant's screen vs its reference's screen (run first if missing); families pass."""
    if kind != "single":
        return "GO"
    if not member_path(ref, "screen").exists() and train(ref, "screen") == "CRASH":
        return "CRASH"
    cand, base = (load_member(member_path(n, "screen"))["fold_auc"] for n in (idea, ref))
    verdict = "KILL" if killed(cand, base) else "GO"
    log_line(f"| {datetime.now():%m-%d %H:%M} | QUEUE kill-check {idea} vs {ref} | screen | {cand.mean():.5f} vs {base.mean():.5f}, Δ {cand.mean() - base.mean():+.5f}, threshold {-KILL_FRACTION * (1 - base.mean()):+.5f} | {int((cand < base).sum())}/{len(cand)} folds down | {verdict} |")
    return verdict


def add_to_library(idea: str, ref: str | None) -> str:
    """Put a passed member in library.txt: in place of `ref` when it replaces one, else appended."""
    names = library()
    names = [idea if n == ref else n for n in names] if ref in names else [*names, idea]
    LIBRARY_FILE.write_text("".join(f"{n}\n" for n in dict.fromkeys(names)))
    return "done"


def do_step(step: str, entry: Entry) -> str:
    """Run one queue step for real."""
    idea, kind, ref = entry
    if step == "screen":
        return train(idea, "screen")
    if step == "kill":
        return kill_check(idea, kind, ref)
    if step == "full":
        log_line(f"| {datetime.now():%m-%d %H:%M} | QUEUE {idea} | full | est {estimate_seconds(step, entry) / 60:.1f} min | started | - | - |")
        return train(idea, "full")
    if step == "gate":
        passed = single(idea, ref, "full") if kind == "single" else blend(idea, "full", load_data()[0][TARGET].to_numpy())
        return "PASS" if passed else "FAIL"
    if step == "library":
        return add_to_library(idea, ref)
    if step == "prune":
        dropped = prune("full", wait=lambda m: wait_for_window(PRUNE_SECONDS_PER_MEMBER * m, f"prune round [{m} members]"))
        return f"DROPPED {' '.join(dropped)}" if dropped else "UNCHANGED"
    stack.main()
    return "done"


def main(ideas_file: Path) -> None:
    """Run every entry of the ideas file in order."""
    state_file = ideas_file.with_name(f"{ideas_file.stem}.state.json")
    for entry in parse_ideas(ideas_file.read_text()):
        run_steps(entry, state_file, do_step, wait=lambda step, e: wait_for_window(estimate_seconds(step, e), f"{step} {e[0]}"))
    log_line(f"| {datetime.now():%m-%d %H:%M} | QUEUE done {ideas_file.name} | - | - | - | - | - |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).resolve())
