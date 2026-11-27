"""Plumbing tests for the queue runner: kill rule, hours guard, resume, prune selection."""

import json
from datetime import datetime

import numpy as np
from prune import weakest_failing
from run_queue import fits_window, killed, next_start, parse_ideas, run_steps


def test_kill_when_every_fold_is_down_beyond_the_threshold() -> None:
    ref = np.array([0.950, 0.951, 0.949])
    assert killed(ref - 0.002, ref)


def test_no_kill_when_one_fold_is_up() -> None:
    ref = np.array([0.950, 0.951, 0.949])
    assert not killed(ref + np.array([-0.004, -0.004, 0.0001]), ref)


def test_no_kill_when_the_gap_is_inside_the_threshold_of_one_thousandth_at_auc_095() -> None:
    ref = np.array([0.950, 0.951, 0.949])
    assert not killed(ref - 0.0009, ref)


def test_fits_window_refuses_a_job_ending_after_20() -> None:
    assert not fits_window(datetime(2026, 11, 27, 19, 50), 15 * 60)


def test_fits_window_refuses_before_8() -> None:
    assert not fits_window(datetime(2026, 11, 27, 7, 30), 60)


def test_fits_window_accepts_a_job_inside_the_window() -> None:
    assert fits_window(datetime(2026, 11, 27, 10, 0), 30 * 60)


def test_next_start_is_8_today_before_8_and_8_tomorrow_after_20() -> None:
    assert next_start(datetime(2026, 11, 27, 7, 30)) == datetime(2026, 11, 27, 8, 0)
    assert next_start(datetime(2026, 11, 27, 20, 30)) == datetime(2026, 11, 28, 8, 0)
    assert next_start(datetime(2026, 11, 30, 19, 55)) == datetime(2026, 12, 1, 8, 0)


def test_parse_ideas_skips_comments_and_blank_lines() -> None:
    text = "# batch\ncatboost_cat blend\n\nte_v2 single te_xgb\nprune\n"
    assert parse_ideas(text) == [("catboost_cat", "blend", None), ("te_v2", "single", "te_xgb"), ("prune", "prune", None)]


def test_resume_runs_only_the_steps_not_in_the_state_file(tmp_path) -> None:
    state_file = tmp_path / "queue_state.json"
    state_file.write_text(json.dumps({"cat:screen": "done", "cat:kill": "GO", "cat:full": "done"}))
    calls = []
    run_steps(("cat", "blend", None), state_file, lambda step, entry: calls.append(step) or "PASS", wait=lambda step, entry: None)
    assert calls == ["gate", "library", "stack"]
    assert json.loads(state_file.read_text())["cat:stack"] == "PASS"


def test_a_kill_stops_the_idea_before_the_full_run(tmp_path) -> None:
    calls = []
    run_steps(("v2", "single", "ref"), tmp_path / "s.json", lambda step, entry: calls.append(step) or ("KILL" if step == "kill" else "done"), wait=lambda step, entry: None)
    assert calls == ["screen", "kill"]


def test_a_failed_gate_skips_library_and_stack(tmp_path) -> None:
    calls = []
    run_steps(("v2", "blend", None), tmp_path / "s.json", lambda step, entry: calls.append(step) or ("FAIL" if step == "gate" else "done"), wait=lambda step, entry: None)
    assert calls == ["screen", "kill", "full", "gate"]


def test_weakest_failing_drops_the_lowest_failing_member_and_none_when_all_pass() -> None:
    contributions = {"a": np.full(5, 3e-4), "b": np.full(5, 1e-5), "c": np.full(5, -2e-5)}
    assert weakest_failing(contributions) == "c"
    assert weakest_failing({"a": np.full(5, 3e-4), "d": np.array([2e-4, 2e-4, 2e-4, 2e-4, -1e-5])}) is None
