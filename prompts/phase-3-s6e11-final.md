# Phase 3: S6E11 queue runner, NN/CatBoost members, final picks, Advanced/Ensemble notebooks

Phase 2 built the S6E11 folder contract and a GBDT/GLM OOF library: logistic stack of 5 members, nested CV 0.95066, public LB 0.95170. This phase moves the loop from the conversation into code (a queue runner), adds new model families through it, prepares the 2 final picks, and writes the readable notebooks on top of `oof/`. Branch: `s6e11/phase-3-final`; the phase ends with a PR into `main`. Not the last phase of the track (Phase 4 does the retro and templates). Paste everything below the line into a fresh Claude Code session opened in `Kaggle_Competitions/`, or tell it: "Follow `prompts/phase-3-s6e11-final.md`".

---

Before anything else, invoke these skills and follow them for the whole session:
1. `/ponytail:ponytail`
2. `/andrej-karpathy-skills:karpathy-guidelines`
3. `/agent-skills:interview-me` for the questions below and your own, one at a time, before the plan
4. `/kvbiii-ml-reference`
5. `/agent-skills:test-driven-development` for the queue runner's plumbing only (resume, kill rule, hours guard); experiments are measured, not unit-tested

Then enter plan mode and present a plan before writing code.

## Read first
- `prompts/README.md`: the Git and PR workflow
- `prompts/HANDOFF.md`, section "Phase 2": the folder contract, commands, measurements, open issues and the ranked Phase 3 ideas
- `S6E11_Predicting_Credit_Card_Fraud_Classification/experiments.md`: calibration table (the screen's limits), ladder, stack and LB table
- `S6E11_.../experiments/common.py`, `gate.py`, `stack.py`: reuse `run_member`, `Member`, the gates and the blenders; don't rewrite them
- `playbook/binary-playground.md`: lever 4 (RealMLP and TabM configs), lever 5 (full refit), "Validation hygiene", "Leaderboard noise"
- `~/.claude/rules/jupyter_notebook.md`: for the two notebooks
- Memories `kaggle-venv-notebook-execution` and `kaggle-cli-submissions-parsing`

## Why
User goal (interview, 2026-11-26): "consistent top 10%" on binary Playground, self-built, judged on the **private** LB. The S6E11 bar is self-built ≥ 0.951. Phase 2 reached LB 0.95170 but CV 0.95066. Asked how Phase 3 relates to the repo structure, the user chose (2026-11-27) to put the loop in code: "Queue runner first", so the December loop skill can reuse it and token use stays low.

## Goal
1. **Queue runner**: one command takes an idea list and runs each idea unattended: screen, then a full run if the screen doesn't kill it, then the gate, a library update and a re-stack. Everything is appended to `experiments.md`. It can resume after an interruption. It comes first because every other goal runs through it.
2. **New model families as gated members**: RealMLP and TabM (GPU), and CatBoost with all columns categorical. Each one is logged with screen score, per-fold full CV, gate verdict and runtime.
3. **Library upkeep**: a leave-one-out nested-CV pruning pass (GLM now gets ≈ 0 weight) and a second GLM-boost view (LGBM 4 leaves on the GLM margin).
4. **Test-side full refit** (lever 5) for the top members. These cost LB probes and need approval.
5. **Final 2 picks ready by 2026-11-30 20:00 local**: both files in `submissions/`, the pick rule and reasoning in `experiments.md`, and the user selects them on Kaggle.
6. **`Modeling_Advanced.ipynb` and `Modeling_Ensemble.ipynb`** in the S6E11 folder. They read `oof/` and the logs only, with no retraining. Advanced: members, gates, per-fold tables, correlations. Ensemble: load the library, hill climb vs stack, write the chosen submission. Both run top to bottom headless in about a minute.

## Rules
- **Self-built only** (user, 2026-11-26). No public OOF or submission files, nothing fitted to the public LB.
- **Ask before every Kaggle submit** (user, 2026-11-26). Check the remaining daily submissions first, and log every LB score next to its CV.
- **Machine hours 08:00–20:00** (user, re-confirmed 2026-11-27). The queue must not start a job whose estimate ends after 20:00; it waits for the next morning. Estimate every full run from measured per-fold time.
- **Final pick rule** (user, 2026-11-27): the agent proposes it after seeing Phase 3 results and the user confirms. Keep at least one pick purely CV-backed (playbook, "Leaderboard noise").
- Gate and validation rules from Phase 2 stand (decided 2026-11-27):
  - same `folds.npy`;
  - honest trees, never early-stopped on the scored fold;
  - an FE replacement is gated on single-model CV, a new family on nested hill-climb gain;
  - ≥ +5e-5 and ≥ 4/5 folds;
  - a shuffled-label canary on any new encoder or feature builder;
  - the screen only kills ideas trailing by > 1e-3.
- The deadline is 2026-11-30 23:59 UTC (00:59 local on Dec 1). The picks are due at 20:00 local on Nov 30, so nothing waits on the last evening.

## Questions before starting
1. Can the Kaggle CLI/API mark final submissions, or does the user pick in the web UI? First verify from the CLI help. If unclear: the user picks in the UI, and you hand over the two file names and their submission refs.
2. Do the playbook's RealMLP and TabM settings still match the installed pytabkit 1.8.0 API? `RealMLP_TD_Classifier` and `TabM_D_Classifier` exist there (checked 2026-11-27), but the parameter names haven't been checked. Verify them against the installed source. If one doesn't fit: use that class's defaults plus the playbook's epochs and batch size, and say so.
3. Which members get the full refit and probe? If unanswered: the top 2 by stack weight, one probe each.

These are only the questions known when this file was written. After reading the Read first list and the code this phase touches, and before presenting the plan, ask your own: anything this file assumes that the code or docs contradict, any choice with more than one reasonable answer, any risk to data, users or other systems. Ask one at a time with your recommended answer first, and skip what you can settle from the code or docs yourself. Record each answer as a dated decision in your HANDOFF section.

## Steps
1. Queue runner, with tests for resume, the kill rule and the hours guard. Re-run one Phase 2 idea through it. → verify: pytest green; the re-run reproduces that member's full CV to 1e-5; its `experiments.md` rows match the Phase 2 format.
2. Batch 1 through the queue: CatBoost all-categorical, GLM-boost on LGBM-4, then the pruning pass. → verify: one row per idea with verdict and runtime; `library.txt` and the re-stack CV are updated.
3. Batch 2: RealMLP, then TabM. Measure one fold, estimate the full run, then run it. → verify: rows logged; the canary ≈ 0.5 on any new input builder; a new stack CV is logged if a member passed.
4. Full refit of the chosen members and LB probes (ask first). → verify: the LB of each probe sits next to the fold-average version in the LB table.
5. Propose the final-pick rule, get the user's confirmation, write both files. → verify: the files exist, the rule is in `experiments.md`, and the user has confirmed the picks before 2026-11-30 20:00 local.
6. Both notebooks, run headless. → verify: `jupyter nbconvert --execute` exits 0 and their stack score equals `stack.py`'s.
7. Update `prompts/HANDOFF.md` (Phase 3 section; add the S6E11 retro stub to the playbook), delete this file and its row in `prompts/README.md`, commit (no data, OOF or CSV; check `git diff --cached --name-only`), push `s6e11/phase-3-final` and open the PR into `main` (workflow in `prompts/README.md`). → verify: PR link; this file is gone from the branch.

## Output
- Chat:
  - the member table (CV, Δ, folds up, runtime, verdict);
  - stack CV before and after;
  - the LB probes;
  - the two picks with their reasoning;
  - total machine time and the queue's turn count vs Phase 2.
- Append "Phase 3" to `prompts/HANDOFF.md`: what was built (queue contract, notebooks), decisions, open issues, what Phase 4 should turn into templates.
- PR: from `s6e11/phase-3-final` into `main` (no CI in this repo), link given to the user. Don't merge; the user reviews and merges.
