# Phase 2: S6E11 loop infrastructure, GBDT/GLM ladder, first stacked submission

Phase 1 gave kvbiii-ml saved OOFs, honest early stopping, array-based hill climbing and stacking, and a cross-fitted m-estimate encoder. This phase builds the competition folder contract on S6E11, runs the self-built GBDT/GLM ladder through the screen → full CV → OOF library loop, submits the first stack (with approval), and stops for the joint review. Branch: `s6e11/phase-2-loop`; the phase ends with a PR into `main`. Paste everything below the line into a fresh Claude Code session opened in `Kaggle_Competitions/`, or tell it: "Follow `prompts/phase-2-s6e11-loop.md`".

---

Before anything else, invoke these skills and follow them for the whole session:
1. `/ponytail:ponytail`
2. `/andrej-karpathy-skills:karpathy-guidelines`
3. `/agent-skills:interview-me` for the questions below and your own, one at a time, before the plan
4. `/kvbiii-ml-reference` (updated in Phase 1)
5. `/agent-skills:test-driven-development` for the encoder and fold plumbing only; experiments are measured, not unit-tested

Then enter plan mode and present a plan before writing code.

## Read first
- `prompts/README.md`: the Git and PR workflow
- `prompts/HANDOFF.md`, section "Phase 1": the API contracts you call
- `playbook/binary-playground.md`: the whole file. Levers 1–3 and 6 are this phase's ladder; "Validation hygiene" and "Leaderboard noise" are rules.
- `S6E11_Predicting_Credit_Card_Fraud_Classification/`: `EDA.ipynb`, `Modeling_Baseline.ipynb` (CV ~0.9468, the user's own baseline) and `Modeling_Feature_Engineering.ipynb` (9 of 35 cells run; reuse what helps)
- Memory `kaggle-venv-notebook-execution`: headless nbconvert gotchas in this venv

## Why
The user's goal (interview, 2026-11-26): "consistent top 10%" on binary Playground, self-built, judged on the **private** LB. For S6E11 the bar is self-built CV/LB ≥ 0.951, up from LB 0.94677 (rank 2343/3274). The best self-built public notebook peaks at about 0.9515; everything above 0.953 uses public OOF banks or LB probing. The user wants an "automated loop", with ideas analysed "on subsample or only selected columns to not wait plenty of hours", and then to "stop after that to see some results, both me and Claude analyze them".

## Goal
1. **Folder contract** in the S6E11 folder: `folds.npy` (fixed `StratifiedKFold(5, shuffle, SEED)`), `agent/facts.md` + `facts.json` written by a new last section of `EDA.ipynb` (schema, cardinalities, decimals/digit structure, target rates per key, drift), `experiments/` (one script per idea, screen and full mode), `oof/<member>.npz` (oof, test, per-fold AUC, config), `submissions/<id>.csv`, `experiments.md` (log), with `oof/` and `submissions/` gitignored.
2. **Trusted screening**: a measured subsample size and fold count whose AUC ranking of 3–4 known variants matches full 5-fold CV. Written into `experiments.md` with the runtime of each mode.
3. **Ladder members in the OOF library**: baseline, then exact-value + digit-key m-estimate TE, 4-leaf LGBM, XGB depth 2, GLM, GLM-boost. Each one is gated (≥ +5e-5 and ≥ 4/5 folds over the previous best) and logged.
4. **First stack**: hill climb vs logistic stack, nested CV on the same folds, the better one written to `submissions/`. Submitted **only after the user approves**, with the LB logged.
5. **Review checkpoint**: stop and report the ladder table, the CV vs LB position, and a ranked list of Phase 3 ideas.

## Rules
- **Self-built only** (user, 2026-11-26): no public OOF/submission files, nothing fitted to the public LB. Public notebooks are read for techniques only.
- **Ask before every Kaggle submit** (user, 2026-11-26), and log every LB score in `experiments.md`. Check the remaining daily submissions first (`kaggle competitions submissions playground-series-s6e11`).
- **Machine hours 08:00–20:00 only** (user, 2026-11-26). No job may be planned to run past 20:00. Estimate each full run before starting it (rows × folds × measured per-fold time).
- **Notebook structure EDA / Baseline / Advanced / Ensemble** (user, 2026-11-26). This phase uses scripts in `experiments/`; `Modeling_Advanced.ipynb` and `Modeling_Ensemble.ipynb` come in Phase 3 or 4. Notebook edits follow `~/.claude/rules/jupyter_notebook.md`.
- Never early-stop on the scored fold. Shuffled-label canary on any new encoder: AUC ≈ 0.5.
- The deadline is 2026-11-30 23:59 UTC (00:59 local on Dec 1). Phase 3 still needs time after this checkpoint.

## Questions before starting
1. Should the original dataset (`credit_card_fraud_2026.csv`, a Kaggle dataset) be pulled for per-value lookup-rate features? The research says rows or a teacher model hurt (−0.0037), and lookups are untested. If unanswered: skip it in this phase and list it as a Phase 3 idea.
2. Is the user's baseline notebook the source of the `baseline` OOF, or a script re-run of its config? If unanswered: re-run its config as a script on `folds.npy`, so it shares the same folds.

These are only the questions known when this file was written. After reading the Read first list and the code this phase touches, and before presenting the plan, ask your own: anything this file assumes that the code or docs contradict, any choice with more than one reasonable answer, any risk to data, users or other systems. Ask one at a time with your recommended answer first, and skip what you can settle from the code or docs yourself. Record each answer as a dated decision in your HANDOFF section.

## Steps
1. Folder contract, `folds.npy`, `.gitignore` entries, and the EDA facts section (run the notebook headless). → verify: `agent/facts.md` exists and lists every column with its cardinality and decimal structure; `git status` shows no data, OOF or CSV files staged.
2. Baseline member on `folds.npy`, saved to `oof/`. → verify: its full CV is within 0.0005 of the notebook's ~0.9468, and the `.npz` loads with OOF length = train rows.
3. Screening calibration. → verify: the table in `experiments.md` shows screen vs full ranks for 3–4 variants, and the chosen screen mode is ≥ 5× faster than full.
4. Ladder members, one at a time through screen → gate → full run. → verify: one `experiments.md` row per idea with screen score, per-fold full CV, gate verdict and runtime; an `.npz` exists for every accepted member.
5. Stack. → verify: nested CV of the stack beats the best single member, and the hill climb vs logistic numbers are both logged.
6. Ask the user to approve the submission, submit, log the LB. → verify: the LB score in `experiments.md` next to its CV.
7. Update `README.md` (one line on the new folder contract) and `prompts/HANDOFF.md`, and delete this file and its row in `prompts/README.md`. Commit the S6E11 scripts, `experiments.md`, `agent/` and the EDA notebook (no data, OOF or CSV), push `s6e11/phase-2-loop` and open the PR into `main`. → verify: PR link; this file is gone from the branch; `git diff --cached --name-only` before the commit listed no data files.

## Output
- Chat: the ladder table (member, CV, Δ, folds up, runtime), the stack CV, LB and rank, the time left until the deadline, and the ranked Phase 3 ideas. This is the review checkpoint: stop here.
- Append "Phase 2" to `prompts/HANDOFF.md`: what was built, the folder contract as built, decisions, open issues, Phase 3 ideas.
- PR: from `s6e11/phase-2-loop` into `main` (no CI in this repo), link given to the user. Don't merge; the user reviews and merges.
