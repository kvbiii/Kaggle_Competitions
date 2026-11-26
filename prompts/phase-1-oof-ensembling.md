# Phase 1: kvbiii-ml saves OOFs and ensembles from arrays

First phase of the track. It adds what the S6E11 loop needs from kvbiii-ml: saved OOF predictions, honest early stopping, array-based hill climbing and stacking, and a cross-fitted m-estimate target encoder. It also installs the library editable into the Kaggle venv. Branches: `feat/phase-1-oof-ensembling` in kvbiii-ml (PR into `main`, CI green) and `track/phase-1-kaggle-loop` in Kaggle_Competitions (commits `prompts/`, `playbook/`, HANDOFF). Paste everything below the line into a fresh Claude Code session opened in `Kaggle_Competitions/`, or tell it: "Follow `prompts/phase-1-oof-ensembling.md`".

---

Before anything else, invoke these skills and follow them for the whole session:
1. `/ponytail:ponytail`
2. `/andrej-karpathy-skills:karpathy-guidelines`
3. `/agent-skills:interview-me` for the questions below and your own, one at a time, before the plan
4. `/kvbiii-ml-reference` (stale: written at rev 770a6af, the repo is at 0125d88; diff before trusting it)
5. `/agent-skills:api-and-interface-design`, `/agent-skills:test-driven-development`

Then enter plan mode and present a plan before writing code.

## Read first
- `prompts/README.md`: the two-repo Git and PR workflow
- `playbook/binary-playground.md`: sections "Validation hygiene", levers 1 and 6. They define what the new code must do.
- `~/Desktop/KubiK/Programming/Python/Paczki/kvbiii-ml/kvbiii_ml/modeling/training/cross_validation.py` (`fit`, ~L286) and `base_trainer.py` (eval_set injection): where OOF predictions get dropped and where early stopping gets its eval set
- `kvbiii_ml/modeling/optimization/ensemble_weights_tuner.py`: today's weight tuner retrains every model. The new code works on saved arrays instead.
- `kvbiii_ml/data_processing/feature_engineering/` (`TargetEncodingFeatureGenerator`, `DigitsEncodingFeatureGenerator`) and `data_processing/preprocessing/` (`TargetEncoderWithOriginal`): reuse whatever already covers the encoder.

## Why
The user wants consistent top 10% on the private LB of binary Playground competitions with self-built solutions (interview, 2026-11-26). Planning found that nothing saves OOF predictions, so hill climbing, stacking and comparing experiments are impossible. The research (playbook) shows that exact-value and digit-key m-estimate TE, shallow GBDTs, GLMs and a nested-CV stack get self-built S6E11 solutions from about 0.947 to about 0.9515. S6E11 closes 2026-11-30 23:59 UTC, so this phase is the critical path.

## Goal
1. **OOF predictions survive `CrossValidationTrainer.fit`**: available after fitting, aligned to the input row order, alongside the existing averaged test prediction. Phase 2 saves them to `oof/<member>.npz`.
2. **Early stopping never uses the scored fold**: if it does today, add an opt-in inner holdout carved from the training rows of each fold. Measure the OOF difference on one dataset.
3. **Hill climb from arrays**: a Caruana blend over an OOF matrix + y + fold ids + test matrix. Rank or logit space, with replacement, starting from the best member, weights taken at the best iteration, patience stop, fold-honest score (fit on 4 folds, score the 5th).
4. **Logistic stack from arrays**: logistic regression on logit-ranks with nested CV over several outer seeds and a C grid. Same inputs and outputs as the hill climb, so the two can be compared on the same folds.
5. **Cross-fitted m-estimate target encoder**: exact values and digit keys (cents, floor, floor/10, floor×10, leading 1–2 digits, length) of chosen columns, `(Σy + m·prior)/(n + m)` with m=20 by default, cross-fitted inside the training fold. Usable as a step in `preprocessing_pipeline`. Skip it if an existing class can do this with a parameter.
6. **Kaggle venv ready**: kvbiii-ml editable in `Kaggle_Competitions_venv`, plus `pytabkit` (pulls torch). `requirements.txt` updated.

## Rules
- **Editable install into the Kaggle venv** (user, 2026-11-26): `uv pip install --python Kaggle_Competitions_venv/bin/python -e ~/Desktop/KubiK/Programming/Python/Paczki/kvbiii-ml`. The PyPI release can wait until after S6E11.
- **Self-built only** (user, 2026-11-26): nothing here reads other people's OOFs or submissions.
- Existing public signatures keep working. Every new behaviour is additive or opt-in, and the ~600 existing tests stay green.
- Installing pytabkit downgrades `nvidia-nccl-cu13` 2.32→2.30 (dry run, 2026-11-26). XGBoost and LightGBM on CUDA must still fit afterwards. The LightGBM CUDA build recipe is in `requirements.txt` and in memory `cuda-toolchain-env`.
- Don't estimate how long implementation takes. Do estimate run times (user rule).

## Questions before starting
1. Should the hill climb allow negative weights, as `EnsembleWeightTunerCV` already does? If unanswered: non-negative only, matching the research recipe, and say so.
2. If pytabkit or torch breaks the CUDA GBDTs and the fix is not quick: stop and ask whether RealMLP should move to its own venv.
3. Does the hill climb need to be fast on 700k rows × ~30 members? If unanswered: yes; measure it and report the runtime.

These are only the questions known when this file was written. After reading the Read first list and the code this phase touches, and before presenting the plan, ask your own: anything this file assumes that the code or docs contradict, any choice with more than one reasonable answer, any risk to data, users or other systems. Ask one at a time with your recommended answer first, and skip what you can settle from the code or docs yourself. Record each answer as a dated decision in your HANDOFF section.

## Steps
1. Read how early stopping and validation predictions flow through `CrossValidationTrainer` and `BaseTrainer`, and report what you find. → verify: a one-paragraph finding in chat, citing file:line, saying whether early stopping uses the scored fold.
2. OOF exposure (and the inner holdout if step 1 shows it is needed), test first. → verify: a new test asserts that the per-fold AUC computed from the returned OOF equals the trainer's own `valid_scores` on a toy set; the full suite is green.
3. Hill climb and logistic stack from arrays, test first. → verify: a known-answer test (one perfect member plus noise members → that member gets all the weight), a nested-score test, and a runtime printout on synthetic data of 700k × 30.
4. Cross-fitted m-estimate encoder (or a parameter on an existing class). → verify: a shuffled-label canary gives OOF AUC within 0.5 ± 0.01, and a digit-key unit test covers cents, floor, lead1 and lead2 on known floats.
5. Update `~/.claude/skills/kvbiii-ml-reference/SKILL.md` for the new APIs and the new rev. → verify: the skill names the new classes and the rev it was checked at.
6. Editable install + pytabkit into the Kaggle venv. → verify: `Kaggle_Competitions_venv/bin/python -c "import kvbiii_ml, pytabkit, torch; print(torch.cuda.is_available())"` prints `True`, and an XGB `device="cuda"` and LGBM `device="cuda"` smoke fit each pass.
7. Write HANDOFF, delete this file and its row in `prompts/README.md`. Push both branches and open both PRs (workflow in `prompts/README.md`). The Kaggle_Competitions PR adds `prompts/`, `playbook/` and `requirements.txt`. → verify: the kvbiii-ml PR link with CI green, the Kaggle_Competitions PR link, and this file gone from its branch.

## Output
- Chat: the new API names, the measured OOF effect of the early-stopping fix, the hill-climb runtime, the install result, any deviations and why.
- Create `prompts/HANDOFF.md` with section "Phase 1": what was built, the API contracts Phase 2 calls, decisions, open issues.
- PRs: kvbiii-ml `feat/phase-1-oof-ensembling` into `main` with CI green, and Kaggle_Competitions `track/phase-1-kaggle-loop` into `main`. Give both links to the user. Don't merge.
