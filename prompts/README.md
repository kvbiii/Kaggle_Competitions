# Build phases: agent-driven Kaggle loop

Run one phase per fresh Claude Code session, opened in `Kaggle_Competitions/`. Paste everything below the line of that phase's file, or tell the session to follow it. Everything in this folder is tracked in git. Each phase's PR deletes that phase's prompt file and its row below; git history keeps them. The last phase of the track moves `HANDOFF.md` to `docs/kaggle-loop-handoff.md` and deletes this folder. Finish, verify and merge a phase before starting the next.

| # | File | Result |
|---|---|---|
| 1 | `phase-1-oof-ensembling.md` | kvbiii-ml saves OOF predictions, hill-climbs and stacks from arrays, cross-fitted m-estimate TE; editable install in the Kaggle venv |
| 2 | `phase-2-s6e11-loop.md` | S6E11 loop infrastructure, GBDT/GLM ladder, first stacked submission, review checkpoint |
| 3 | written from HANDOFF after phase 2 | S6E11 NN members, test refit, final 2 picks by 2026-11-30 20:00 local |
| 4 | written from HANDOFF after the close | private-LB retro, template notebooks, root `CLAUDE.md` and loop skill for December |

Each phase ends by appending a short section to `prompts/HANDOFF.md`: what was built, decisions made, anything left open. The next phase reads it first.

## Git and PR workflow (every phase)

Two repositories are involved:
- `Kaggle_Competitions` (this repo, remote `kvbiii/Kaggle_Competitions`): no CI. The check command is `uv run --with ruff ruff check <changed .py files>` plus running the phase's verify commands.
- `kvbiii-ml` (`~/Desktop/KubiK/Programming/Python/Paczki/kvbiii-ml`, remote `kvbiii/kvbiii-ml`): CI in `.github/workflows/python-publish.yml` (ruff format/check, `ty --error-on-warning`, pytest 3.11–3.14). Locally: `uv run ruff format --check && uv run ruff check && uv run ty check --error-on-warning && uv run pytest`.

`main` only changes through merged PRs in both repos.

**Start of the session** (in each repo the phase touches):

```sh
git switch main && git pull --ff-only
git switch -c <phase branch named in the prompt>
```

Stop if `main` has uncommitted changes to files this phase touches, or the previous phase's PR is still open. The competition folders (`S5E8_...` to `S6E11_...`) are untracked by design: leave the ones this phase does not touch alone.

**During:** small commits, one per verified step, each with a message that says what changed and why. Run the check command before each commit. No `Co-Authored-By` or "Generated with" lines (user rule in `~/.claude/CLAUDE.md`).

**End of the session:**

1. Check command green locally.
2. `git push -u origin <branch>`
3. `gh pr create --base main --title "Phase N: <result>" --body-file <tmp file>` with sections **What changed**, **How it was verified**, **Decisions**, **Open issues**, **How to try it**.
4. kvbiii-ml only: `gh pr checks --watch`. If CI fails, fix it on the branch and push again until green.
5. Give the user the PR link(s). **Don't merge.** The user reviews and merges.

**Never commit:** `kaggle.json`, `.env`, `data/`, `*.csv`, `oof/`, `submissions/`, `catboost_info/`, notebook checkpoints. Check `git diff --cached --name-only` before every commit.
