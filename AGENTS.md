# Project Agent Rules

## Collaboration Defaults

- The user will fully help when agent input, credentials, hardware access, or decisions are needed.
- Work toward concrete research results, not just tidy code.
- Challenge the user's assumptions when evidence points elsewhere, and offer stronger alternatives with pros and cons.
- Before high-impact execution, ask the questions needed to align on goals, constraints, and success criteria.
- Do not silently stop when useful progress is still possible. Continue analyzing, planning, coding, or reporting until a real user decision or external blocker is reached.

## Project Structure

- Treat `expvision_dllm/` and `scripts/` as frozen legacy reference code.
- Prefer new experiments in `expvision_dllm_clean/` and `clean_scripts/`.
- Add new experiment families as new modules or runners instead of reopening large legacy files.
- Keep changes tightly scoped to the current experiment, analysis, or documentation goal.

## Git And Worktrees

- Use one branch per experiment or documentation direction.
- Prefer project-local worktrees under `.worktrees/`; this directory must stay ignored.
- Keep `main` stable as the synchronization point with GitHub.
- Use focused commits: one logical code, analysis, or documentation change per commit.
- After each meaningful change, summarize:
  - files changed
  - why the change was made
  - verification command and result
  - experiment command, if applicable
  - output directory, manifest, or scoreboard entry

## Experiment Results

- Do not commit raw `outputs_clean/`, `logs/`, model weights, or trace-heavy artifacts to normal git.
- Keep local raw outputs available for analysis; do not delete or overwrite historical runs unless explicitly requested.
- Commit compact, valuable records: specs, run manifests, scoreboards, analysis scripts, CSV/JSON summaries, and interpretation notes.
- Use Git LFS or an external artifact store only for selected high-value `results.jsonl` files that need to be preserved verbatim.
- Every reported result should identify its baseline, environment, GPU set, model, command, output directory, total pass rate, bucket metrics, and wins/losses.

## Hardware Baseline

- A6000 baseline experiments should use GPUs `0,1` unless the user says otherwise.
- Set `CUDA_VISIBLE_DEVICES=0,1` and `TOKENIZERS_PARALLELISM=false` for those runs.
- When comparing runs, prefer same-hardware comparisons before cross-server claims.

## Research Claims

- Do not claim SOTA from memory.
- For literature comparisons, cite the paper, dataset, metric, model, and evaluation setting.
- Separate direct apples-to-apples comparisons from suggestive or non-comparable comparisons.
