# Historical Result Archive Design

Date: 2026-05-28  
Branch: `docs/result-archive`  
Raw result source: `/home/shx/projects/dllm_infilling/outputs_clean`

## Objective

Create a durable, readable record of previous LCAL/LCAS experiments without committing every raw result file to normal git.

The archive should let a future reader answer:

- Which runs mattered?
- What command and environment produced them?
- What was the pass rate and bucket breakdown?
- Which run is the correct baseline for a comparison?
- Which raw local directory contains the full `results.jsonl`?
- Which results are obsolete, diagnostic-only, or superseded?

## Why Raw Outputs Stay Out Of Normal Git

The current local `outputs_clean/` directory is large and contains many full `results.jsonl` files. These files are valuable, but normal git is a poor storage layer for all of them because it makes clone, diff, and branch operations slow.

The archive keeps raw outputs local and commits compact records:

- run registry
- scoreboards
- bucket summaries
- wins/losses
- analysis scripts
- notes explaining why each run matters

Selected raw `results.jsonl` files can later be moved to Git LFS or external artifact storage if they are needed for exact reproduction.

## Archive Structure

Use these committed paths:

- `docs/results/run_registry.md`: human-readable table of meaningful runs.
- `docs/results/run_registry.json`: machine-readable run metadata.
- `analysis_outputs/experiment_scoreboard.md`: current headline scoreboard.
- `analysis_outputs/<topic>/`: compact CSV/JSON outputs from analysis scripts.

Each registry entry should include:

- run id
- local raw output path
- model
- hardware/environment
- script or runner
- command summary
- baseline relationship
- total pass rate
- bucket metrics
- wins/losses if available
- status: `global_best`, `env_control`, `candidate`, `diagnostic`, `superseded`, or `obsolete`
- short interpretation

## Triage Rules

Keep detailed records for:

- global best checkpoints
- same-hardware controls
- strategy-best candidates
- runs used in a paper/table/scoreboard
- diagnostics that explain why a branch was rejected

Mark as superseded or obsolete:

- early smoke runs
- parameter sweeps whose conclusion is captured by a later summary
- failed runs with no unique diagnostic value
- duplicate reruns that do not change conclusions

Do not delete local raw outputs during archive work unless the user explicitly asks.

## First Archive Pass

The first pass should ingest and classify the known important runs from the existing scoreboard:

- `old_union_gpus23`
- `union_gpus01_control`
- `eval12_nomiddle_gpus01_control`
- `midcons`
- `midaggr`

Then scan `outputs_clean/` for additional full runs and classify them into:

- baseline utilities
- LCAL-v3 alpha/ratio experiments
- short-safe S-series experiments
- official-CAL / bounded repair experiments
- model-generalization experiments

## Success Criteria

- A reader can reconstruct the current project state from committed docs without opening every raw `results.jsonl`.
- Important historical runs are preserved by reference and summary.
- Raw output paths remain valid locally.
- The archive distinguishes same-environment comparisons from cross-environment comparisons.
- The archive does not make SOTA or paper claims without sourced comparisons.

## Risks

- Some old runs may lack complete metadata; mark missing fields explicitly as `unknown`.
- Local raw paths are machine-specific; use them as local provenance, not universal artifact URLs.
- Recomputing summaries from large files may take time; prefer incremental scripts and cached compact outputs.
