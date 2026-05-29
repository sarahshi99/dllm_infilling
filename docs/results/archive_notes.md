# Result Archive Notes

Raw experiment outputs remain local under `/home/shx/projects/dllm_infilling/outputs_clean`.

Normal git stores compact records:

- `docs/results/run_registry.md`
- `docs/results/run_registry.json`
- `docs/results/historical_result_analysis.md`
- `docs/results/model_generalization_registry.md`
- `docs/results/model_generalization_registry.json`
- `analysis_outputs/experiment_scoreboard.md`
- compact CSV/JSON diagnostics under `analysis_outputs/`

Canonical historical runs:

- `old_union_gpus23`: old global checkpoint, `787/1033 = 76.19%`.
- `union_gpus01_control`: same-policy control on the newer environment, `785/1033 = 75.99%`.
- `eval12_nomiddle_gpus01_control`: gate-control run, same pass/fail set as `union_gpus01_control`.
- `midcons`: best newer-environment candidate, `791/1033 = 76.57%`.
- `midaggr`: record-only candidate, more wins but too many short/mid losses.

Cross-model local records:

- `model_generalization_runs/20260512_114917_lcas_v3_full`: LCAS v3 runs for LLaDA and Dream-Coder.
- `model_generalization_runs/20260513_dreamcoder_official_full`: Dream-Coder official-canvas full runs.

Interpretation:

- `docs/results/historical_result_analysis.md` explains which runs are canonical, which directions were superseded, and why the current A6000 `midcons` run is the best same-hardware checkpoint.

Artifact policy:

- Do not commit raw full `outputs_clean/` to normal git.
- Use Git LFS or external storage only for selected raw files that must be preserved verbatim.
- Never delete local raw outputs during archive work without explicit user approval.
- Registry generation excludes smoke runs and incomplete no-summary runs by default. Use `analysis/build_run_registry.py --include-smoke` or `--include-incomplete` only for audits.
