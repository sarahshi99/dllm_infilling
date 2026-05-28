# Result Archive Notes

Raw experiment outputs remain local under `/home/shx/projects/dllm_infilling/outputs_clean`.

Normal git stores compact records:

- `docs/results/run_registry.md`
- `docs/results/run_registry.json`
- `analysis_outputs/experiment_scoreboard.md`
- compact CSV/JSON diagnostics under `analysis_outputs/`

Canonical historical runs:

- `old_union_gpus23`: old global checkpoint, `787/1033 = 76.19%`.
- `union_gpus01_control`: same-policy control on the newer environment, `785/1033 = 75.99%`.
- `eval12_nomiddle_gpus01_control`: gate-control run, same pass/fail set as `union_gpus01_control`.
- `midcons`: best newer-environment candidate, `791/1033 = 76.57%`.
- `midaggr`: record-only candidate, more wins but too many short/mid losses.

Artifact policy:

- Do not commit raw full `outputs_clean/` to normal git.
- Use Git LFS or external storage only for selected raw files that must be preserved verbatim.
- Never delete local raw outputs during archive work without explicit user approval.
- Registry generation excludes smoke runs and incomplete no-summary runs by default. Use `analysis/build_run_registry.py --include-smoke` or `--include-incomplete` only for audits.
