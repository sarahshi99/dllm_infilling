# Completed Experiment Analysis, 2026-05-29

## Scope

This note consolidates the work requested for the current DLLM code infilling phase after the A6000 runs completed. No new GPU experiments are launched in this phase because the GPUs are reserved for other users. The goal here is to make the completed results reviewable, reproducible, and useful for the next research decision.

## Superpowers Used

- `receiving-code-review`: used because an external review reported three P2 issues in result reporting and GPU launch behavior.
- `test-driven-development`: used for the reviewed bugfixes; the repository now has regression tests for missing pairwise comparisons, singular bucket-rate keys, and GPU override behavior.
- `verification-before-completion`: used before reporting that review issues are addressed or experiments are complete.
- `writing-plans`: used earlier for the CCF-A roadmap; this note updates the execution status of that plan.

## Review Fix Status

The P2 review items are resolved in the current worktrees:

| Review item | Current behavior | Verification |
|---|---|---|
| Missing pairwise comparison was mislabeled as `baseline` | Non-control rows now render `comparison missing`; only the real control row renders `baseline` | `tests/test_build_a6000_scoreboard_section.py` |
| Registry ignored singular `oracle_bucket_pass_rate` | Registry reads `oracle_bucket_pass_rates`, `oracle_bucket_pass_rate`, and `bucket_pass_rates` | `tests/test_build_run_registry.py` |
| Wait script checked one GPU set but launched on hardcoded `0,1` | `wait_and_run_lcal_a6000_baselines.sh` passes `GPU_IDS` through `CUDA_VISIBLE_DEVICES` | `tests/test_a6000_launchers.py` |

Verification commands:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest \
  tests/test_build_a6000_scoreboard_section.py \
  tests/test_a6000_launchers.py

PYTHONPATH=. /home/shx/miniconda3/envs/dllm_env/bin/python \
  tests/test_build_run_registry.py
```

## Completed A6000 Runs

All five A6000 runs have `1033` rows and `summary.json`:

| Run | Raw output directory | Pass | Delta vs A6000 control |
|---|---|---:|---:|
| `a6000_control` | `outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529` | `787/1033 = 76.19%` | baseline |
| `midcons` | `outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626` | `795/1033 = 76.96%` | `+8` wins, `0` losses |
| `mid_precision` | `outputs_clean/full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000_20260528_221517` | `787/1033 = 76.19%` | `0` |
| `true_long` | `outputs_clean/full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000_20260528_221755` | `787/1033 = 76.19%` | `0` |
| `combined` | `outputs_clean/full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000_20260528_221756` | `787/1033 = 76.19%` | `0` |

Compact tracked outputs:

- `analysis_outputs/a6000_midcons_longrescue/a6000_scoreboard_section.md`
- `analysis_outputs/a6000_midcons_longrescue/*_vs_a6000_control/summary.json`
- `analysis_outputs/a6000_midcons_longrescue/*_vs_a6000_control/bucket_summary.csv`
- `analysis_outputs/a6000_midcons_longrescue/*_vs_a6000_control/pairwise.csv`
- `docs/results/a6000_midcons_longrescue_report.md`
- `docs/results/long_underestimate_detector_report.md`

## Main Finding

`midcons` is the current A6000 checkpoint. It is the only completed candidate that improves the same-hardware control, and it does so without pairwise losses.

Bucket-level effect:

| Oracle bucket | Control pass rate | `midcons` pass rate | Net change |
|---|---:|---:|---:|
| `<=8` | `89.80%` | `89.97%` | `+1` |
| `9-12` | `77.16%` | `78.45%` | `+3` |
| `13-16` | `54.44%` | `58.89%` | `+4` |
| `17-24` | `20.73%` | `20.73%` | `0` |
| `25+` | `16.13%` | `16.13%` | `0` |

The eight `midcons` wins are all recovered by `official_mid_rescue`:

| Task | Oracle len | Control len | `midcons` len |
|---|---:|---:|---:|
| `SingleLineInfilling/HumanEval/139/L2` | 13 | 8 | 13 |
| `SingleLineInfilling/HumanEval/143/L5` | 12 | 9 | 12 |
| `SingleLineInfilling/HumanEval/37/L5` | 9 | 6 | 12 |
| `SingleLineInfilling/HumanEval/39/L3` | 8 | 6 | 13 |
| `SingleLineInfilling/HumanEval/73/L1` | 13 | 9 | 13 |
| `SingleLineInfilling/HumanEval/75/L12` | 11 | 4 | 11 |
| `SingleLineInfilling/HumanEval/80/L3` | 13 | 8 | 11 |
| `SingleLineInfilling/HumanEval/82/L1` | 13 | 6 | 13 |

Interpretation: the conservative medium rescue is useful because it fixes under-selected short-to-medium completions. It does not solve true long infilling.

## Negative Evidence

`mid_precision`, `true_long`, and `combined` are not improvements. Their value is diagnostic:

- extra mid precision guards were too conservative and removed the useful mid rescue behavior;
- true-long rescue triggered zero times under the safety gates;
- combining the two inherited both problems.

The `true_long` summary records:

- `official_true_long_rescue_trigger_count = 0`;
- `official_long_suspicion_trigger_count = 11`;
- `official_long_suspicion_true_long_precision = 27.27%`;
- `official_repair_true_long_precision = 8.86%`.

That is not enough evidence for another same-family long-rescue GPU run.

## Long-Tail Diagnosis

For the `midcons` A6000 run:

- oracle length `>=17`: `113` tasks;
- passed long tasks: `22`;
- failed long tasks: `91`;
- under-selected failed long tasks: `90/91`;
- failed long tasks whose final source is still `base`: `71/91`.

Most failed long samples select very short lengths. The top selected lengths among failed long tasks are:

| Selected length | Count |
|---:|---:|
| 3 | 39 |
| 6 | 8 |
| 9 | 7 |
| 7 | 6 |
| 13 | 5 |
| 4 | 5 |
| 12 | 4 |

The offline long-underestimate sweep evaluated `16776` rules. The best rule had only `35.48%` true-long precision, `36.26%` failed-long recall, and `40.86%` short-risk. No strict viable rule was found.

Decision: stop spending GPU on the current official-CAL true-long gate family. The next credible long-tail experiment needs a stronger signal, such as denoising trajectory features, a learned length classifier, DreamOn-style dynamic canvas control, or LR-DLLM-style length regularization.

## Literature Position

The current result should not be called SOTA.

Local `midcons` on LLaDA-8B-Base reaches `76.96%` on the local `HumanEval-SingleLineInfilling` protocol. DreamOn reports much higher HumanEval-Infilling single-line pass@1 for DreamCoder/DiffuCoder with its dynamic canvas method, but the model family, canvas protocol, and evaluation setup must be matched before making a direct claim. LR-DLLM is also directly relevant because it targets unknown-length generation, but its reported setting is not yet matched by the current local protocol.

Current paper-positioning claim should be narrower:

> Same-hardware evidence shows that confidence-curve agreement can recover medium-length DLLM code infilling without short-bucket regressions, but long-tail infilling remains dominated by length underestimation and requires a stronger length modeling mechanism.

## Requirement Tracker

| Requirement from discussion | Status | Where to inspect |
|---|---|---|
| Use Git worktrees | Done | `git_workspace/.worktrees/a6000-midcons-longrescue`, `git_workspace/.worktrees/result-archive` |
| Preserve `outputs_clean/` | Done | raw outputs remain local and symlinked into worktrees |
| Do not commit raw heavy outputs | Done | compact reports and summaries are tracked instead |
| Add project rules about collaboration and challenging assumptions | Done | `git_workspace/AGENTS.md` and copied worktree `AGENTS.md` |
| Explain/use spec files | Done | `docs/superpowers/specs/*.md` contain design contracts; `docs/superpowers/plans/*.md` contain executable plans |
| Restore/use git metadata | Done | worktrees are attached to branches and pushed; raw non-git root remains separate |
| Rerun midcons on A6000 | Done | `midcons` A6000 run above |
| Analyze completed experiments | Done in this note plus linked reports | `docs/results/*` and `analysis_outputs/*` |
| Literature comparison | Partial, source-checked anchors recorded; no SOTA claim | `docs/results/literature_sota_notes.md` |
| Cross-model validation | Historical registry done; new runs paused by current no-GPU instruction | result archive `docs/results/model_generalization_registry.md` |

## Next Engineering Plan

Do not start new GPU jobs until the user says the cards are available again.

Next CPU-only work that can proceed safely:

1. Keep improving result archive interpretation and paper tables.
2. Draft the method section around medium rescue and long-tail failure analysis.
3. Convert the long-tail diagnosis into a new spec for a trajectory-feature or learned length-detector experiment.

Next GPU work, only after hardware is available:

1. Run a small smoke test for any new long detector before full evaluation.
2. Re-run cross-model checks only under matched prompt/canvas settings.
3. Compare to DreamOn/LR-DLLM only after protocol fields are aligned.
