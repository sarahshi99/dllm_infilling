# Experiment Results

Updated: 2026-05-31 15:56 CST

## Current A6000 Checkpoint

Baseline: A6000 union control.

Environment: local A6000 environment recorded in prior result reports.

GPU set: A6000 control and candidate runs were reported as A6000 runs; future reports must name exact `CUDA_VISIBLE_DEVICES`. Future GPU experiments in this agent plan should use `CUDA_VISIBLE_DEVICES=2,3` unless the user changes the allocation.

Model: `GSAI-ML/LLaDA-8B-Base`.

Dataset: `HumanEval-SingleLineInfilling`, test split, `1033` tasks.

Command references:

- Full recovery launcher: `bash clean_scripts/resume_lcal_a6000_four_policies_offline.sh`.
- Pairwise analysis: `/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_lcal_pairwise.py ...`.
- Scoreboard generation: `/home/shx/miniconda3/envs/dllm_env/bin/python analysis/build_a6000_scoreboard_section.py`.
- Paper-agent evidence snapshot: `/home/shx/miniconda3/envs/dllm_env/bin/python analysis/build_paper_agent_evidence_snapshot.py`.

Output directories:

- Control: `outputs_clean/full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529`.
- `midcons`: `outputs_clean/full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626`.
- `mid_precision`: `outputs_clean/full_lcal_official_bounded_repair_mid_precision_supp2_best13_16_veto13_a6000_20260528_221517`.
- `true_long`: `outputs_clean/full_lcal_official_bounded_repair_true_long_off17_d8_r085_supp2_a6000_20260528_221755`.
- `combined`: `outputs_clean/full_lcal_official_bounded_repair_mid_precision_plus_true_long_a6000_20260528_221756`.

## Main Table

| Run | Pass | Rate | Delta vs A6000 control | `<=8` | `9-12` | `13-16` | `17-24` | `25+` | Comparison |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A6000 control | `787/1033` | `76.19%` | baseline | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | same-hardware |
| `midcons` | `795/1033` | `76.96%` | `+8` wins, `0` losses | `89.97%` | `78.45%` | `58.89%` | `20.73%` | `16.13%` | same-hardware |
| `mid_precision` | `787/1033` | `76.19%` | `0` wins, `0` losses | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | same-hardware |
| `true_long` | `787/1033` | `76.19%` | `0` wins, `0` losses | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | same-hardware |
| `combined` | `787/1033` | `76.19%` | `0` wins, `0` losses | `89.80%` | `77.16%` | `54.44%` | `20.73%` | `16.13%` | same-hardware |

## Interpretation

`midcons` is the current A6000 best checkpoint. It gives a same-hardware improvement with no pairwise losses, concentrated in short-to-medium and medium buckets.

The true-long line is negative evidence. Existing true-long gates trigger zero useful changes under safety constraints, and the offline sweep found no safe heuristic rule from the current scalar result fields.

The fresh paper-agent evidence snapshot independently recomputed the central metrics from raw local `results.jsonl` files and wrote:

- `docs/paper_agent/evidence_snapshot.json`
- `docs/paper_agent/evidence_snapshot.md`

It confirms:

- `midcons` pairwise result: `8` wins, `0` losses, `+8` net.
- `midcons` long failures: `91` failed `oracle >= 17` rows.
- Under-selected failed long rows: `90/91 = 98.90%`.
- Failed long rows still ending from `base`: `71`.
- Long-underestimate sweep: `16776` evaluated rules, `0` strict viable rules.

## Probe-Curve Signal Audit

Command:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_probe_curve_long_signals.py
```

Tracked outputs:

- `docs/paper_agent/probe_curve_signal_audit.json`
- `docs/paper_agent/probe_curve_signal_audit.md`
- `docs/paper_agent/probe_curve_signal_audit.zh.md`

Result:

- rows with probe-curve features: `1033/1033`.
- rows with stopping traces: `0/1033`.
- evaluated single-feature thresholds: `4106`.
- strict viable thresholds: `0`.
- best threshold: `long_score_max <= 0.229253`, with `63.04%` true-long precision, `31.87%` failed-long recall, `8.70%` short-risk, and `2.17%` current-pass risk.

Interpretation: existing probe-curve scalar features are informative but not safe enough as a direct GPU policy. The next CPU step should be multivariate or learned scoring; trajectory analysis requires a trace-enabled smoke run.

## Paper Relevance

This supports a narrow but honest paper claim: medium-length under-selection can be repaired safely by confidence-curve agreement. It does not yet support a broad CCF-A claim or a SOTA claim.

## Next Result Needed

The next result should be one of:

- a diagnostic feature snapshot proving a stronger long-tail signal exists;
- a smoke GPU run showing no short-bucket regression;
- a full same-hardware run improving long buckets;
- or a rigorous negative result that justifies pivoting toward dynamic canvas or length regularization.
