# Experiment Plan v3: Probe-Curve-First Length Modeling

Updated: 2026-05-31 15:24 CST

## Objective

Advance from the current `midcons` A6000 checkpoint toward a CCF-A-grade length-control method for DLLM code infilling. The plan prioritizes reproducible diagnostics before expensive GPU runs.

## Baseline And Environment

- Dataset: `HumanEval-SingleLineInfilling`, test split, `1033` tasks.
- Current model: `GSAI-ML/LLaDA-8B-Base`.
- Same-hardware baseline: A6000 union control, `787/1033 = 76.19%`.
- Current checkpoint: A6000 `midcons`, `795/1033 = 76.96%`.
- GPU policy: future experiments must use GPU `2,3` only, unless the user explicitly changes the allocation. Do not kill, preempt, or interrupt existing processes; wait or queue if cards `2,3` are occupied. This plan continues with CPU-only analysis until offline gates justify a smoke run.

## Engineering Review Summary

Architecture: keep legacy `expvision_dllm/` and `scripts/` frozen. New work belongs in `expvision_dllm_clean/`, `clean_scripts/`, `analysis/`, `analysis_outputs/`, and `docs/paper_agent/`.

Data flow:

```text
existing results.jsonl
  -> compact evidence builder / diagnostic script
  -> paper-agent evidence snapshot and probe-curve audit
  -> experiment_results and dashboard
  -> GPU runner only if offline criteria pass
```

Failure posture: no GPU policy should be launched from a heuristic that has not passed offline short-risk and recall checks. Raw outputs stay local; compact summaries are tracked.

## Datasets

Primary:

- `HumanEval-SingleLineInfilling`, `1033` tasks.

Required next validation:

- HumanEval-Infilling multi-line if protocol alignment is possible.
- SantaCoder-FIM or another FIM-style benchmark only after the single-line story is stable.

## Baselines

Internal baselines:

- fixed length.
- oracle length.
- CAL-lite.
- LCAS-v3b.
- LCAL.
- official-CAL bounded repair.
- A6000 union control.
- A6000 `midcons`.

External anchors:

- DreamOn dynamic canvas.
- LR-DLLM length regularization.
- Autoregressive code infilling models only under explicitly matched settings.

## Metrics

Primary:

- pass@1.
- pairwise wins/losses vs same-hardware control.
- oracle-length bucket pass rates: `<=8`, `9-12`, `13-16`, `17-24`, `25+`.

Secondary:

- selected-minus-oracle length error.
- under-selection rate.
- trigger count and trigger precision by source.
- short-risk rate for proposed long detectors.
- current-pass risk rate.
- decode/probe overhead.

## Experiment Phases

### E0: Evidence Snapshot

Build a compact paper-agent evidence snapshot from existing raw outputs. This must verify row counts, pass rates, bucket metrics, wins/losses, and long-failure under-selection counts.

Success: generated summary matches existing reports and is tracked as compact documentation.

### E1: Long-Signal Diagnostic Expansion

Extend CPU-only diagnostics beyond existing scalar result fields, starting with probe-curve shape features because current A6000 full-run outputs do not contain saved `stopping_trace` or `step_traces`.

Available now:

- length-probe curve shape features;
- disagreement between base, official-CAL, and long probe selections;
- failure signatures that do not require oracle at inference time.

Not available from the current full run:

- denoising trajectory summaries from `stopping_trace` or step traces.

Fresh CPU audit result:

- `analysis/analyze_probe_curve_long_signals.py` evaluated `4106` single-feature probe-curve thresholds.
- `strict_viable_thresholds = 0`.
- The best threshold had `63.04%` true-long precision and `31.87%` failed-long recall, but `8.70%` short-risk, above the `5%` safety gate.

Success: find a candidate signal or learned score with low short-risk and enough failed-long recall to justify a smoke GPU run.

Kill: no candidate with short-risk `<=5%` and at least `10` failed-long triggers, unless a clearly argued lower-precision/high-recall tradeoff is documented.

### E2: Learned Length Classifier Or Scorer

If E1 single-feature rules fail, train or fit a lightweight length-risk classifier on existing diagnostic fields and validate with strict split discipline. This must not use oracle at inference time.

Success: held-out diagnostic precision/recall beats hand rules and preserves short safety.

Kill: classifier relies on run-specific artifacts that do not transfer across model family or environment.

### E3: Trace-Enabled GPU Smoke Then Full Run

Only after E1 or E2 passes offline gates, run a smoke experiment on GPU `2,3` when those cards are available or can be waited for safely. If the next hypothesis depends on trajectory information, the smoke run must enable `--save-step-traces` so the missing trajectory signal is actually captured. Then run full `1033` only if smoke results do not show short regression.

Command pattern:

```bash
CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false <runner command>
```

Success: full candidate improves total pass rate and long buckets without short-bucket regression.

Kill: any controlled short-bucket regression not offset by a strong, documented long-bucket gain.

### E4: Cross-Model Protocol Alignment

Re-run the stable protocol on cached Dream-Coder Base/Instruct and LLaDA Instruct where available. Add Dream/DiffuCoder only if model cache or network access is available.

Success: method effect transfers or yields an interpretable non-transfer result.

Kill: prompt/canvas mismatch prevents apples-to-apples interpretation.

## Compute Budget

CPU-only diagnostics should run first and can proceed now. GPU runs should wait for available cards or use wait/queue scripts. Full runs over `1033` samples require explicit manifests naming model, command, GPU set, output directory, and comparison baseline.

## Reproducibility Requirements

Every reported experiment must include:

- baseline;
- environment;
- GPU set;
- model;
- command;
- output directory;
- total pass rate;
- bucket metrics;
- wins/losses;
- same-hardware or cross-hardware comparison label.

## Priority

1. Evidence snapshot and paper-agent docs.
2. Probe-curve multivariate or learned diagnostic scoring for long under-selection.
3. Trace-enabled GPU smoke experiment only after diagnostic gates.
4. Full A6000 candidate.
5. Cross-model protocol-matched validation.

## Current Decision

Use `midcons` as the current checkpoint. Do not launch another full true-long GPU run from the existing official-CAL gate family or from a single-feature probe-curve threshold.
