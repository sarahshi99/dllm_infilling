# Research Planning CPU Claim Audit

Verdict: `claim_boundary_consolidated_cpu_only`.
Branch/HEAD: `codex/risk-controlled-dynamic-rescue` / `a345f656f001e60db94c173401e9c382e4497621`.
Frozen test: `sealed_not_touched`.

## Consolidated Findings

- Dream-Coder 15-case taxonomy: {'canvas_recoverable': 7, 'stable_pass': 7, 'rescue_limited_or_noncanvas': 1}.
- Dream-Coder canvas recoveries by stratum: {'short_primary_pass': 1, 'medium_near_long_underselection': 2, 'missed_failed_long': 1, 'triggered_failed_long': 3}.
- LLaDA attribution anchor: C oracle canvas recovers `29/89` hard cases, missed-long `29`, triggered-long `0`, refinement incremental `2`.
- Controller V3 best validation pass point: `targeted_missed_long` / `probe_trace_fused` / k=`15` gives `91/127`, wins/losses `3/1`, frozen gate `False`.
- Controller V3 selected exploratory policy interventions: `{'win': 1, 'neutral': 4}` across `5` interventions.
- Controller V3 frozen-gate passing points: `0`.
- Synthetic second-regime policy pass counts: `{'control_fixed_fresh': 18, 'best_deployable_cal_lite_fresh': 18, 'oracle_sufficient_canvas_fresh': 18}` out of `{'control_fixed_fresh': 18, 'best_deployable_cal_lite_fresh': 18, 'oracle_sufficient_canvas_fresh': 18}`.

## Claim Boundary

- Strong: LLaDA H200 supports separable canvas-limited vs rescue-limited regimes, with missed-long canvas recoverability and triggered-long resistance.
- Mixed: Dream-Coder supports oracle-canvas recoverability but not the exact LLaDA missed-vs-triggered split on the current 15-case subset.
- Negative: Controller V3 has weak validation signal but no frozen-test authorization; this closes the current controller route unless a new evidence source appears.
- Insufficient: synthetic second-regime minimal is an unblock/sanity result, not official or stress benchmark evidence.

## Outputs

- `summary.json`
- `dreamcoder_case_taxonomy.csv`
