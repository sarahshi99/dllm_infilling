# H200 Controller Feasibility Audit

audit_verdict: `mixed_controller_failure`
branch: `codex/risk-controlled-dynamic-rescue`
commit: `2db9a778da5696f333067f8456c07c7c7d7941d4`

## Label Distribution

- validation recoverable rows: `17/127`
- validation harmable rows: `67/127`
- validation action-row benefit/harm: `31/155`

## Feature Ranking

- best validation recoverability feature variant: `trace_only`
- AUC: `0.7342245989304813`
- top-k recoverable precision: `0.29411764705882354` with k=`17`

## Risk Certification

- zero-harm interventions required for 5% conditional harm upper95: `59`
- Conditional risk is `P(harm | intervene)`.
- Population policy harm is `P(intervene and harm)` over all validation rows.
- Net gain is `P(benefit) - P(harm)`.

## Reasons

- calibration has fewer recoverable rows than zero-harm 5% certification needs
- non-KEEP action harm labels strongly outnumber benefit labels
- some recoverability ranking signal exists outside the first controller
