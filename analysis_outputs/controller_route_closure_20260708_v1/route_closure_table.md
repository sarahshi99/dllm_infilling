# Controller Route-Closure Table

Verdict: `controller_route_closed_for_current_paper_no_v4`.

This table closes Controller V1/V2/V3 as validation-only weak/negative evidence. Frozen test remains sealed and no Controller V4 is authorized without a new evidence source.

| Controller | Oracle upper bound | Selected policy | Interventions | Validation wins/losses | Validation pass | Harm upper95 | Gate | Frozen-test decision |
|---|---|---|---:|---:|---:|---|---|---|
| `V1 H200 replay` | 106/127; 17 wins / 0 losses | logistic probe_only + benefit_only, threshold 999.0 (zero intervention) | 0 | 0/0 | 89/127 | n/a; no nonzero gate-passing calibration point | failed | sealed; test_evaluation_count=0 |
| `V2 H200 validation` | 106/127 inherited H200 action-bank upper bound | no gate-passing deployable policy; best nonzero diagnostic is ordinal_only + probe_trace_fused | 41 for best nonzero diagnostic; 0 for gate-safe selected policy | 5/4 for best nonzero diagnostic | 90/127 for best nonzero diagnostic; 89/127 for zero-intervention families | 7.06% population upper95 for best nonzero diagnostic; above 5% budget | failed | sealed; test_evaluation_count=0 |
| `V3 H200 candidate screen` | same H200 action-bank headroom; top-k oracle-win hits up to 5 in best pass-count point | Family A targeted_missed_long, probe_only, k=5 exploratory policy | 5 | 1/0 | 90/127 | 2.33% population upper95; conditional upper95 45.07% | exploratory gate passed; frozen-test gate failed | sealed; test_evaluation_count=0 |

## Closure Read

- `V1 H200 replay`: Oracle action-bank headroom exists, but first deployable selector collapses to KEEP under harm budget.
- `V2 H200 validation`: Recoverability signal exists but cannot satisfy risk budget; short bucket also regresses for best nonzero point.
- `V3 H200 candidate screen`: Safer top-k signal is too weak for frozen-test authorization; stronger pass-count point has a short-bucket loss.

## Evidence Paths

- `V1 H200 replay`: `analysis_outputs/controller_validation_h200_20260707_v1_replay/report.md`
- `V2 H200 validation`: `analysis_outputs/controller_v2_h200_20260707_phase3_v2_validation/report.md`
- `V3 H200 candidate screen`: `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/report.md`

## Paper Use

Use this as a negative/diagnostic controller table: oracle headroom exists, but deployable risk-controlled selection does not clear validation and frozen-test gates. Do not present V1/V2/V3 as positive deployable controller results.
