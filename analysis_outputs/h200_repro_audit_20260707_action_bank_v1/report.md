# H200 Action Bank and Controller V1 Replay Audit

repro_verdict: `h200_material_outcome_drift`
branch: `codex/risk-controlled-dynamic-rescue`
commit: `c05d2f8191c8ff31cec2e7472970262ed9d01526`

## Action Bank

- H200 rows: `4635`
- H200 task/action coverage: `927` tasks / `4635` rows
- outcome agreement vs old bank: `0.9495145631067962`
- H200 benefit/harm labels non-KEEP: `193` / `1237`
- old benefit/harm labels non-KEEP: `191` / `1242`

## Controller V1 Replay

- old selected: `{"feature_variant": "probe_only", "model_family": "logistic", "policy_variant": "benefit_only", "score_threshold": 999.0}`
- H200 selected: `{"feature_variant": "probe_only", "model_family": "logistic", "policy_variant": "benefit_only", "score_threshold": 999.0}`
- old gate/test decision: `False` / `sealed`
- H200 gate/test decision: `False` / `sealed`
- H200 selected validation pass count: `89`
- H200 selected validation wins/losses vs primary: `0` / `0`

## Decision

- Core Tier 1 baselines already produced `h200_material_outcome_drift`; this remains the server migration verdict.
- Controller V2 stays blocked until the material H200 drift is triaged by the research owner.
- Frozen test remains sealed with evaluation count 0.
