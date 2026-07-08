# H200 Reproduction Audit

verdict: `h200_material_outcome_drift`
branch: `codex/risk-controlled-dynamic-rescue`
commit: `7e7117c186bfc7d2ba5bae924449d1af1926775f`

## Core Baselines

| Run | Old GPU | Old Pass | H200 Pass | Delta | H200 Wins | H200 Losses | Outcome Agreement | Hash Agreement | Avg Sec Old | Avg Sec H200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `control` | A6000 | 787 | 787 | 0 | 4 | 4 | 0.9923 | 0.9622 | 6.025 | 1.323 |
| `midcons` | A6000 | 795 | 794 | -1 | 4 | 5 | 0.9913 | 0.9613 | 11.978 | 1.465 |
| `route2` | A6000 | 801 | 795 | -6 | 3 | 9 | 0.9884 | 0.9555 | 5.462 | 1.556 |
| `v6` | A6000 | 802 | 796 | -6 | 3 | 9 | 0.9884 | 0.9555 | 4.877 | 1.721 |
| `cal` | A6000 | 774 | 769 | -5 | 4 | 9 | 0.9874 | 0.9613 | 3.570 | 1.170 |

## Interpretation

- Reproduction verdict is `h200_material_outcome_drift`.
- Candidate hash mismatch is recorded separately from pass/fail drift because deterministic GPU kernels can still produce text changes without changing outcome.
- Controller V2 must not start automatically when verdict is `h200_material_outcome_drift`.
