# H200 Evidence Base Decision

更新时间：2026-07-07 UTC

## 决策

研究者已接受当前 H200 rerun 结果作为后续 controller、validation、action bank、baseline 和论文主表的 evidence base。

- `h200_material_outcome_drift` 已记录为 reproducibility/audit 事实，但不再作为 Controller V2 的停止条件。
- 旧 A6000 结果保留为 historical reference，不与 H200 结果静默混合。
- 后续 Phase 3 controller 设计、calibration、validation 和主方法比较均以 H200 rerun 结果为准。
- 不再继续排查 H200 环境漂移，除非出现评测 harness 错误、checkpoint 损坏、数据 split 破坏、action bank 错误或 frozen test lock 异常。
- Frozen test 继续保持 `sealed`，`test_evaluation_count=0`。

## H200 Evidence Base

| Artifact | Path |
|---|---|
| Bootstrap audit | `analysis_outputs/h200_bootstrap_20260705_103617/` |
| Tier 1 reproduction audit | `analysis_outputs/h200_repro_audit_20260707_tier1_v2/` |
| Action bank | `analysis_outputs/controller_action_bank_h200_20260707_tier1_offline/` |
| Controller V1 replay | `analysis_outputs/controller_validation_h200_20260707_v1_replay/` |
| Action-bank/controller comparison | `analysis_outputs/h200_repro_audit_20260707_action_bank_v1/` |
| Material drift triage | `analysis_outputs/h200_material_drift_triage_20260707_material_drift_triage/` |

## H200 Baselines

| Run | H200 Pass@1 |
|---|---:|
| Control | `787/1033` |
| Midcons | `794/1033` |
| Route2 | `795/1033` |
| V6 | `796/1033` |
| Local same-protocol CAL | `769/1033` |

Controller V1 H200 replay remains a negative result: zero intervention, validation `89/127`, wins/losses `0/0`, frozen test sealed.

H200 oracle action-bank upper bound on validation is `106/127` with `17` wins and `0` losses. This is diagnostic headroom, not a deployable controller.

## Controller V2 Authorization

Phase 3 Controller V2 is authorized to proceed on H200 train/calibration/validation only:

- risk/label feasibility audit;
- ordinal canvas adequacy model;
- pairwise action ranker;
- independent harm guard;
- calibration and validation;
- one frozen test run only if validation gate passes.

Validation gate failure keeps frozen test sealed. A one-pass gain alone does not authorize test.
