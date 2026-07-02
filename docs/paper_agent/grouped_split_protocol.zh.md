# Grouped Split Protocol

更新时间：2026-07-02 CST

## 目的

后续 risk-controlled controller 的拟合、阈值选择、校准和最终评测必须按原始 `HumanEval/<id>` 分组，避免同一原始题目的不同 infill location 跨 split 泄漏。

## 规则

- 分组单位：`HumanEval/<id>`。
- 固定 seed：`20260702`。
- split：train / calibration / validation / test。
- 同一 group 的所有 row 必须只出现在一个 split。
- 当前 oracle action-ceiling / generation-ceiling 结果只能作为 diagnostic ceiling，不能称为 held-out method result。

## 输出

- split manifest: `analysis_outputs/grouped_split_20260702_accel2/split_manifest.json`
- train tasks: `analysis_outputs/grouped_split_20260702_accel2/train_tasks.json`
- calibration tasks: `analysis_outputs/grouped_split_20260702_accel2/calibration_tasks.json`
- validation tasks: `analysis_outputs/grouped_split_20260702_accel2/validation_tasks.json`
- test tasks: `analysis_outputs/grouped_split_20260702_accel2/test_tasks.json`
- row assignment: `analysis_outputs/grouped_split_20260702_accel2/row_split_assignment.csv`

## 使用约束

- controller feature/threshold/rule selection 只能使用 train。
- risk calibration 使用 calibration。
- model selection / ablation selection 使用 validation。
- 最终论文数字只在 test 上报告，并且不得用本轮 oracle canvas results 调整 test 规则。
