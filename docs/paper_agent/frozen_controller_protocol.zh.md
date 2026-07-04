# Frozen Controller Protocol

更新时间：2026-07-03 CST

## Central Claim

Unknown-length DLLM infilling exhibits two coupled but separable regimes: canvas inadequacy and rescue inadequacy. Missed true-long failures are substantially trigger/canvas-limited, whereas already-triggered failures remain rescue-limited under the current longer-trajectory and trace-remasking action family. The main deployable opportunity is therefore risk-controlled, non-oracle prediction of when and how much to expand.

中文：unknown-length DLLM infilling 至少有 canvas inadequacy 与 rescue inadequacy 两个相互耦合但可分离的 regime。missed true-long failures 主要受 trigger/canvas 限制；已经触发的 failures 在当前 longer-trajectory 和 trace-remasking action family 下仍然 rescue-limited。因此当前可部署机会是风险受控、非 oracle 地预测何时扩展以及扩展到多长。

## Split Lock

- grouped split: `analysis_outputs/grouped_split_20260702_accel2`
- test lock: `analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json`
- train：拟合 controller 参数。
- calibration：概率校准与 harm threshold / operating point。
- validation：方法、ablation 与 operating point 选择。
- test：只允许最终冻结方法评测一次。

## 禁止项

- 不得把 oracle action-ceiling 结果写成 held-out controller result。
- 不得用 test pass/fail、error type、reference code、oracle length 或 action outcome 构造 inference feature。
- validation gate 通过前，test lock 必须保持 `sealed` 且 `test_evaluation_count=0`。
- test 运行后不得反向修改 feature、model、threshold 或 action set。
- E/F/G remasking 只作为 diagnostic actions，不进入 deployable action set。

## Deployable Action Set

- `KEEP_PRIMARY`
- `EXPAND_16`
- `EXPAND_24`
- `EXPAND_32`
- `EXPAND_48`
- actual canvas 定义：`max(primary_selected_length, target_length)`。

## Validation Gate

- calibration intervention harm 的 95% upper confidence bound 不超过 5%。
- validation `<=8` bucket 不出现净 regression。
- validation 总 Pass@1 不低于 V6 same-protocol baseline。
- validation 至少满足：总 Pass@1 提高、long bucket 提高且 aggregate 不下降、同准确率下降低 compute，或 risk-coverage Pareto 改善。
