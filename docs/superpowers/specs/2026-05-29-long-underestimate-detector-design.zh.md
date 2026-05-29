# Long Underestimate Detector 设计

## Objective

在启动新的 GPU true-long rescue 实验前，先用已完成 A6000 `midcons` 结果做离线诊断，判断当前 result fields 是否包含足够安全的 long-underestimation trigger。

## Evidence

现有 A6000 证据：

- `midcons` 改善 short/medium，但 `17-24` 和 `25+` 不变。
- `true_long` branch 在 safety gates 后触发 0 次。
- `midcons` long failures 中，`90/91` 是 under-selected。
- `71/91` long failures 最终仍来自 `base`。

这说明 long-tail 问题真实存在，但当前 official-CAL trigger 不够可靠。

## Design

先做只读 offline sweep：

- 输入：`midcons` 的 `results.jsonl`。
- 输出：candidate rules 的 precision/recall/risk table。
- 不加载模型。
- 不改变任何 result。
- 不启动 GPU。

候选 signals：

- selected length 很短；
- best long length 足够大；
- gap 较大；
- long ratio/raw score 足够强；
- support count；
- final source。

评价每条 rule：

- triggers；
- true-long precision；
- failed-long recall；
- short-risk；
- current-pass risk。

## Why This First

直接启动 GPU true-long policy 风险高，因为现有 signals 可能把 short/medium false positives 当成 long cases。离线 sweep 能在不消耗 GPU 的情况下判断 signal 是否足够判别。

## Success Criteria

理想 rule 应满足：

- trigger count 不太小；
- true-long precision 高；
- short-risk 低；
- 能覆盖足够 failed-long cases；
- 不主要影响已经 passing 的 short samples。

实际结果中，没有 rule 满足严格标准。

## Implementation Notes

- 新增 `analysis/diagnose_long_underestimate_policy.py`。
- 新增 `tests/test_diagnose_long_underestimate_policy.py`。
- 输出在 `analysis_outputs/long_underestimate_detector/a6000_midcons/`。
- 报告写入 `docs/results/long_underestimate_detector_report.md`。

## Self-Review

该设计满足用户当前“不启动新 GPU 实验”的约束，并把 long-tail 决策建立在已有结果证据上。负面结果本身有研究价值：它说明当前 confidence fields 不足以安全解决 true-long infilling，需要更强 length modeling。
