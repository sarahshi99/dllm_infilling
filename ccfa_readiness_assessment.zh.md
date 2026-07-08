# CCF-A Readiness Assessment

更新时间：2026-07-08 UTC

## 总体判断

当前项目已经从“继续调 controller”转为 `diagnostic-driven mixed paper`。这不是一个 positive controller paper：H200 Controller V1/V2/V3 都显示 deployable risk-controlled intervention 只有弱验证信号，frozen test 仍然 sealed，`test_evaluation_count=0`。

Claim Readiness Gate verdict：`diagnostic_mixed_candidate_with_blocking_gaps`。

最稳妥的论文主张是：

> Unknown-length DLLM infilling has separable canvas-limited and rescue-limited regimes. Missed true-long cases expose substantial oracle-canvas recoverability, while already-triggered long failures remain resistant to longer trajectories and trace-guided remasking. Deployable risk-controlled control shows weak but insufficient validation signal, revealing a gap between diagnostic upper bound and safe inference-time intervention.

## 已有强证据

1. H200 evidence base 已被研究者接受，主表和后续 controller 以 H200 rerun 为准；A6000 只保留为 historical reference。
2. H200 core baselines 已全量重跑：Control `787/1033`，Midcons `794/1033`，Route2 `795/1033`，V6 `796/1033`，Local CAL `769/1033`。
3. H200 action bank 覆盖 frozen train/calibration/validation：`927` tasks × `5` actions = `4635` rows，test rows `0`。
4. Oracle action-bank validation upper bound 为 `106/127`，相对 primary `17/0` wins/losses，说明 action bank 中确实存在可恢复空间。
5. True-long attribution 的核心机制清晰：C oracle-sufficient canvas 恢复 `29/89` hard cases，全部来自 missed failed-long；triggered failed-long 为 `0/33`，支持 canvas inadequacy 与 rescue inadequacy 分离。
6. Controller V1/V2/V3 的负结果是受控的：所有开发都停在 validation，frozen test 未打开。

## Weak Evidence

1. Controller V3 有弱 validation signal：保守 top-k policy 可做到 `90/127`、`1/0` wins/losses、population harm upper95 `2.33%`，但净增只有 `+1`。
2. Family A 的最高 pass-count 点为 `91/127`、`3/1` wins/losses、net `+2`，但出现一个 `<=8` short-bucket loss，不能授权 frozen test。
3. Second-backbone feasibility audit 推荐 Dream-Coder Base：checkpoint 已缓存，runner 与 evaluator 可复用，历史 full SingleLine evidence 存在。但 fresh oracle-sufficient H200 diagnostic 被工具审批层阻塞，当前只能记录为 feasibility + extracted diagnostic，而不是完整 second-backbone confirmation。
4. Dream-Coder 现有结果支持“可做跨 backbone 诊断”的可行性，但还不足以声称 model-agnostic generalization。

## Negative Evidence

1. Controller V1 H200 replay 仍为 zero intervention：validation `89/127`，wins/losses `0/0`。
2. Controller V2 最好的非零点为 `90/127`、`5/4` wins/losses，population harm upper95 `7.06%`，未通过 primary gate。
3. Controller V3 没有满足 frozen-test gate 的 deployable policy；最终 route decision 为 `weak_validation_signal_test_sealed`。
4. LR-DLLM final attempt verdict 为 `blocked_missing_algorithmic_detail`：仓库中没有 protocol-matched Stage I/II adapter，不能称为 official reproduction。
5. Second-regime audit 发现 MultiLine/RandomSpan alias 存在，但本机 `data/` 缺少所需 JSONL，因此未运行 second-regime diagnostic。

## CCF-A Blocking Gaps

1. 缺少完整 second-backbone minimal diagnostic：Dream-Coder fresh oracle-sufficient canvas run 尚未执行成功。
2. 缺少 second-regime diagnostic：MultiLine 或 RandomSpan 本地数据文件缺失。
3. deployable controller 没有达到 frozen-test gate；因此没有 sealed test improvement。
4. LR-DLLM 没有同协议 baseline，只能作为 blocked baseline 记录。
5. 当前 paper contribution 需要靠机制诊断、负结果严谨性和泛化 audit 支撑，不能写成 SOTA 方法论文。

## Next Required Experiments

1. 手动或在审批层修复后运行 Dream-Coder Base oracle-sufficient canvas minimal diagnostic，输出 second-backbone diagnostic 的 fresh results。
2. 补齐 `HumanEval-MultiLineInfilling` 或 `HumanEval-RandomSpanInfilling` 本地 JSONL，运行 second-regime minimal diagnostic。
3. 若要增强 CCF-A 竞争力，优先验证 central claim 是否跨 backbone 或跨 regime 成立，而不是继续在人类验证集上调 Controller V4。
4. 如果 LR-DLLM 官方代码或足够算法细节释放，再重做 protocol-matched Stage I sanity；在此之前不要把 heuristic local adapter 写成 LR-DLLM reproduction。

## Paper Framing Decision

采用 `diagnostic-driven mixed paper`：

- 主贡献：揭示 unknown-length DLLM infilling 中 canvas-limited 与 rescue-limited 两种机制，以及 action-bank oracle upper bound 与 safe deployable controller 之间的缺口。
- 正结果：oracle/action-ceiling 与部分 validation top-k 证明恢复空间存在。
- 负结果：V1/V2/V3 controller 无法安全转化为 frozen-test policy。
- 不声称：SOTA、controller success、unknown-length solved、frozen test improvement、model-agnostic generalization。

当前可以开始论文骨架和正式写作，但 CCF-A 强投稿仍需要至少一个泛化诊断补强：second-backbone fresh diagnostic 或 second-regime diagnostic。
