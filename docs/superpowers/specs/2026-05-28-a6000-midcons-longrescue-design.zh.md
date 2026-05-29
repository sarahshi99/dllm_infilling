# A6000 Midcons 与 True-Long Rescue 设计

## Objective

在 A6000 上重新建立可靠同硬件基准，验证两个假设：

- `midcons` 是否在 A6000 上仍能带来 short-safe medium rescue；
- 独立 true-long rescue branch 是否能改善 long buckets，而不损害 short buckets。

## Current Evidence

最新 tracked scoreboard 显示，old union checkpoint 为 `787/1033 = 76.19%`，新环境 control 略低，而 `midcons` 在新环境中改善 medium bucket。此前结果混合了策略变化和环境变化，因此需要 A6000 same-hardware control。

## Baseline Runs

必须先运行：

- A6000 union control；
- A6000 `midcons`；
- pairwise comparison。

成功条件：

- 每个 run 有 `1033` rows；
- 每个 run 有 `summary.json`；
- 每个 candidate 都和同一个 A6000 control 做 pairwise analysis；
- 输出 compact artifacts，不提交 raw `results.jsonl`。

## Candidate Strategy A：Mid-Rescue Precision

目标：保留 `midcons` 的 medium wins，同时降低 false positives。

候选 guards：

- 限制 official selected length；
- 限制 delta；
- 增加 support count；
- 对过大 short jump 加 veto；
- 使用 best-long evidence 防止噪声曲线触发。

评估：

- total pass rate；
- wins/losses vs A6000 control；
- bucket pass rates；
- short-bucket loss rate；
- final source histogram。

## Candidate Strategy B：True-Long Rescue

目标：针对 `17-24` 和 `25+` buckets，建立与 mid rescue 分离的分支。

该分支应先保守或 record-only：

- official length 高；
- delta 大；
- long ratio 强；
- support 充分；
- source gate 保护 short samples。

不应让 true-long branch 在没有证据前影响 `<=8` 和 `9-12`。

## Cross-Model Validation

在 A6000 主线稳定后，再对 Dream-Coder、DiffuCoder、Dream、LLaDA variants 做 validation。每个 cross-model run 必须记录 model path、tokenizer behavior、prompt format 差异、evaluation compatibility、总 pass rate、bucket metrics，以及比较是否真正可比。

## Literature And SOTA Comparison

不得凭记忆声称 SOTA。需要对比：

- DreamOn；
- CAL；
- LR-DLLM；
- 相关 autoregressive code infilling baselines。

所有对比必须标明 dataset、metric、model、canvas/length setting 和 evaluation protocol。

## Result Recording

需要输出：

- A6000 control output path；
- candidate output paths；
- pairwise summaries；
- bucket tables；
- `analysis_outputs/experiment_scoreboard.md` 更新；
- `docs/results/` 下的解释报告。

## Implementation Boundaries

- 新代码优先写在新文件中。
- 不修改 raw outputs。
- 不提交模型权重或大型 raw `outputs_clean/`。
- 不在未验证前把 candidate 设为默认。

## Open Risks

- `midcons` 收益可能小，不足以支撑强论文 claim。
- True-long heuristic 可能缺乏足够精确信号。
- Cross-model confidence curves 可能不迁移。
- Literature protocol 可能不匹配，导致不能直接 claim SOTA。
