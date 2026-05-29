# 历史结果归档设计

## Objective

整理本地历史实验输出，让有价值结果可查、可引用、可复现，同时避免把大型 raw artifacts 提交到普通 git。

需要回答：

- 哪些 runs 是 canonical？
- 每个 run 的 pass rate 和 bucket breakdown 是什么？
- raw outputs 在哪里？
- 哪些结果已经 superseded，哪些仍应作为 baseline 或 diagnostic reference？

## Why Raw Outputs Stay Out Of Normal Git

`outputs_clean/` 中包含大量 `results.jsonl`、logs 和 trace-heavy artifacts。普通 git 不适合承载这些文件，因为：

- repository 会快速膨胀；
- checkout 和 review 成本高；
- 大量 raw lines 对 GitHub 页面阅读价值低；
- 未来若需要保留 raw，应使用 Git LFS 或外部 artifact store。

普通 git 应只保存紧凑、可审阅的记录。

## Archive Structure

建议结构：

- `docs/results/run_registry.md`：人类可读 run table。
- `docs/results/run_registry.json`：机器可读 metadata。
- `docs/results/archive_notes.md`：归档策略与 canonical run 解释。
- `docs/results/model_generalization_registry.md/json`：cross-model runs。
- `analysis_outputs/experiment_scoreboard.md`：主 scoreboard。

每条 run metadata 应尽量包含：

- run id；
- raw path；
- model；
- sample count；
- pass count/rate；
- bucket metrics；
- status；
- mask length source；
- 是否有 summary。

## Triage Rules

- `global_best`：曾经或当前主 checkpoint。
- `env_control`：环境迁移或同策略对照。
- `candidate`：有意义的候选方法。
- `superseded`：被后续方法替代但仍有参考价值。
- `diagnostic`：用于解释失败、上界、下界或参数敏感性。

默认排除 smoke runs 和 incomplete runs；只有审计时显式包含。

## First Archive Pass

第一轮归档应覆盖：

- fixed-length baseline；
- oracle-length diagnostic；
- CAL-lite 系列；
- LCAS/LCAL 系列；
- bounded repair/union checkpoints；
- A6000 control 和 `midcons`；
- Dream-Coder/LLaDA cross-model runs。

## Success Criteria

- 可以在 GitHub 上阅读主要历史结果。
- 每个报告结果都有 raw path。
- 不提交 heavy raw files。
- registry 可重复生成。
- 结果解释和 artifact policy 明确。

## Risks

- 旧 runs 的 summary schema 不一致；
- 部分 runs 缺失 bucket metrics；
- Cross-model protocol 不匹配，不能直接做 SOTA claim；
- 如果不记录失败方向，未来容易重复消耗 GPU。
