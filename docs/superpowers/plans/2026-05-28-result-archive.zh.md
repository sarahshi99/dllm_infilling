# 历史结果归档实施计划

> **给 agentic worker：** 英文版包含完整代码块与命令。本中文版本翻译目标、架构、任务和验证逻辑；执行代码以英文版为准。

**目标：** 构建可提交的紧凑历史实验结果归档，同时让大型 raw outputs 保留在本地。

**架构：** 新增一个 registry builder，扫描本地 `outputs_clean/` run directories，尽可能提取 config 与 summary metadata，并写出人类可读和机器可读 registry。人工解释保留在 Markdown 中，使研究判断可见，而不是埋在代码里。

**技术栈：** Python 3、JSON/JSONL、Markdown、现有本地 `outputs_clean/` 和 `analysis_outputs/` 目录。

---

## 文件结构

- 创建 `analysis/build_run_registry.py`：扫描 run directories，写出 `docs/results/run_registry.json` 与 `docs/results/run_registry.md`。
- 创建/修改 `docs/results/archive_notes.md`：记录 artifact policy 与 canonical runs。
- 修改 `analysis_outputs/experiment_scoreboard.md`：链接 registry。

## Task 1：新增 Registry Builder

`analysis/build_run_registry.py` 负责：

- 遍历 `outputs_clean/` 与 `outputs_clean/202604`；
- 读取 `summary.json`、`config.json` 和 `results.jsonl`；
- 统计 sample count、pass count、pass rate、model、mask length source、bucket pass rates；
- 过滤 smoke 和 incomplete runs，除非显式传入 flags；
- 写出 JSON 与 Markdown 两种 registry。

验证：

- `python -m py_compile analysis/build_run_registry.py`
- 用本地 outputs 生成 registry；
- 检查 canonical runs 是否出现，且 pass rate 正确。

## Task 2：新增 Archive Notes

`docs/results/archive_notes.md` 应记录：

- raw outputs 的本地位置；
- normal git 只保存 compact records；
- canonical historical runs；
- cross-model local records；
- artifact policy。

核心策略：

- 不提交完整 raw `outputs_clean/`；
- 不删除历史 raw outputs；
- 只有选定高价值 raw 文件才考虑 Git LFS 或外部 artifact storage。

## Task 3：把 Scoreboard 链接到 Registry

在 `analysis_outputs/experiment_scoreboard.md` 中增加指向 registry 的说明，使主 scoreboard 和历史归档互相可发现。

## 已完成结果

当前 `docs/results/run_registry.json` 包含 `49` 个 meaningful local runs。`docs/results/historical_result_analysis.md` 进一步解释了：

- fixed-length baseline；
- oracle-length 上界；
- CAL-lite/LCAS/LCAL 演进；
- A6000 control；
- A6000 `midcons` 当前最佳 checkpoint；
- cross-model historical records。
