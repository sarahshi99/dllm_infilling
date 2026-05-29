# 项目 Agent 准则

## 协作默认规则

- 当需要输入、凭据、硬件访问或决策时，用户会全力配合。
- 工作目标是产生具体研究结果，而不只是整理代码。
- 如果证据指向不同方向，要敢于反驳用户假设，并给出更强方案及利弊。
- 在高影响执行前，先提出必要问题，对齐目标、约束和成功标准。
- 只要仍有有价值的进展可做，就不要静默停止；继续分析、规划、编码或报告，直到遇到真正需要用户决策或外部状态改变的阻塞。

## 项目结构

- 将 `expvision_dllm/` 和 `scripts/` 视为冻结的历史参考代码。
- 新实验优先放在 `expvision_dllm_clean/` 和 `clean_scripts/`。
- 新实验族应新增模块或 runner，不要重新打开大型 legacy 文件。
- 改动要严格限制在当前实验、分析或文档目标内。

## Git 与 Worktree

- 每个实验方向或文档方向使用一个分支。
- 优先使用项目本地 worktree，位置为 `.worktrees/`；该目录必须保持 ignored。
- `main` 是与 GitHub 同步、方便阅读的稳定入口。
- 使用聚焦提交：每个 commit 只包含一个逻辑上的代码、分析或文档变化。
- 需要在 GitHub 上方便阅读的文档，应在分支验证后把紧凑、已审阅记录落到 `main`。Pull request 是合并前的 GitHub 审阅 checkpoint；直接合并是在本地验证后立即写入 `main`。当仍有未解决审阅、代码风险较大或 ownership 不清时，优先走 PR；对用户已授权且本地验证通过的文档整理，可以直接合并。
- 每次有意义改动后都要总结：
  - 改了哪些文件；
  - 为什么改；
  - 验证命令和结果；
  - 如适用，实验命令；
  - 输出目录、manifest 或 scoreboard 条目。

## 双语 Markdown 规则

- Markdown 文档先写英文源版本。
- 英文版本 commit 或上传后，为每个人类可读 Markdown 文件增加中文对应版本。
- 中文文件与英文文件放在同一路径，文件名 stem 后加 `.zh.md`，例如 `docs/results/report.md` 与 `docs/results/report.zh.md`。
- 翻译时保留技术术语、命令、路径、模型名、指标和引用的准确性。
- 中文译本要具有学术准确性和可读性，不能只做松散摘要。若文件是包含大量代码块的可执行计划，中文版本可以通过引用保留代码/命令，同时翻译研究意图、任务结构、预期输出和验证逻辑。

## 实验结果

- 不要把原始 `outputs_clean/`、`logs/`、模型权重或 trace-heavy artifact 提交到普通 git。
- 本地原始输出要保留用于分析；除非用户明确要求，不要删除或覆盖历史运行。
- 提交有价值的紧凑记录：spec、run manifest、scoreboard、分析脚本、CSV/JSON 摘要和解释性笔记。
- 只有被明确选中的高价值 `results.jsonl` 才使用 Git LFS 或外部 artifact 存储。
- 每个报告结果都应标明 baseline、环境、GPU 组、模型、命令、输出目录、总 pass rate、bucket 指标和 wins/losses。

## 硬件基准

- A6000 baseline 实验默认使用 GPU `0,1`，除非用户另行指定。
- 对这类运行设置 `CUDA_VISIBLE_DEVICES=0,1` 和 `TOKENIZERS_PARALLELISM=false`。
- 比较运行时，优先做同硬件对照，再做跨服务器结论。

## 研究主张

- 不要凭记忆声称 SOTA。
- 文献对照必须引用论文、数据集、指标、模型和评测设置。
- 区分真正 apples-to-apples 的直接对比、仅具启发性的非完全可比对比，以及不可比结果。
