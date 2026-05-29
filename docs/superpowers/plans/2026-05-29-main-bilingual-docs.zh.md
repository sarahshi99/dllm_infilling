# Main 双语文档实施计划

> **给 agentic worker：** 执行本计划时使用 `superpowers:executing-plans`。步骤用 checkbox 跟踪。

**目标：** 将已验证的 A6000/result-archive 文档落到 `main`，并为当前所有英文 Markdown 文件增加中文对应版本。

**架构：** 将实验分支和归档分支合并到 `main`；英文 Markdown 保持为源记录；在同一路径新增 `.zh.md` 中文译本；更新 `AGENTS.md`，使未来 Markdown 工作遵循“英文先写、中文后译”的策略。

**技术栈：** Git branches/worktrees、Markdown、本地 unit tests、`git diff --check`、推送 GitHub `main`。

---

## Task 1：合并已验证文档分支

**文件：**
- 合并分支：`exp/a6000-midcons-longrescue`
- 合并分支：`docs/result-archive`

- [x] Fetch 远端 `main`。
- [x] 确认本地 `main` 已最新。
- [x] 将 `exp/a6000-midcons-longrescue` 合并进 `main`。
- [x] 将 `docs/result-archive` 合并进 `main`。

## Task 2：增加双语规则

**文件：**
- 修改：`AGENTS.md`
- 新建：`AGENTS.zh.md`

- [x] 解释 PR 与直接合并的行为差异。
- [x] 增加未来 Markdown 规则：先写英文，再增加 `.zh.md` 中文版本。
- [x] 翻译时保留技术路径、命令、模型名、指标和引用。

## Task 3：增加中文对应版本

**文件：**
- 为每个当前英文 Markdown 文件在同一路径新建 `.zh.md`。

- [x] 翻译项目规则和 setup 文档。
- [x] 翻译 scoreboards 和结果报告。
- [x] 翻译 result registries 和 archive notes。
- [x] 翻译 Superpowers specs/plans；中文版本面向阅读，同时保留可复现所需命令和决策。

## Task 4：验证并推送

**文件：**
- 所有 Markdown 文件。

- [x] 验证每个英文 `.md` 都有同路径 `.zh.md`。
- [x] 运行 unit tests。
- [x] 运行 `git diff --check`。
- [ ] Commit 并 push `main`。
