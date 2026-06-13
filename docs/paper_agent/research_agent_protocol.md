# Paper-Agent Research Protocol

`AGENTS.md` 是轻量路由器；本文件只在用户明确要求长期研究、overnight、实验监督或继续 CCF-A 论文推进时读取。

## 核心目标

将 `dllm_infilling` 推进为有竞争力的 CCF-A 级别论文。长期研究 agent 的职责不是单纯整理代码，而是持续产生：

- 可验证的实验结果；
- 可复现的实验命令和输出记录；
- 清晰的论文 central claim、风险和下一步计划；
- 可在 GitHub 上阅读的 compact dashboard/checkpoint；
- 阶段化的 skill 输出文档。

## No-Goal Overnight 工作方式

本项目不使用 Codex Goal 工具。overnight 的实现依靠操作流程，而不是 `create_goal`。

### 真实边界

- Codex 只能在当前会话/当前 turn 中持续工作；如果模型结束、进程退出、网络断开或终端关闭，就不会自动开启下一轮。
- `tmux`/`screen` 可以保持 shell 和 Codex CLI 进程不断线，但不能保证模型永不 final。
- 长实验可以在后台继续运行；agent 可以在会话仍活跃时轮询、分析并启动下一步。
- 断线或 final 后，恢复依靠 checkpoint、dashboard、activity ledger 和 resume prompt。

### No-Goal Continuation Discipline

不使用 Goal 时，长期工作依靠当前 turn 内的连续工具调用、后台 shell 进程、checkpoint 和恢复 prompt。普通 CLI/IDE 对话中的 prompt 不能保证模型永不 `task_complete`，因此必须把“继续”写成可执行纪律，而不是只写成愿望。

长期 paper-agent turn 的硬规则：

- 禁止 promise-only stop：不要在“我会……”“接下来……”“下一步……”“我先……然后……”等只承诺下一步的句子后结束当前 turn。
- 任何前瞻性状态更新后，必须立刻执行对应工具调用；例如说“我会检查 review”，下一条动作就必须是记录 reviewer gate disabled、执行 local diff review fallback，或运行明确的 verification command。
- 如果刚完成一个 coherent milestone，必须继续做下一项已明确且安全的动作：focused diff review、Visibility Summary、staged-file check、commit/push 判断、brief brainstorming、experiment brief、启动下一项安全实验，或 checkpoint 更新。
- 如果不能继续，必须把原因分类为：用户确认门、权限/网络/硬件等待、真实 blocker、暂停协议、或没有安全下一步。只说“我会继续”不是 blocker。
- 本项目默认禁用 reviewer/subagent discovery；这不是 session blocker。只降级 review gate，记录 `reviewer_gate_disabled` 和残余风险，然后继续下一项安全且已批准的动作。
- 如果 host 仍然结束了 turn，下一次恢复必须先读 `pause_checkpoint.current.md`、`current_action.md`、dashboard 和 activity ledger，判断上一轮是否停在 promise-only stop；若是，直接执行那句承诺的动作，而不是重新总结。

允许停止的情况只有：

- 用户明确暂停、停止或要求等待确认；
- 下一步会改变研究方向、消耗大量 GPU、删除/覆盖数据、扩大 scope，且需要用户确认；
- 权限、硬件或网络不可用，且已记录 blocker；
- 当前没有安全且已批准的下一步，并且 checkpoint/dashboard 已更新；
- 已启动后台长实验，且已记录 PID/session、log path、output path、轮询方式和下一次恢复 prompt。

明确不允许因为以下原因停止：

- 只是没有 independent reviewer/subagent 工具，或这些路径被本项目禁用；
- 只是完成了 verification、local review 或 commit 判断；
- 只是还可以再做一个 CPU-only diagnostic，但已经存在清晰、安全、计划内的实验下一步。

如果需要真正的定时唤醒，使用 Codex app 的 thread automation 或外部 `tmux`/`cron`/手动 resume prompt。不要把 Codex Goal 当作替代品。

### 推荐启动方式

在远程服务器或不会睡眠的机器上启动：

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
tmux new -s codex-paper
codex --search -C /home/shx/projects/dllm_infilling/git_workspace
```

如果已经有 session：

```bash
tmux attach -t codex-paper
```

### 给 Codex 的 overnight prompt 必须包含

- 不调用 Goal 工具。
- 不要在完成第一个小步骤后 final。
- 长实验用 `tmux`/日志文件运行。
- 实验完成后自动分析、更新 dashboard/checkpoint，并按已批准计划启动下一步。
- 如果无法安全启动下一步，写 checkpoint 并停止。

示例：

```text
这是长期 paper-agent research session。不要调用 create_goal/update_goal/get_goal。
请在当前会话中持续推进，不要在完成第一个小步骤后 final。
不要 promise-only stop：如果你说“我会/接下来/下一步/我先”，必须同一 turn 立刻调用工具执行；不能在只承诺下一步的句子后 task_complete。
先读取 AGENTS.md 和 docs/paper_agent/research_agent_protocol.md，再从 pause_checkpoint.current.md 和 paper_agent_dashboard.zh.md 恢复。
按照已批准计划运行实验、轮询完成、分析结果、更新文档，并在安全且计划明确时启动下一项实验。
长实验必须在 tmux/screen 或明确日志文件中运行；运行前展示 exact command。
不要调用 independent reviewer/subagent discovery；只记录 reviewer_gate_disabled，执行 local diff review + fresh verification fallback，然后继续下一项安全动作。
如果遇到真正 blocker，更新 checkpoint/dashboard，写清 blocker 类型、已尝试内容、下一步恢复入口，并告诉我需要什么。
```

## 实验行动说明规则

在 paper-agent 长期研究会话中，每一次实质性写代码、修改分析逻辑、运行 CPU diagnostic、启动 GPU 实验、或更新会影响论文结论的文档之前，必须先给用户一个可校正的行动说明。行动说明同时写入文档，并在终端/对话中用短格式输出。

默认写入：

- `docs/paper_agent/current_action.md`：当前正在做的最小动作，保持短而可覆盖。
- `docs/paper_agent/experiments/<timestamp>_<slug>.md`：需要长期保留的实验、diagnostic 或重要代码变更记录。

行动说明至少包含：

- action name；
- 当前阶段和对应 plan version；
- reviewer motivation：为什么 CCF-A 审稿人会关心这一步；
- hypothesis 或 maintenance rationale；
- baseline、dataset、model、metric 和 comparison type；
- exact command 或将要修改的文件；
- GPU set、env、log path、output path，如果适用；
- expected outcome；
- success criteria；
- kill criteria；
- known risks：哪些情况会让本次证据不能支持论文结论；
- expected documentation outputs：本次结束后会更新哪些文档。

如果动作只是纯文档维护，也必须写出维护目标、涉及文件、是否影响 central claim、以及验证方式。如果动作紧急到无法先写完整实验说明，至少先在终端输出短说明，并在动作完成后补写 `current_action.md` 或 activity ledger。

行动说明不是用户确认门。当前已批准研究方向内的 smoke、CPU diagnostic、analysis run 或计划内 GPU 实验，只要 hypothesis、baseline、dataset、metric、command、GPU/env/log/output path、expected outcome 和 kill criteria 已写清，且不会中断他人任务或覆盖历史输出，就应直接启动。

必须请求用户确认的情况只有：改变研究方向、启动大规模 GPU/训练资源消耗、删除或覆盖数据、修改 central claim、扩大实验 scope、或需要在多个互斥研究路线中由用户选择。

## 长实验执行模式

### 运行前

启动训练、评测或 benchmark 前，必须写清楚：

- exact command；
- GPU set；
- conda/env；
- output directory；
- log file；
- baseline；
- 预计观察指标；
- 成功条件和 kill criteria。

GPU 命令默认遵守：

```bash
CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false
```

除非用户另行指定。

### 运行中

长实验优先使用 `tmux`/`screen` 或明确日志文件。建议命名：

```text
logs/paper_agent/<timestamp>_<experiment_name>.log
outputs_clean/<experiment_name>_<timestamp>/
```

运行中必须记录：

- tmux session/window 名称，或 shell PID；
- exact command；
- log path；
- output path；
- start time；
- 当前状态。

agent 可以定期轮询日志，但不要把完整日志粘贴进对话或文档。只记录摘要、关键错误、最终指标和可复现路径。

### 运行后

实验完成后按顺序做：

1. 检查 exit status、日志尾部和输出目录。
2. 运行轻量 sanity checks，例如 row count、JSON parse、关键 metric 存在性。
3. 生成 compact analysis 或 scoreboard。
4. 更新 `experiment_results.*.md`、dashboard、checkpoint 和必要的 activity ledger。
5. 对照 current plan 判断下一步是否已经明确。
6. 如果下一步已在计划中且安全，展示 exact command 后继续；否则暂停并请求用户确认。

不要把 CPU-only diagnostic 当作无限前置门。若现有 CPU 证据已经足够说明某条路线不可行，应转向更有信息量的 trace-enabled smoke、baseline/protocol alignment、或计划内 GPU 实验；若计划内 GPU 实验的 brief 已满足安全条件，应启动实验而不是继续追加低价值离线检查。

## Skill 阶段管线

本节只定义项目级 skill routing。真正调用任何 skill 时，必须读取并遵守该 skill 自身的 `SKILL.md`。如果本节与 skill 硬规则冲突，以 skill 硬规则和用户当前指令为准。

Skills 是串行 phase gates，不是循环。每个阶段最多运行一个 broad skill/workflow。阶段结束必须交付可读文档或验证记录；用户确认进入下一阶段前，不要自动回头重跑上一个 broad skill。恢复上下文时优先读 skill 产物，而不是重新运行 workflow。

### No-Subagent Policy

为避免长期实验会话在 reviewer/subagent/tool discovery 附近出现 promise-only stop，本项目默认不使用任何 subagent 路径。

- 禁止自动调用 `superpowers:subagent-driven-development`、`superpowers:dispatching-parallel-agents`、Task/Spawn subagent、parallel-agent、multi-agent reviewer、guardian/reviewer subagent 或 `tool_search` 进行工具发现。
- 执行 implementation plan 时，只允许 `superpowers:executing-plans` 或手动串行执行。即使任务逻辑上可并行，agent 也必须逐项推进，并在每个实验/实现 milestone 后更新 checkpoint、dashboard 或 action brief。
- Code review gate 不依赖 independent reviewer/subagent。默认做 local diff review + fresh verification；记录 `reviewer_gate_disabled` 和残余风险后继续。
- 如果用户在单独 turn 中手动指定某个 Superpowers skill，可以只运行该指定 skill；若该 skill 要求 subagent、Task/Spawn、parallel-agent 或 reviewer discovery，必须停止该路径并切换到本地串行 fallback。

### Skill Architecture

Research Strategy Skills：

- gstack `/office-hours`
- gstack `/plan-ceo-review`
- gstack `/plan-eng-review`

Superpowers Basic Engineering Workflow：

- `superpowers:brainstorming`
- `superpowers:using-git-worktrees`
- `superpowers:writing-plans`
- `superpowers:executing-plans`
- `superpowers:test-driven-development`
- `superpowers:requesting-code-review`
- `superpowers:finishing-a-development-branch`

Cross-Cutting Quality Gates：

- `superpowers:systematic-debugging`
- `superpowers:verification-before-completion`
- `superpowers:requesting-code-review`，如果重要修改尚未经过 review

Optional Coding Style Guard：

- `karpathy-guidelines`，如果已安装

### 研究策略阶段：gstack

#### gstack `/office-hours`

使用时机：

- 研究方向需要重新发现、重构或判断是否值得继续；
- 需要把“用户需求”映射为研究社区、审稿人和 minimum publishable contribution。

输出：

- `research_design.initial.*.md`、`research_design.current.*.md` 或新的 design review 文档。

不要反复重跑。已有 research design 足够时，优先读取产物。

#### gstack `/plan-ceo-review`

使用时机：

- 论文故事、central claim、novelty、scope、reviewer appeal 需要压力测试；
- 当前实验结果可能改变论文 framing。

输出：

- research design 中的 stress review，或单独 review 文档。

#### gstack `/plan-eng-review`

使用时机：

- 实验工程、datasets、baselines、metrics、ablations、compute budget、reproducibility 需要重新设计；
- 计划要从研究想法变成实验路线。

输出：

- `experiment_plan.current.*.md`
- 必要时更新 `experiment_plan.history.*.md`

### 工程实现阶段：Superpowers Basic Workflow

#### `superpowers:brainstorming`

使用时机：

- 新实验族、新方法、新 diagnostic、新 runner 或重要行为变化前；
- spec 还不清晰时。

输出：

- 明确 spec 或 design doc。
- 用户确认前不要进入实现。

长期 paper-agent 的例外：

- 如果用户已经授权 autonomous experimentation，且实验仍在当前 plan version 和 central claim 范围内，可以做 brief brainstorming，而不是完整 broad skill 阶段。
- Brief brainstorming 必须列出 2-3 个候选实验、推荐路线、审稿人动机、风险和为什么现在跑；写入 `current_action.md` 或实验 brief 后即可进入执行。
- 只有当 brief 会改变研究方向、扩大 scope、启动大规模 GPU/训练消耗、或在多个互斥路线中需要用户偏好时，才停下来请求用户确认。

#### `superpowers:using-git-worktrees`

使用时机：

- 开始新实验族；
- 执行 implementation plan 前；
- 预计会修改 runner、analysis pipeline、实验脚本或多个文件；
- 当前 `git_workspace` 已经混乱，继续在原地做会增加风险。

规则：

- 先检测当前是否已经在 isolated workspace。
- 不要无脑创建 worktree。
- 优先使用 `.worktrees/`，并确认 ignored。
- 小文档更新、单个轻量 analysis note 可不建 worktree。

输出：

- worktree path、branch、baseline verification 状态。

#### `superpowers:writing-plans`

使用时机：

- spec 已明确，需要转成可执行任务计划。

输出：

- `docs/superpowers/plans/<date>-<topic>.md`
- 计划必须包含文件、测试、命令、预期结果和 commit 边界。

#### `superpowers:executing-plans`

使用时机：

- 已有 implementation plan，需要执行。
- 任务可拆分时，也必须由当前 agent 按 plan 串行执行。
- 任务高度耦合、需要谨慎控制 GPU 或需要长时间观察实验时，用 `executing-plans` 或手动执行。

subagent 禁用：

- 不使用 `superpowers:subagent-driven-development`。
- 不派发 implementer subagent、reviewer subagent、Task/Spawn 或 parallel-agent。
- 不调用 `tool_search` 去发现 subagent/reviewer 工具。
- 每个 plan task 完成后，当前 agent 自己检查 diff、测试结果、文档更新和未提交文件。

输出：

- 完成的任务、review 结果、测试结果、commits 或未提交 diff 状态。

#### `superpowers:test-driven-development`

使用时机：

- 新 feature；
- bugfix；
- refactor；
- analysis 脚本行为变化；
- runner 或实验逻辑变化。

规则：

- 先写失败测试，再写实现。
- 如果是纯文档、一次性探索脚本或用户明确允许 prototype，可例外，但要记录原因。

输出：

- red/green 验证命令和结果。

### 横向质量门

#### `superpowers:systematic-debugging`

使用时机：

- 测试失败；
- 实验结果与预期矛盾；
- GPU run 异常；
- 指标下降但原因不明；
- 文档结论与 raw evidence 不一致。

规则：

- 不要先改代码再找原因。
- 先复现、定位、提出假设、验证假设，再修复或调整实验方案。

输出：

- root cause、evidence、修复或下一步实验。

#### `superpowers:verification-before-completion`

使用时机：

- 声称代码完成前；
- 声称实验结果有效前；
- commit/push 前；
- 更新 dashboard 的“已完成”前；
- 进入下一实验前。

规则：

- 必须 fresh run verification command。
- 不能用“之前跑过”“应该通过”“看起来没问题”代替。

输出：

- verification commands、关键输出、是否通过。

#### `superpowers:requesting-code-review`

使用时机：

- 重要代码修改；
- 实验 runner 修改；
- analysis pipeline 修改；
- 影响论文结论的文档或指标修改。

规则：

- 不调用 independent reviewer、Task、Spawn、subagent 或 `tool_search` discovery；直接记录 `reviewer_gate_disabled`。
- fallback 是 local diff review + fresh verification，必须说明残余风险。
- reviewer gate disabled 不阻塞 commit 判断、实验启动或下一项计划内工作；fallback 完成后继续下一项安全动作。

输出：

- review findings、修复情况、残余风险。

#### `superpowers:finishing-a-development-branch`

使用时机：

- 一个 implementation branch 或 worktree 阶段完成；
- 准备 merge、PR、清理 worktree 或交付给用户前。

输出：

- merge/PR/cleanup 选择，最终状态和残余风险。

### 可选代码风格护栏：Karpathy skill

如果已安装 `karpathy-guidelines`，非平凡代码修改、runner 修改、analysis 脚本修改和 diff review 时默认作为轻量代码质量护栏。

它不是独立研究阶段，不需要用户单独确认。它与 Superpowers 的关系是：

- Superpowers 决定流程；
- Karpathy 约束代码风格：先说明 assumptions，保持 simple/surgical，不做无关抽象，用测试闭环。

## 当前 paper-agent 恢复入口

恢复时默认读取：

1. `docs/paper_agent/pause_checkpoint.current.md`
2. `docs/paper_agent/paper_agent_dashboard.zh.md`
3. `docs/paper_agent/activity_ledger.zh.md` 最近 20-40 行，如果存在

仅在需要时读取：

- `experiment_plan.current.*.md`：准备执行或修改实验方案。
- `experiment_plan.history.*.md`：解释方案变化或确认没有跑偏。
- `experiment_results.*.md`：解释结果或做结论。
- `evidence_snapshot.md` / audit summaries：需要证据锚点。
- `research_design.current.*.md`：改变 central claim 或论文 framing。
- `overnight_log.*.md`：审计具体历史、checkpoint 缺失、或用户要求完整回顾。

不要把 `overnight_log.*.md` 作为常规恢复入口。

## 文档职责

长期研究文档位于 `docs/paper_agent/`。

关键文档：

- `paper_agent_dashboard.zh.md` / `.en.md`：给用户在 GitHub 上快速阅读的首页。
- `pause_checkpoint.current.md`：恢复入口，必须短而实用。
- `activity_ledger.zh.md` / `.en.md`：短时间线，记录每个实质动作。
- `research_design.initial.*.md`：初始研究设计，原则上冻结。
- `research_design.current.*.md`：当前研究 framing。
- `experiment_plan.initial.*.md`：初始实验方案，原则上冻结。
- `experiment_plan.current.*.md`：当前最新方案。
- `experiment_plan.history.*.md`：每次方案调整及原因。
- `experiment_results.*.md`：实验结果摘要和解释。
- `open_questions.*.md`：当前未决问题。
- audit summaries：具体诊断或分析结果。

## 双语写作规则

- 英文版用于后续 research memo 或论文草稿，要求精确、克制、学术化。
- 中文版用于用户快速阅读，要求准确、专业、术语一致，不是随意摘要。
- 长时间执行中可以先写 compact ledger；milestone 完成后再补齐双语 dashboard/results/plan。
- 不要因为双语维护而阻塞实验监督；但每个 coherent milestone 最终必须有中文可读摘要。
- 技术术语、路径、命令、metric、model name、paper title 不要乱翻译。

## Dashboard 更新规则

`paper_agent_dashboard.zh.md` 必须保持短而清楚，包含：

- 当前研究目标；
- 当前 central claim；
- 当前实验方案版本；
- 最近完成了什么；
- 最新结果摘要；
- 关键方案调整；
- 当前最大风险；
- 下一步计划；
- 需要用户决策的问题。

每个 milestone 后更新 dashboard。不要把 dashboard 写成流水账。

## Checkpoint 更新规则

`pause_checkpoint.current.md` 必须能让新会话快速恢复。至少包含：

- timestamp；
- branch；
- current phase；
- completed items；
- current central claim；
- current experiment plan version；
- latest evidence/results；
- running or just-ended commands；
- modified files；
- known risks；
- open questions；
- next 3 actions；
- recommended resume prompt。

如果 checkpoint 与 `git status` 不一致，恢复者必须先判断 checkpoint 是否过期。

## GitHub 可见性规则

用户需要能够从 GitHub 或终端快速理解实验进展。每个 coherent milestone 结束后，必须输出一个 `Visibility Summary`，并更新对应 compact docs。

`Visibility Summary` 至少包含：

- updated local docs：本次更新的 dashboard、checkpoint、results、plan、audit 或 experiment brief；
- code and test files：本次涉及的代码和测试文件；
- verification commands and result；
- git state：branch、ahead/behind、dirty/untracked summary；
- GitHub visibility verdict：`visible_on_github`、`local_only_not_pushed`、或 `not_committed`；
- GitHub links，如果已经 commit 并 push；
- local paths，如果尚未 push 或无法确认远端可见。

不要声称某个结果“可以从 GitHub 看到”，除非已经确认相关文件被 commit 到当前分支并成功 push。若网络、权限或用户指令导致不能 push，必须明确写：

```text
GitHub visibility: local_only_not_pushed
```

提交和推送规则：

- 不要把 unrelated dirty files 一起 stage。
- 提交前列出 staged files。
- 每个 commit 只包含一个逻辑 milestone，例如一个实验 brief、一个 diagnostic、一个结果更新、或一个 protocol/doc change。
- 如果本轮只完成本地文档更新且用户没有授权 push，可以不 push，但必须给出本地路径和下一步 push/PR 建议。
- 如果用户要求 GitHub 链接优先，则 milestone 完成后优先做 focused commit/push，再输出链接。

## 实验方案版本管理

`experiment_plan.initial.*.md` 是初始方案，创建后不覆盖。

`experiment_plan.current.*.md` 是当前方案。

`experiment_plan.history.*.md` 记录每次修改：

```markdown
### Plan Revision vN -> vN+1

- Timestamp:
- Trigger:
- Previous plan:
- New plan:
- Evidence:
- Reason for change:
- Expected benefit:
- Risk introduced:
- What remains unchanged:
- Kill criteria affected:
- Related commits or files:
```

只有 history 记录完成后，才能更新 current plan。

## 结果报告规则

每个实验结果必须标明：

- baseline；
- environment；
- GPU set；
- model；
- command；
- output directory；
- total pass rate；
- bucket metrics；
- wins/losses；
- same-hardware 或 cross-hardware；
- 与当前 central claim 的关系；
- 是否通过 offline gate 或 GPU gate。

不要凭记忆声称 SOTA。外部比较必须说明是否 apples-to-apples。

## Claim Readiness Gate

任何 SOTA、CCF-A、paper-ready、submission-ready、competitive 或 publishable claim 都必须先通过 Claim Readiness Gate。没有通过 gate 时，必须直接写出 `not_ready`，不能用模糊积极表述替代。

每次 Claim Readiness Gate 必须输出一个 verdict：

- `not_ready`：证据不足，不能作为论文主张。
- `weak_candidate`：有局部有效结果，但缺少关键审稿证据。
- `paper_candidate`：有清晰 central claim、主要实验和负结果边界，但仍缺少投稿前验证。
- `submission_candidate`：已具备 protocol-matched baselines、消融、误差分析、复现材料和清晰论文故事。

Gate 必须逐项检查：

- central claim 是否精确、可证伪、不是事后包装；
- baseline 是否 protocol-matched，包括 dataset、metric、model、prompt/evaluation setting；
- comparison 是否 same-hardware、cross-hardware、apples-to-apples 或 suggestive，并明确标注；
- 是否有当前方法相对最强内部 baseline 的 total pass rate、bucket metrics、wins/losses；
- 是否有 long-tail、short-risk、current-pass-risk 等与本项目 claim 直接相关的风险指标；
- 是否有 ablation 说明哪一部分带来收益；
- 是否有 negative evidence 和 failure modes，而不是只报告正结果；
- 是否有 error analysis，尤其是 true-long failure、length underestimation 和 regression cases；
- 是否有统计稳定性或 deterministic split discipline；
- 是否有可复现 artifact：exact command、env、commit、output directory、raw result preservation policy、analysis script；
- 是否对外部 SOTA 或相关论文进行了最新且可比的 literature/protocol alignment。

输出位置：

- milestone 级判断写入 `experiment_results.*.md`；
- 面向用户的短 verdict 写入 `paper_agent_dashboard.*.md`；
- 如果 verdict 改变 central claim，先更新 `experiment_plan.history.*.md` 或 `research_design.current.*.md`，再更新 dashboard。

如果任一关键项缺失，verdict 最高只能是 `weak_candidate`。如果缺少 protocol-matched external baseline 或无法确认 apples-to-apples comparison，禁止使用 SOTA 表述。

## 阶段推进规则

每个阶段结束时必须回答：

- 这个阶段产出了什么文档？
- 当前证据支持什么？
- 当前证据不支持什么？
- 是否改变 current plan？
- 是否需要用户确认？
- 下一阶段是什么？

如果用户要求“我确认后再进入下一个 skill”，则阶段结束后必须停止，等待用户确认。

如果用户要求“overnight 自动执行已批准计划”，则可以在同一长期研究会话中继续执行已批准计划中的下一项实验或分析，但不要开启新的 broad skill 阶段。

每个阶段结束时若已经写出“下一阶段是什么”，且下一阶段安全、已批准、无需用户确认，就必须在同一 turn 继续执行下一阶段的第一个具体动作。不要把“下一阶段是什么”当作 final answer。

## 暂停协议

用户说“暂停”“优雅暂停”“stop”“hard stop”“不要继续”“先停一下”或类似指令时：

1. 停止新增研究、实验、代码阅读和 skill workflow。
2. 不执行 checkpoint 中的 next actions。
3. 如果有运行中命令，记录 PID/session、command、log path、output dir、状态。
4. 更新 dashboard、checkpoint、activity ledger 和必要日志。
5. 最终回复后停止，等待用户明确恢复。

## 阻塞协议

遇到无法继续的 session blocker 时：

1. 写入 checkpoint/dashboard。
2. 如果可以，commit 并 push 当前可读文档。
3. 告诉用户：
   - 阻塞是什么；
   - 已尝试什么；
   - 需要用户提供什么；
   - 解除后下一步是什么。

不要为了绕开权限或硬件约束而采取破坏性操作。

质量门降级不是 session blocker。尤其是 reviewer/subagent disabled，只记录残余风险并继续；不要把它写成“需要用户解除后才能继续”的 blocker。

## 当前研究状态摘要

截至 2026-06-01 的 paper-agent 状态：

- 当前分支：`paper-agent-overnight`。
- 当前方案：`v3`，probe-curve-first long-length modeling。
- 当前 checkpoint：A6000 `midcons`，same-hardware `+8` wins、`0` losses，主要改善 short/medium。
- true-long buckets 仍未改善。
- single-feature probe-curve thresholds 未通过 offline gate。
- simple strict-split multivariate probe score 也未通过 held-out safety gate。
- full GPU work 需有清晰 experiment brief 和安全 GPU 条件；trace-enabled smoke 或计划内 GPU 实验不应被无限 CPU diagnostic 阻塞。
- 未来 GPU 默认使用 `CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false`，不得中断其他任务。
