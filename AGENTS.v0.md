# 项目 Agent 规则

本项目的实际 git 仓库根目录是：

```text
/home/shx/projects/dllm_infilling/git_workspace
```

启动 Codex、运行 git 命令、读取项目指令时，默认应以该目录为工作目录。上一级 `/home/shx/projects/dllm_infilling` 不是当前应使用的 git root。

本项目的长期目标是将 `dllm_infilling` 推进为一篇有竞争力的 CCF-A 级别研究论文。agent 的目标不是只整理代码，而是持续产生可验证的研究进展、实验计划、实验结果、论文判断和可复现记录。

## 协作默认规则

- 优先推进具体研究结果，而不是只输出高层建议。
- 当证据显示用户假设可能有问题时，必须直接指出，并给出更强替代方案及其利弊。
- 高影响操作前，必须先对齐目标、约束、风险和成功标准。
- 不要在仍有有用工作可做时静默停止。继续阅读、分析、规划、编码、验证或记录，直到遇到真实用户决策或外部阻塞。
- 不要删除、覆盖、回滚用户已有工作，除非用户明确要求。
- 不要使用 destructive git commands，例如 `git reset --hard`、强制覆盖、批量删除历史结果等，除非用户明确要求。

## 上下文预算规则

普通短任务不要默认启用长期研究工作流。普通短任务包括：小型 coding/debugging 问题、API 成本问题、单个文件解释、单个命令问题、局部代码审查、或用户没有要求 paper-agent/overnight/长期 autonomous research 的任务。

普通短任务中：

- 不要扫描全仓库，除非用户明确要求或局部证据显示必须扩展范围。
- 不要读取 `docs/paper_agent/`、`outputs_clean/`、`logs/` 或大型 `results.jsonl`，除非它们与当前问题直接相关。
- 开始时最多读取 2-5 个直接相关文件。
- 扩大范围前，先说明为什么要扩大、准备检查哪些文件或目录。
- 优先使用 targeted `rg` 查询和短片段读取，不要 dump 整个文件。
- 只有用户明确要求时，才使用 gstack review/qa/health/ship/cso 这类 broad skills。
- 如果用户要求长期研究、过夜工作、paper-agent 工作，或说“持续工作直到我回来”，则改用下面的长期研究规则。

## 长期研究 Agent 规则

当用户要求论文规划、过夜工作、长期研究推进、autonomous research agent、或“持续工作直到我回来查看结果”时，遵守本节。

- 优先自主推进，而不是频繁提问。
- 只有在以下情况才停下来问用户：
  - 某个选择会实质改变研究方向。
  - 需要用户提供凭证、账号、API key、远程权限或硬件权限。
  - 下一步可能删除、覆盖或破坏数据。
  - 同一个阻塞已经连续出现三次。
  - 缺少的信息会导致后续计划大概率错误。
- 被阻塞时，必须先写入英文和中文日志；如果可以，commit 并 push 当前文档；然后说明阻塞是什么、已经尝试了什么、需要用户决定什么、解除阻塞后下一步是什么。
- 不要只告诉用户“应该怎么做”。要实际执行阅读、分析、规划、文档整理、验证和同步。
- 如果长时间运行，至少每 60-90 分钟主动做一次轻量 plan alignment check，检查当前工作是否仍然服务于 CCF-A 论文目标。这个检查不是全局扫描，也不是完整文档重读；默认只看当前 checkpoint/dashboard/current plan 的必要短片段。

## 低 Token 持续工作模式

当用户提到“低 token 模式”、“low-token mode”、“缓存命中低”、“token 消耗太高”、“节省上下文”或类似要求时，立即启用本节。低 token 模式只改变上下文读取、日志记录和 skill 调用策略；它不意味着停止工作，也不削弱长期 autonomous research agent 的持续推进目标。

低 token 模式的核心原则：

- 继续持续工作，但每一步只读取完成当前动作所需的最小证据。
- 优先读取 compact checkpoint 和 dashboard，不要反复读取完整日志、完整 history 或大段源代码。
- 先用 `rg`、`git status`、文件目录和短片段定位，再读取精确文件片段。
- 不要把长命令输出、完整 JSON、完整 results、完整日志粘贴进文档或回复；只记录摘要、路径和可复现命令。
- 每个阶段最多保留一个 compact evidence anchor，例如 `evidence_snapshot.md`、`pause_checkpoint.current.md` 或某个 audit summary。
- 除非用户明确要求，不要为了“确认上下文”而重新扫描全仓库。

低 token 模式下的默认启动读取顺序：

1. `AGENTS.md`
2. `git status --short --branch`
3. `docs/paper_agent/pause_checkpoint.current.md`，如果存在
4. `docs/paper_agent/paper_agent_dashboard.zh.md`
5. `docs/paper_agent/evidence_snapshot.md`
6. `docs/paper_agent/experiment_plan.current.en.md`
7. `docs/paper_agent/open_questions.en.md`
8. 只有当实验方案发生变化时，才读取 `docs/paper_agent/experiment_plan.history.en.md`

如果 `git status` 显示的 dirty/untracked files 不在 `pause_checkpoint.current.md` 的 captured status 中，说明 checkpoint 可能已经过期。此时必须先检查相关 dirty/untracked 文件名、最近的 compact activity ledger 或 `overnight_log.*.md` 末尾 40-120 行，再决定从哪里恢复。不要盲目执行 checkpoint 中的旧 next action。

如果 `activity_ledger.en.md` / `activity_ledger.zh.md` 不存在，低 token 模式下第一次记录进展时创建它们。缺少 activity ledger 本身不是阻塞。

禁止把 `overnight_log.*.md` 作为常规恢复入口。只有在真正长间隔恢复、需要审计某个具体时间点、查找缺失证据或用户明确要求时，才读取相关片段。

低 token 模式下仍然必须持续推进：

- 如果当前计划清楚，直接执行下一步。
- 如果缺少局部证据，只读取相关文件片段或运行轻量检查。
- 如果需要修改代码，按当前 plan 的下一项任务推进。
- 如果需要记录进展，先更新 compact activity ledger；milestone 完成后再更新 dashboard、checkpoint、current plan 或结果摘要，而不是每个小动作都扩写完整叙事日志。

## Skill 调用节制规则

gstack 和 Superpowers 是阶段工具，不是每轮都要重新运行的上下文加载器。

- 已完成的 `/office-hours`、`/plan-ceo-review`、`/plan-eng-review` 不要反复重跑；优先读取它们产出的 `research_design.current.*.md` 和 `experiment_plan.current.*.md`。
- 只有当研究目标、central claim、实验路线或工程架构发生实质变化时，才重新运行相应 gstack skill。
- Superpowers `brainstorming` 只在需要重新定义 spec 或做新的创造性方向选择时运行。
- Superpowers `writing-plans` 只在已有 spec 需要变成可执行任务计划时运行；已有计划未过期时，继续执行计划而不是重新写计划。
- `systematic-debugging`、`verification-before-completion` 和 `requesting-code-review` 按需使用，但要尽量引用已有 evidence anchor，而不是重新加载全项目上下文。

## 暂停协议

用户说“暂停”、“优雅暂停”、“stop”、“hard stop”、“不要继续”、“先停一下”或类似指令时，暂停协议优先级高于长期 autonomous research agent 规则。

暂停协议：

- 立即停止新增研究、实验、代码阅读和 skill workflow。
- 不要执行 checkpoint 中的 next actions。
- 不要执行自己写入文档的 resume prompt。
- 如果有正在运行的命令，记录 PID、命令、输出目录和状态；除非有数据破坏风险，不要随意 kill。
- 只允许完成最小收尾：更新 dashboard、pause checkpoint、evidence snapshot、必要日志和 `git status`。
- 最终回复后停止，不再调用工具，直到用户明确恢复。

pause checkpoint 必须明确写入：

- 当前是否处于 paused state。
- 是否还有运行中命令。
- 下次恢复时应该读哪些 compact files。
- 下次恢复时最应该做的 3 件事。
- 推荐 resume prompt，但只能写入文档，不能由当前 agent 自动执行。

## 推荐长期工作流

优先使用 gstack 负责研究方向、论文故事、scope、strategy、engineering review；使用 Superpowers 负责 spec、implementation plan、debugging、verification、code review discipline。

1. `gstack /office-hours`
   - 使用 Builder mode。
   - 适配为 CCF-A 论文规划。
   - 将“用户”理解为研究社区、审稿人、相关领域读者和 baseline 论文作者。
   - 将“需求”理解为领域中尚未解决、但审稿人会认可其重要性的问题。
   - 将“最小产品”理解为 minimum publishable contribution。
   - 输出 research design doc。

2. `gstack /plan-ceo-review`
   - 压力测试论文故事。
   - 检查 novelty、importance、reviewer appeal、scope、central claim、weak assumptions。
   - 判断当前方向是否现实地有机会成为 CCF-A 论文。

3. `gstack /plan-eng-review`
   - 将研究方向落成实验工程计划。
   - 覆盖 datasets、baselines、metrics、ablations、compute budget、reproducibility、expected failure modes、kill criteria 和实验优先级。

4. Superpowers
   - `superpowers:brainstorming` 用于明确 spec。
   - `superpowers:writing-plans` 用于生成可执行计划。
   - `superpowers:systematic-debugging` 用于调查 bug、实验异常和不符合预期的结果。
   - `superpowers:verification-before-completion` 用于完成前验证。
   - `superpowers:requesting-code-review` 用于重要代码修改、实验流程修改或论文结论修改后的审查。

## 强制上下文回顾规则

长期研究任务中，agent 必须在以下时机回顾已有文档，不能只依赖上下文记忆：

- 会话开始或 resume 后。
- 每次上下文压缩、长时间运行后重新继续时。
- 开始新的实验前。
- 修改实验方案前。
- 解释实验结果前。
- 声称某个结论成立前。
- commit 或 push 前。
- 遇到实验结果与预期不一致时。
- 每完成一个 coherent milestone 后。
- 至少每 60-90 分钟主动做一次轻量 plan alignment check。

默认回顾使用低 token 分层读取。常规回顾时，优先阅读：

- `docs/paper_agent/pause_checkpoint.current.md`，如果存在
- `docs/paper_agent/paper_agent_dashboard.zh.md`
- `docs/paper_agent/experiment_plan.current.en.md`
- `docs/paper_agent/open_questions.en.md`

仅在以下情况读取 `research_design.current.en.md`：

- central claim、研究目标或论文 framing 可能变化。
- 需要判断是否仍然服务于 CCF-A 目标。
- 用户明确要求审视研究设计。

仅在以下情况读取 `experiment_plan.history.en.md`：

- 准备修改实验方案。
- 需要解释某次方案调整的原因。
- 需要确认没有偏离初始方向。

仅在以下情况读取 `overnight_log.*.md`：

- 距离上次有效 checkpoint 或人工查看已经超过约 10 小时，且 dashboard/checkpoint/current plan/activity ledger 不足以恢复上下文。
- 需要审计某个具体时间点。
- 需要查找 dashboard/checkpoint 中没有保留的证据。
- 用户明确要求回顾完整过程。

60-90 分钟的轻量 plan alignment check 应该保持很短，通常只需要确认：

- 当前执行的任务是否仍对应 `experiment_plan.current.*.md` 的下一步。
- 是否出现需要更新 `experiment_plan.history.*.md` 的方案变化。
- 是否出现需要用户决策或暂停的 blocker。
- 是否需要更新 dashboard 或 checkpoint。

轻量 plan alignment check 不应读取完整 `overnight_log.*.md`，不应扫描 `outputs_clean/`，不应重跑已完成的 gstack/Superpowers workflow。

## 长周期完整校准规则

低 token 模式不是永远只看最少上下文。长期 autonomous research agent 必须按时间做分层校准：

1. 每 60-90 分钟：轻量 plan alignment check。
   - 只读 checkpoint、dashboard、current plan、open questions 的必要片段。
   - 目标是确认当前动作没有跑偏。

2. 每 8-12 小时，或真正进入 overnight/次日恢复时：完整 compact-doc refresh。
   - 完整读取 `paper_agent_dashboard.*.md`、`pause_checkpoint.current.md`、`experiment_plan.current.*.md`、`open_questions.*.md`。
   - 完整读取 `research_design.current.*.md`，确认 central claim 和 CCF-A framing 没有漂移。
   - 如果自上次校准后实验方案变化过，完整读取 `experiment_plan.history.*.md`。
   - 不扫描全仓库，不读取 raw `outputs_clean/`，不读取大型 `results.jsonl`，除非当前校准发现证据缺口。

3. 每 18-24 小时、真正 overnight 后、或 dashboard/checkpoint/activity ledger 互相矛盾时：日志级 refresh。
   - 优先读取 `activity_ledger.*.md` 全文。
   - 如果 `overnight_log.*.md` 仍然不大，可以读取全文。
   - 如果 `overnight_log.*.md` 已经很大，先读取标题、最近 100-200 行和相关时间段片段；然后更新或创建 compact checkpoint，避免下次继续读完整日志。

完整校准的目标是防止方向漂移，不是重新做全部研究。完整校准后必须写一个短的 `Context Calibration` 条目到 activity ledger，说明读了哪些文档、发现了什么偏差、是否调整计划。

如果中文版比英文版更新，也必须读取对应 `.zh.md` 文件并同步修正英文版。同步时只修正必要段落，不要重写整份文档。

## 文档结构

长期研究推进必须维护以下双语文档。英文版是学术原文，中文版是准确、完整、术语一致的学术翻译；中文版不能只是摘要。

```text
docs/paper_agent/
  paper_agent_dashboard.en.md
  paper_agent_dashboard.zh.md

  overnight_log.en.md
  overnight_log.zh.md
  activity_ledger.en.md
  activity_ledger.zh.md

  research_design.initial.en.md
  research_design.initial.zh.md
  research_design.current.en.md
  research_design.current.zh.md

  experiment_plan.initial.en.md
  experiment_plan.initial.zh.md
  experiment_plan.current.en.md
  experiment_plan.current.zh.md
  experiment_plan.history.en.md
  experiment_plan.history.zh.md

  experiment_results.en.md
  experiment_results.zh.md
  open_questions.en.md
  open_questions.zh.md
```

低 token 模式关闭时，每一次实质性阶段性判断、实验分析、计划更新、命令结果、风险判断、阻塞说明，都必须追加到 `overnight_log.en.md` 和 `overnight_log.zh.md`。

低 token 模式开启时，记录分两层：

1. Compact activity ledger：每个实质性动作都要追加一条短记录到 `activity_ledger.en.md` 和 `activity_ledger.zh.md`。
   - 每条记录最多 3-5 行。
   - 记录时间、动作、证据路径、结果摘要和下一步。
   - 不粘贴长输出，不复制完整 JSON，不展开完整推理。

2. Bilingual milestone log：只有在以下情况才追加到 `overnight_log.en.md` 和 `overnight_log.zh.md`：
   - coherent milestone 完成。
   - 实验方案改变。
   - 重要验证完成。
   - 阻塞出现。
   - 用户明确要求详细日志。

这样必须保证事实不丢：activity ledger 保留完整时间线，overnight log 保留可阅读的阶段叙事。

每条日志至少包含：

- `timestamp`
- `current phase`
- `what was done`
- `evidence or files inspected`
- `decision made`
- `uncertainty/risk`
- `next action`

## GitHub 阅读 Dashboard

为了方便用户早上在 GitHub 上阅读进展，必须维护：

- `docs/paper_agent/paper_agent_dashboard.zh.md`
- `docs/paper_agent/paper_agent_dashboard.en.md`

dashboard 不是流水账，必须保持短而清楚，包含：

- 当前研究目标
- 当前 central claim
- 当前实验方案版本
- 昨晚完成了什么
- 最新实验结果摘要
- 方案发生过哪些关键调整
- 当前最大风险
- 下一步计划
- 需要用户决策的问题

每次 milestone 后，先更新详细日志和计划，再更新 dashboard。

## 实验方案版本规则

实验方案必须版本化管理。

- `experiment_plan.initial.*.md` 是初始实验方案，一旦创建后不得覆盖，只能修正错别字或格式。
- `experiment_plan.current.*.md` 始终代表当前最新方案。
- `experiment_plan.history.*.md` 必须记录每一次实验方案调整。
- 只有当调整被记录到 history 后，才能更新 current plan。

任何实验方案调整都必须写入 history，格式如下：

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

## 写作规则

- 英文写作要精确、克制、学术化，后续应能直接复用到 research memo、paper planning note 或论文草稿。
- 中文翻译要忠实、准确、完整，不改变技术含义。
- 不要为了中文流畅而改变技术含义。
- 关键术语保持一致，例如 central claim、novelty、baseline、ablation、metric、kill criteria、reproducibility。
- 推测必须标记为 assumption 或 hypothesis。
- 有依据的判断必须写明 evidence，例如文件、实验、日志、命令输出或代码位置。
- 不要声称 SOTA，除非有明确论文、数据集、metric、model、evaluation setting 和可比结果。
- 文献比较必须区分 apples-to-apples comparisons 和 suggestive or non-comparable comparisons。

## 项目结构

- 将 `expvision_dllm/` 和 `scripts/` 视为冻结的 legacy reference code。
- 新实验优先放在 `expvision_dllm_clean/` 和 `clean_scripts/`。
- 新实验族应新增模块或 runner，不要重新打开大型 legacy 文件。
- 改动必须严格限制在当前实验、分析或文档目标内。
- 不要把原始 `outputs_clean/`、`logs/`、模型权重或 trace-heavy artifacts 提交到普通 git。
- 本地原始输出要保留用于分析；除非用户明确要求，不要删除或覆盖历史运行。
- 提交紧凑且有价值的记录：specs、run manifests、scoreboards、分析脚本、CSV/JSON 摘要和解释性笔记。
- 只有被明确选中的高价值 `results.jsonl` 文件才使用 Git LFS 或外部 artifact store 保存原文。

## 实验结果报告规则

每个报告出来的实验结果都必须标明：

- baseline
- environment
- GPU set
- model
- command
- output directory
- total pass rate
- bucket metrics
- wins/losses
- 该比较是 same-hardware 还是 cross-hardware

A6000 baseline 实验默认使用 GPU `0,1`，除非用户另行指定。

使用：

```bash
CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=false
```

比较不同运行时，优先做 same-hardware comparison，再做 cross-server claim。

## Git 规则

- 长期研究任务优先使用分支 `paper-agent-overnight`。
- 如果当前已经在合适的分支上，可以继续使用，但必须说明原因。
- 可行时，每个实验方向或文档方向使用一个分支。
- 优先使用项目本地 worktree，位置为 `.worktrees/`；该目录必须保持 ignored。
- 保持 `main` 作为与 GitHub 同步的稳定入口。
- 使用 focused commits：每个 commit 只包含一个逻辑上的代码、分析或文档变化。
- 定期 commit 文档更新。
- 如果 GitHub remote 已配置且 push 安全可用，完成 coherent milestone 后 push 到远程。
- 需要在 GitHub 上方便阅读的文档，应在分支验证后把紧凑、已审阅记录落到 `main`。
- Pull request 是合并前的 GitHub review checkpoint；direct merge 会把验证后的分支立即写入 `main`。
- 当仍有未解决审阅、较大代码风险或 ownership 不清时，优先走更安全的 PR 路径。
- 对用户已授权且本地验证通过的文档整理，可以 direct merge。

每次有意义改动后，总结：

- 改了哪些文件
- 为什么改
- 验证命令和结果
- 如适用，实验命令
- 输出目录、manifest 或 scoreboard 条目

## 阻塞规则

如果无法继续：

1. 把 blocker 写入英文和中文日志。
2. 如果可以，commit 并 push 当前文档。
3. 告诉用户：
   - 阻塞是什么
   - 已经尝试了什么
   - 需要用户提供什么
   - 阻塞解除后的下一步
