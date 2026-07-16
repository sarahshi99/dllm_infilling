# 项目 Agent 规则

## 仓库根目录

本项目的实际 git 仓库根目录是：

```text
/home/shx/projects/dllm_infilling/git_workspace
```

启动 Codex、运行 git 命令、读取项目指令时，默认以该目录为工作目录。上一级 `/home/shx/projects/dllm_infilling` 不是当前应使用的 git root。

## 禁用 Goal 工具

本项目禁止使用 Codex Goal 工具。

- 不要调用 `create_goal`、`update_goal` 或 `get_goal`。
- 用户说“长期目标”“长期研究”“overnight work”“持续推进”“autonomous research”“一直工作到我回来”等，都不等于允许 Goal 工具。
- overnight 或长时间工作必须依靠当前 Codex 会话、`tmux`/`screen`、后台实验日志、checkpoint 文档和恢复 prompt 实现，不依靠 Goal 自动续轮。
- 如果用户讨论 Goal，只用文字解释其代价和替代方案，不要实际调用 Goal 工具。

## 默认上下文纪律

默认少读、精准读、按需扩展。

- 开始时优先读取直接相关文件，不要默认扫描全仓库。
- 优先使用 `rg`、`git status`、目录列表和短片段读取定位问题。
- 不要读取大型 `results.jsonl`、`outputs_clean/`、`logs/` 或整份长日志，除非当前任务确实需要。
- API 成本、缓存命中、单文件解释、局部 debug、局部 code review 等属于短任务；这些任务不要自动进入 paper-agent 长期研究流程。
- 短任务完成后停止，不继续推进论文实验。

## 长期研究会话

当用户明确要求长期研究、overnight、paper-agent、实验监督、继续 CCF-A 论文推进，或持续运行实验时，进入长期研究会话。

进入长期研究会话时：

- 先读取 `docs/paper_agent/research_agent_protocol.md` 一次。
- 然后优先从 checkpoint 和 dashboard 恢复，不要每轮重读 protocol。
- 长期研究中的阶段顺序、skill 使用、subagent 禁用策略、worktree 策略和 overnight 实验监督细节，以 `docs/paper_agent/research_agent_protocol.md` 为准。
- 在当前会话内持续推进：规划、运行实验、轮询结果、分析、写文档、提交进展。
- 不要因为完成一个小步骤就 final；只有真正阻塞、用户要求暂停、或当前会话没有可安全推进的下一步时才停止。
- 本项目默认不使用 subagent、parallel-agent、Task/Spawn/reviewer discovery 或 multi-agent reviewer；review gate 统一降级为 local diff review + fresh verification fallback，然后继续下一项安全且已批准的研究动作。
- 若 Codex 会话中断或模型结束，依靠 `pause_checkpoint.current.md`、dashboard、activity ledger 和 resume prompt 恢复。

### No Promise-Only Stop

长期研究会话中，禁止以只承诺下一步的文字结束当前 turn。

- 如果说“我会……”“接下来……”“下一步……”“我先……然后……”或类似前瞻性表述，必须在同一 turn 里立刻调用工具执行该动作。
- 不要在“我会检查 reviewer”“我会继续运行验证”“我会进入下一步实验”等句子后停止。
- 如果不能立刻执行下一步，必须说明真实原因，并把 blocker、已完成内容、下一步恢复入口写入 checkpoint/dashboard；不要把计划性句子当作收尾。
- commentary/status update 之后，如果还有安全且已明确的下一步，必须继续调用工具；只有暂停协议、真实 blocker、用户确认门、权限等待或没有安全下一步时才可以停止。
- 不要反复“检查”Task/Spawn/reviewer/subagent 可用性；本项目默认不走这些路径。直接记录 `reviewer_gate_disabled`，做本地 review/verification fallback，并继续执行下一项安全动作。

## Skill 串行阶段规则

gstack 和 Superpowers 是阶段工具，不是循环工具。

- 每个阶段最多运行一个 broad skill/workflow。
- 一个 skill 阶段完成后，必须产出用户可读的文档或报告。
- 用户确认进入下一阶段前，不要自动回头重跑上一个 broad skill。
- 已完成的 `/office-hours`、`/plan-ceo-review`、`/plan-eng-review`，优先读取其产物，不要重新运行。
- 恢复上下文时只读 compact 输出，例如 dashboard、checkpoint、current plan、workflow status，不重新加载完整 skill 本体。
- 只有用户明确要求、阶段产物不存在，或有新证据证明旧产物失效时，才重新运行相同 skill。
- 调用 broad skill 前，先说明为什么这次需要调用；调用后记录输出文件。
- 具体长期研究 skill 管线见 `docs/paper_agent/research_agent_protocol.md` 的 `Skill 阶段管线`；进入长期研究会话后读取一次即可。
- 代码/实验实现阶段默认遵循 Superpowers basic workflow：`brainstorming` -> `using-git-worktrees` -> `writing-plans` -> `executing-plans` -> `test-driven-development` -> `requesting-code-review` -> `finishing-a-development-branch`。
- 禁止自动使用 `superpowers:subagent-driven-development`、`superpowers:dispatching-parallel-agents`、Task/Spawn subagent、parallel-agent 或 reviewer subagent。即使 Superpowers 插件启用，执行 implementation plan 也必须串行。
- `systematic-debugging` 和 `verification-before-completion` 是横向质量门，不因 basic workflow 已列出其他 skill 而省略。
- 如果已安装 `karpathy-guidelines`，非平凡代码修改和 diff review 默认使用它作为轻量代码质量护栏；细节见 protocol，不在 `AGENTS.md` 展开。
- 对当前已批准研究方向内的新实验，先做简短 brainstorming / experiment brief，说明候选路线、推荐选择和风险；若不改变研究方向、不覆盖历史输出、不抢占 GPU，就不把 brief 当作用户确认门，而是写入文档后直接启动实验。

## Paper-Agent 恢复入口

恢复长期研究时，默认先读：

1. `docs/paper_agent/pause_checkpoint.current.md`
2. `docs/paper_agent/paper_agent_dashboard.zh.md`
3. `docs/paper_agent/activity_ledger.zh.md` 最近 20-40 行，如果存在

只有在以下情况继续读取更多文档：

- 要修改实验方案：读取 `experiment_plan.current.*.md` 和 `experiment_plan.history.*.md`。
- 要解释或声明实验结论：读取 `experiment_results.*.md`、`evidence_snapshot.md` 或相关 audit summary。
- 要改变论文 framing 或 central claim：读取 `research_design.current.*.md`。
- checkpoint 与 `git status` 不一致：读取相关 dirty/untracked 文件和最近日志片段，先判断 checkpoint 是否过期。

## 实验与服务器安全

- 不要运行 destructive commands，例如 `rm -rf`、`mkfs`、`dd`，除非用户明确批准。
- 长实验、训练、benchmark 优先在 `tmux`/`screen` 中运行，并把日志写到 `logs/` 或实验专属输出目录。
- 运行训练、评测或 benchmark 前，先展示 exact command。
- 不要 kill、抢占或中断已有 GPU 任务，除非用户明确批准。
- 当前约定的未来 GPU 实验默认使用 `CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false`，除非用户另行指定。
- 不要用无限 CPU diagnostic 代替必要实验推进；如果实验设计、baseline、metric、命令、日志、输出目录和 kill criteria 已写清，且 GPU 使用安全，就应启动 smoke 或计划内实验。
- 原始 `outputs_clean/`、`logs/`、模型权重和 trace-heavy artifacts 不提交到普通 git。
- 本地原始输出要保留用于分析；除非用户明确要求，不要删除或覆盖历史运行。

## Workspace hygiene 与 coding-agent containment

- workspace hygiene / coding-agent safety audit 默认只做清单、元数据和只读验证；没有明确授权时，不得创建、删除、替换、复用或 prune worktree。
- 不得把工具工作目录切换到已知活动 execution worktree，也不得编辑其文件。对活动作业只允许读取 tmux、进程、cwd、输出路径和行数等运行元数据；不得发送 signal、tmux 输入或读取部分性能结果。
- cleanup proposal 必须先保存在 pending manifest 中。每个 approved 与 executed 字段必须为 JSON boolean false，且 files_deleted、files_moved、files_truncated 必须保持为 0，直到用户对具体路径和动作明确授权。
- 只读审计脚本不得提供写入或 apply 选项；需要实际 cleanup 时，必须由单独、经用户授权的命令和新的审计记录执行。

## Git 与文档安全

- 不要删除、覆盖、回滚用户已有工作，除非用户明确要求。
- 不要使用 destructive git commands，例如 `git reset --hard`、`git checkout -- <path>`、force push，除非用户明确要求。
- 可行时，每个实验方向或文档方向使用独立分支或 worktree。
- 使用 focused commits：每个 commit 只包含一个逻辑上的代码、分析或文档变化。
- 提交前说明 staged files；不要把无关 dirty files 一起 stage。
- 需要在 GitHub 上方便阅读的研究进展，应维护 compact dashboard/checkpoint，而不是只写长日志。
- 编辑文件时使用当前 Codex 宿主推荐的安全编辑方式；如果某个编辑工具失败，不要为普通 workspace edit 盲目升级权限，先使用宿主允许的安全 fallback。

## 研究主张与结果报告

- 不要凭记忆声称 SOTA。
- 文献或外部比较必须标明 paper、dataset、metric、model、evaluation setting。
- 区分 apples-to-apples comparison、suggestive comparison 和不可比结果。
- 每个报告出来的实验结果必须至少标明 baseline、environment、GPU set、model、command、output directory、total pass rate、bucket metrics、wins/losses，以及 same-hardware 或 cross-hardware。
- 推测必须标注为 assumption 或 hypothesis；有证据的判断必须写明 evidence。

## 暂停与阻塞

用户说“暂停”“优雅暂停”“stop”“hard stop”“不要继续”“先停一下”或类似指令时，立即执行暂停协议。

- 停止新增研究、实验、代码阅读和 skill workflow。
- 不要执行 checkpoint 中的 next actions。
- 只允许完成最小收尾：记录运行中命令、更新 dashboard/checkpoint、保存必要日志、查看 `git status`。
- 最终回复后停止，直到用户明确恢复。

如果无法继续：

1. 把 blocker 写入 checkpoint 或 dashboard。
2. 如果可以，commit 并 push 当前可读文档。
3. 告诉用户阻塞是什么、已尝试什么、需要用户提供什么、解除后下一步是什么。
