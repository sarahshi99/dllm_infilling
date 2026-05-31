# Overnight Log

## Entry 2026-05-31 12:36 CST

- timestamp：2026-05-31 12:36 CST
- current phase：初始化与上下文恢复
- what was done：阅读项目 agent rules，启动长期 autonomous research-agent workflow，并创建 `paper-agent-overnight` 分支。
- evidence or files inspected：`AGENTS.md`、`git status --short --branch`、`git branch --list --all`。
- decision made：在 `paper-agent-overnight` 上工作；将已有 `AGENTS.md` 和 `AGENTS.zh.md` 变更视为用户工作，不 stage 或 revert。
- uncertainty/risk：创建分支时 worktree 已有无关变更；后续 commit 必须只 stage paper-agent files。
- next action：在撰写 paper-agent documents 前，回顾已有 research plans 和 result reports。

## Entry 2026-05-31 12:37 CST

- timestamp：2026-05-31 12:37 CST
- current phase：证据回顾
- what was done：回顾 prior CCF-A roadmap、run registry、A6000 reports、long-underestimate diagnostics、literature notes 和 model-generalization registry。
- evidence or files inspected：`docs/superpowers/plans/2026-05-29-ccfa-roadmap-and-next-experiments.md`、`docs/results/run_registry.md`、`docs/results/a6000_midcons_longrescue_report.md`、`docs/results/long_underestimate_detector_report.md`、`docs/results/literature_sota_notes.md`、`docs/results/model_generalization_registry.md`、`analysis_outputs/experiment_scoreboard.md`。
- decision made：使用 `midcons` 作为当前 A6000 checkpoint；除非新 diagnostics 推翻当前证据，否则不再运行另一个 full official-CAL true-long heuristic run。
- uncertainty/risk：在 prompt/canvas/evaluation settings matched 前，literature comparison 仍只是 suggestive。
- next action：初始化双语 paper-agent design、experiment plan、result、dashboard 和 open-question documents。

## Entry 2026-05-31 12:38 CST

- timestamp：2026-05-31 12:38 CST
- current phase：硬件安全检查
- what was done：在启动任何实验前检查 GPU occupancy。
- evidence or files inspected：`nvidia-smi` 输出显示 GPU 0,1,2,3 上均有 Python processes，每张卡约占用 9.7-10.1 GiB memory，且 utilization 较高。
- decision made：当前不启动 heavy GPU jobs。继续 CPU-only analysis 和文档准备。
- uncertainty/risk：GPU 之后可能可用；任何 queued run 都必须避免中断现有任务。
- next action：创建必需的 `docs/paper_agent/` files。

## Entry 2026-05-31 12:39 CST

- timestamp：2026-05-31 12:39 CST
- current phase：paper-agent document initialization
- what was done：创建双语 dashboard、research design、experiment plan、experiment plan history、experiment results、open questions 和本 log。
- evidence or files inspected：基于 prior result reports 和已检查代码创建 `docs/paper_agent/*.md`。
- decision made：将 research design 和 experiment plan 设为 `v1`；创建时保持 `initial` 与 `current` 同步，并在 history 中记录 initial plan revision。
- uncertainty/risk：第一版文档基于已有 reports；下一步应从 raw outputs 独立验证核心 metrics。
- next action：从已有 `results.jsonl` files 构建 compact evidence snapshot。

## Entry 2026-05-31 12:41 CST

- timestamp：2026-05-31 12:41 CST
- current phase：可复现 evidence snapshot
- what was done：新增一个经过测试的 CPU-only evidence snapshot builder，并从已有 raw outputs 生成 `docs/paper_agent/evidence_snapshot.json` 与 `docs/paper_agent/evidence_snapshot.md`。
- evidence or files inspected：`analysis/build_paper_agent_evidence_snapshot.py`、`tests/test_build_paper_agent_evidence_snapshot.py`、A6000 control 和 candidate `results.jsonl` files，以及生成的 snapshot files。
- decision made：将重新计算得到的 snapshot 作为当前 milestone 的 paper-agent evidence anchor。
- uncertainty/risk：Snapshot 依赖本地 raw outputs；如果这些 outputs 被移动，未来运行必须使用明确的 `--run NAME=PATH` 参数。
- next action：验证文档集合、运行测试，并准备一个 focused commit，排除无关 `AGENTS.*` changes。
