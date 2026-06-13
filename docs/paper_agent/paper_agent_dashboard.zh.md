# Paper Agent Dashboard

更新时间：2026-06-12 19:26 CST

## 当前研究目标

将当前 DLLM 代码 infilling 项目推进为有竞争力的 CCF-A 论文：把已有 LCAL/LCAS 经验进展转化为有原则的 length-control 贡献，并配套可复现实验证据。

## 当前 central claim

DLLM 代码 infilling 的 inference-time length control 可以通过区分 medium rescue 与 true-long detection，安全恢复 medium-length under-selection；但是 true-long infilling 仍主要受 length underestimation 支配，可能需要比当前 official-CAL gate family 更强的 length-modeling signal。

术语口径：文档中的 previous/local baseline 或 local control 是本项目早前跑出的用户自有方法/控制版本，不是 CAL、LR-DLLM 或 DreamOn 方法的本地复现。论文报告值应与“我们之前的方法”和“当前方法”放在同表比较，但列名必须区分。

## 当前实验方案版本

`v3`：probe-curve-first long-length modeling 方案，并采用用户确认的 GPU allocation `2,3`，现在扩展到 full trace-long-rescue diagnostics。当前 A6000 LLaDA-Base checkpoint 是 `midcons`；trace-long-rescue Task 1/2/3/4/5 已完成。Offline Route 1/2/3 analysis 产出 negative evidence，因此不能从这批 traces 创建 route-specific GPU policy runner。

## 本次会话已完成

- 阅读了 `AGENTS.md`、近期计划、结果报告、run registry、文献记录，以及核心 clean runner/analysis 代码。
- 创建了 `paper-agent-overnight` 分支。
- 确认 GPU 0,1,2,3 已有其他 Python 任务占用；未启动重型 GPU 实验。
- 初始化了双语 `docs/paper_agent/` 研究跟踪结构。
- 新增经过测试的 evidence snapshot builder，并从本地 raw outputs 重新生成当前 A6000 evidence snapshot。
- 新增经过测试的 probe-curve signal audit；它没有找到 strict viable single-feature threshold，并确认当前 outputs 没有保存 stopping traces。
- 使用 `8` 个 focused tests、compile checks、audit regeneration、JSON assertions 和 `git diff --check` 验证了 probe-curve milestone。
- 已进入优雅暂停流程；新增 `docs/paper_agent/pause_checkpoint.current.md`，未启动新的研究或 GPU 实验。
- 以低 token 模式恢复，并将已过期 checkpoint 与当前 strict-split probe diagnostic files 对齐。
- 新增并运行 CPU-only strict-split probe-score diagnostic。它未通过 held-out safety gate，因此当前 probe-curve fields 本身不足以支持 GPU smoke run。
- 恢复后核对 dashboard/checkpoint verification 状态，并对 strict-split diagnostic 完成 fresh focused verification：unit test、py_compile、audit regeneration、JSON assertions 和 `git diff --check` 均通过。
- 已完成用户确认的 `GSAI-ML/LLaDA-8B-Instruct + midcons` full cross-model run；第一次直连 HuggingFace 因网络不可达中断，成功 run 使用 `HF_ENDPOINT=https://hf-mirror.com`。
- 核对 CAL、LR-DLLM、DreamOn 的 source PDFs，并将 literature-backbone rerun matrix 写入 `docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md`。
- 用户批准沙箱外执行后，DreamCoder Base 和 DreamCoder Instruct 2-sample smoke 均通过；随后完成两个 full `1033` sample runs，Base 在 GPU2，Instruct 在 GPU3。
- 通过 `/tmp` 下的本地 Git/LFS checkout 下载/探测 `Dream-org/Dream-v0-Base-7B`，完成 2-sample smoke 后启动并完成 Dream-7B full local same-backbone pair：cal_lite baseline 在 GPU2，LCAL official bounded-repair candidate 在 GPU3。
- 通过已配置 proxy 的 Git/LFS 路径在 `/tmp` 下载/探测 `apple/DiffuCoder-7B-Base`，完成 2-sample smoke 后启动并完成 DiffuCoder-Base full local same-backbone pair：cal_lite baseline 在 GPU2，LCAL official bounded-repair candidate 在 GPU3。
- 通过已配置 proxy 完成 `GSAI-ML/LLaDA-1.5` metadata/API/local-weight probe。该 repo 直连 HuggingFace 比 `hf-mirror.com` 更可靠；6 个权重分片已下载到 `/tmp/llada15_probe_20260609` 并完成 byte-size 校验。
- 完成 `GSAI-ML/LLaDA-1.5` 2-sample smoke pair 和 full local same-backbone pair，使用共享 GPU `2/3`。Baseline 在 GPU2 完成，candidate 在 GPU3 完成；两条日志均 exit `0`，两个输出目录均有 `1033` 个 valid rows 和 `summary.json`。
- 完成 `inclusionAI/LLaDA-MoE-7B-A1B-Base` 下载恢复、local API/weight validation、flash-attn 环境根因排查、2-sample smoke gate 和 full local same-backbone pair。Candidate `801/1033 = 77.54%`，local `cal_lite` baseline `777/1033 = 75.22%`，pairwise `31` wins / `7` losses。
- 在不使用 Superpowers skills、subagents、reviewer discovery 或 Goal tools 的约束下完成 trace-long-rescue Task 1/2/3。Action brief/current action 通过 markdown hygiene；已实现 trace feature extraction、route analysis 和 report renderer；focused verification 通过 `Ran 6 tests` / `OK`、py_compile 和 `git diff --check`。
- 完成串行 Task 4 full trace collection：previous local method trace run 在 GPU `2` 上得到 `769/1033 = 74.44%` 和 `35257` trace rows；current `midcons` trace run 在 GPU `3` 上复现 `795/1033 = 76.96%`，并得到 `35768` trace rows。两条日志均以 `COMMAND_EXIT_CODE="0"` 结束，两个 trace files 均覆盖 `1033` 个 task ids。
- 完成 Task 5 offline Route 1/2/3 analysis。Route 1 和 Route 2 在两个 trace sources 上均触发 `0` 行；Route 3 因 single-canvas traces 且 Route 1/2 没有信号，不足以支持 multi-canvas policy cost。

## 最新结果摘要

- A6000 control：`787/1033 = 76.19%`。
- A6000 `midcons`：`795/1033 = 76.96%`，相对 same-hardware control 为 `+8` wins、`0` losses。
- Long buckets 仍未改善：`17-24 = 20.73%`，`25+ = 16.13%`。
- Offline long-underestimate sweep 没有从当前 result fields 中找到安全 heuristic rule。
- 最新 snapshot：`docs/paper_agent/evidence_snapshot.md`，由 `analysis/build_paper_agent_evidence_snapshot.py` 生成。
- Probe-curve audit：`4106` 个 thresholds，`0` 个 strict viable；最佳 short-risk 为 `8.70%`，高于 `5%` GPU gate。
- Strict-split probe-score audit：`5` 个 deterministic task-id folds、`24` 个 features、aggregate held-out `63` triggers、`47.62%` true-long precision、`32.97%` failed-long recall、`22.22%` short-risk、`7.94%` current-pass risk；`strict_heldout_pass=False`。
- Verification：早前 probe-curve audit 的 `8` 个 focused tests 通过；strict-split diagnostic 在 2026-06-04 fresh verification 中通过 `Ran 3 tests` / `OK`、py_compile、audit regeneration、JSON assertions 和 `git diff --check`。
- LLaDA-Instruct cross-model run：`815/1033 = 78.90%`，低于同 backbone 历史 LCAS-v3 baseline `817/1033 = 79.09%`。Pairwise：`17` wins、`19` losses、`798` tie-pass、`199` tie-fail。Runtime 从 `6.8661s` 降到 `4.1766s` per sample including probe，但这是 negative transfer evidence，不是 claim upgrade。
- Literature-backbone plan：`docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md`。
- DreamCoder Base smoke：有效沙箱外 smoke 已通过，`2/2`，前两题与 same-backbone baseline 为 `2` tie-pass，avg total sec including probe `3.1439`。
- DreamCoder Instruct smoke：有效沙箱外 smoke 已通过，`2/2`，前两题与 same-backbone baseline 为 `2` tie-pass，avg total sec including probe `3.0555`。
- DreamCoder Base full run：`832/1033 = 80.54%`，高于同 backbone 本地 official-canvas cal_lite baseline `825/1033 = 79.86%`；pairwise 为 `27` wins、`20` losses、`805` tie-pass、`181` tie-fail；avg total sec including probe 为 `3.7763`，baseline 为 `3.7847`。这是小幅 local positive result，但还不是强 claim。
- DreamCoder Instruct full run：`834/1033 = 80.74%`，低于同 backbone 本地 official-canvas cal_lite baseline `848/1033 = 82.09%`；pairwise 为 `21` wins、`35` losses、`813` tie-pass、`164` tie-fail；avg total sec including probe 为 `3.8472`，baseline 为 `3.8657`。这是 negative transfer evidence。
- DreamCoder 文献位置：Base 高于 CAL DreamCoder-Base anchors（`70.2` average、`76.2` best shown），低于 LR-DLLM DreamCoder-7B `81.6`；DreamOn DreamCoder `92.1` 是 training-based 结果。这些是 anchors，不是 protocol-matched claims。
- Dream-7B local pair：candidate `803/1033 = 77.73%`，本地同 backbone cal_lite baseline `802/1033 = 77.64%`；pairwise 为 `28` wins、`27` losses、`775` tie-pass、`203` tie-fail；avg total sec including probe 为 `3.7337`，baseline 为 `3.6494`。这是 near-tie / slight local positive，不是强 claim。
- Dream-7B 文献位置：candidate 高于 LR-DLLM Dream-7B single-line anchor `76.7`；DreamOn Dream-7B `88.6` 是 training-based 且明显更高。这些是 anchors，不是 protocol-matched claims。
- DiffuCoder-Base local pair：candidate `839/1033 = 81.22%`，本地同 backbone cal_lite baseline `838/1033 = 81.12%`；pairwise 为 `25` wins、`24` losses、`814` tie-pass、`170` tie-fail；avg total sec including probe 为 `3.7538`，baseline 为 `3.6562`。这是 near-tie / slight local positive，不是强 bounded-repair improvement claim。
- DiffuCoder 文献位置：本地两条结果均高于 CAL DiffuCoder-Base anchors（`68.0` average、`74.8` best shown），但 DreamOn DiffuCoder-7B `92.2` 是 training-based 且明显更高。这些是 anchors，不是 protocol-matched claims。
- LLaDA-1.5 probe：本地路径 `/tmp/llada15_probe_20260609`；architecture `LLaDAModelLM`；`model_type=llada`；config `mask_token_id=126336`；tokenizer 中 `<|mdm_mask|>` 解析为 `126336`，但 `tokenizer.mask_token` 为 `None`；6 个 shards byte-size 全部匹配，总大小 `16,031,197,144` bytes。这只是 API/download result，不是 pass-rate result。
- LLaDA-1.5 local pair：candidate `818/1033 = 79.19%`，本地同 backbone `cal_lite` LCAS-v3b baseline `817/1033 = 79.09%`；pairwise 为 `18` wins、`17` losses、`800` tie-pass、`198` tie-fail；avg total sec including probe 为 `6.6453`，baseline 为 `5.4224`。这是 near-tie / slight local positive，不是强 claim。Candidate 在 oracle `<=8` 净损失 `6` 个任务，在 oracle `>=13` 合计净增 `+7`，official-repair true-long precision 只有 `10.91%`。
- LLaDA-MoE local pair：candidate `801/1033 = 77.54%`，本地同 backbone `cal_lite` LCAS-v3b baseline `777/1033 = 75.22%`；pairwise 为 `31` wins、`7` losses、`770` tie-pass、`225` tie-fail；avg total sec including probe 为 `10.6107`，baseline 为 `8.7025`。所有 oracle buckets 均为正增益或持平：`<=8 +9`、`9-12 +5`、`13-16 +4`、`17-24 +6`、`25+ 0`。这是当前最强 local transfer result，但仍不是 external SOTA。
- LLaDA-Base full trace diagnostics：previous local method trace `769/1033 = 74.44%`、`35257` trace rows；current `midcons` trace `795/1033 = 76.96%`、`35768` trace rows。Route 1/2/3 在 offline accounting 下均未通过 Gate A 或 Gate B；不应创建 route-specific GPU policy runner。

## 关键方案调整

- 将 LLaDA-Base `midcons` 视为真实的 short/medium checkpoint，而不是完整解决方案。
- 将 LLaDA-Instruct `midcons` 视为 negative transfer evidence。
- 在定义更强 signal 前，停止把 GPU 投入当前 official-CAL true-long trigger family。
- 优先从当前 outputs 做 probe-curve multivariate/learned scoring；trajectory diagnostics 需要 trace-enabled smoke run。
- 将第一个 simple strict-split linear probe score 视为 negative evidence，而不是 candidate GPU policy。
- 将 trajectory features、learned length classification、DreamOn-style dynamic canvas control 或 LR-DLLM-style length regularization 提升为下一阶段 paper-level 方向。
- 将未来 GPU 实验限制在卡 `2,3`，并采用等待而非中断已有任务的方式。

## Workflow / Skill Status

| Workflow / Skill | Status | Evidence | Output files | Notes |
|---|---|---|---|---|
| gstack `/office-hours` | completed | `research_design.initial.*.md` 和 `research_design.current.*.md` 包含 research community/user、need、minimum publishable contribution；日志 `2026-05-31 12:39 CST` 记录初始化 paper-agent documents | `docs/paper_agent/research_design.initial.en.md`, `docs/paper_agent/research_design.initial.zh.md`, `docs/paper_agent/research_design.current.en.md`, `docs/paper_agent/research_design.current.zh.md` | 没有单独 CLI transcript；完成依据是产出的 research design docs |
| gstack `/plan-ceo-review` | completed | `research_design.current.en.md` 的 `CEO-Style Stress Review` 覆盖 novelty、importance、reviewer appeal、scope、central claim、weakest assumption 和 CCF-A realism | `docs/paper_agent/research_design.current.en.md`, `docs/paper_agent/research_design.current.zh.md` | 结论：当前还不是 CCF-A claim，但有可推进 foothold |
| gstack `/plan-eng-review` | completed | `experiment_plan.current.en.md` 的 `Engineering Review Summary`、datasets、baselines、metrics、ablations/kill criteria/compute budget/reproducibility sections | `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/experiment_plan.current.zh.md`, `docs/paper_agent/experiment_plan.history.en.md`, `docs/paper_agent/experiment_plan.history.zh.md` | 当前版本为 `v3` |
| Superpowers `brainstorming` | not_started | no evidence found | none | 当前方向由 gstack-style research design 和 experiment plan 承接；下次若改变 spec，应先运行 |
| Superpowers `writing-plans` | completed | 当前上下文显示已读取该 skill；`experiment_plan.current.*.md` 和 history 给出可执行计划 | `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/experiment_plan.current.zh.md`, `docs/paper_agent/experiment_plan.history.en.md`, `docs/paper_agent/experiment_plan.history.zh.md` | 没有另建 Superpowers plan 文件 |
| Superpowers `systematic-debugging` | completed | 2026-06-04 verification 中第一次 JSON assertion 使用了错误 schema path；检查 audit JSON 和脚本后确认真实 path 为 `cross_validation.aggregate_heldout` | `docs/paper_agent/current_action.md` | Root cause 是 assertion 命令写错，不是 audit metrics 改变 |
| Superpowers `verification-before-completion` | completed | 日志 `2026-05-31 21:47 CST` 和 `2026-05-31 22:11 CST` 记录早前 probe-curve audit verification；2026-06-04 fresh strict-split verification 记录 unit test、py_compile、audit regeneration、JSON assertions、`git diff --check` | `docs/paper_agent/current_action.md`, `docs/paper_agent/paper_agent_dashboard.zh.md`, `docs/paper_agent/probe_curve_split_score_audit.md` | 关键验证：probe-curve audit `Ran 8 tests` / `OK`；strict-split diagnostic `Ran 3 tests` / `OK` 且 `strict_heldout_pass=False` |
| Superpowers `requesting-code-review` | blocked | 日志 `2026-05-31 21:47 CST` 记录 no independent Task/subagent reviewer tool visible；使用 local diff review + fresh tests 作为 fallback | `docs/paper_agent/overnight_log.en.md`, `docs/paper_agent/overnight_log.zh.md` | 独立 reviewer 未完成；风险已记录 |

## 当前最大风险

当前提升幅度较小且 heuristic 色彩较强；除非下一阶段产生有原则的 long-length control 或强 cross-model/protocol-matched validation，否则不足以支撑 CCF-A 贡献。

## 下一步计划

1. 不要从当前 trace batch 创建 route-specific GPU policy runner；offline gates 已失败。
2. 将 trace diagnostics 记录为该 trace-only / risk-controlled route family 的 negative evidence。
3. 如果后续继续该方向，必须先设计更强 trace feature family 或另写 action brief，再启动任何新 GPU 工作。
4. 继续把 literature anchors、previous local methods、current methods 和 trace diagnostics 分列/分节记录。

## 需要用户决策的问题

目前没有正在运行的 GPU 实验需要接管。当前 trace batch 不支持 route-specific policy full run。
