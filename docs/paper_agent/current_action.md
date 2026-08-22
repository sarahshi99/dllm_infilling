# Current Paper-Agent Action

更新时间：2026-07-31 UTC

## 2026-08-22 DreamOn SingleLine order × parallelism × Markov premise diagnostic

Action name：`DREAMON-SINGLELINE-ORDER-PARALLEL-MARKOV-DIAGNOSTIC-V1`。

状态：`implementation_in_progress`。

目标：在 DreamOn released-source SingleLine protocol 上比较官方 global-confidence 与硬 left-to-right frontier，requested K=`1/2/4`；逐步记录真实并行度和 stale→fresh 右邻分布变化。本轮不训练 Markov head，不使用 reference 或 evaluator 影响生成。

比较边界：官方 `read_problems("single-line")` loader 在当前 pinned HumanEval-Infilling 数据源返回 `1033` 个唯一 task id；该 1033 条为 development/full-allowed population，非 held-out/frozen-test 证据。此前 `927` 是项目内部 non-frozen common manifest，不能作为本轮官方对照分母。

实现：复用 `experiments/dreamon_singleline_adapter.py` 的 pinned source/model/tokenizer/prompt/evaluator 兼容层；新增独立的 trace-enabled decoder、分析器和单元测试。官方 C1/C2/C4 保持全局 active-mask confidence top-K 语义；L1/L2/L4 只选从最左 unresolved mask 开始的连续位置，并在最早 structural action 后强制 fresh forward。

GPU/运行：physical GPU=`0`，`CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false`；在单元测试、12-case smoke、resume 和 C1 协议回归完成后，使用独立 `tmux` 启动六臂 full。输出目录为 `analysis_outputs/dreamon_singleline_order_parallelism_markov_diagnostic_20260822_v1/`；原始生成和完整 trace 保留在该目录下且不提交 Git。

Success criteria：六臂使用同一 1033-row loader population；C 系列保持官方选择语义；L 系列不跨 gap、不复用 structural action 后旧 logits；trace 可聚合真实 commits/forward、top-K 空间结构、Markov offset `1/2/3`；每臂可恢复且无重复 case。

Known risk：official released source hard-codes `number_transfer_tokens=1`，因此需要在不改变 C 系列选择含义的前提下将其参数化，并由 C1 与原始 generator 路径进行逐例/轨迹回归。常规测试、Git revision 和完整 case key 足以覆盖本轮可恢复性；不新增额外 hash/frozen contract/gate。

## 2026-07-31 CAL SingleLine 838 smoke→full（覆盖上一动作）

Action name：`CAL-PHASE1B-SINGLELINE-838`。

状态：`adapter_committed_smoke_blocked_by_external_gpu_processes_0_of_12`。

准确标签：`official-source CAL, initial length 32, on the 838-row / 143-cluster project-non-frozen SingleLine CAL-Rest subset`。

Scope：只参数化 pinned adapter 的 official HumanEval-Infilling benchmark loader/evaluator，从 MultiLine 切到官方 SingleLine；复用同一 `llada_cal.generate`、同一 seed-42 100-demo bias 参数、model/decoder/evaluator。禁止重新拟合 bias，禁止打开 frozen files。

GPU/env：physical GPU=`0`，`CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false`，最多一个项目 GPU 进程。Output=`outputs_clean/official_cal_singleline_20260731_v1/`；log=`logs/paper_agent/20260731_official_cal_singleline.log`；full tmux=`cal_singleline_838_20260731`。

Exact smoke command：

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface /home/shx/miniconda3/envs/dllm_env/bin/python experiments/p1_official_cal_adapter.py --official-cal-root /home/shx/.cache/dllm_infilling/calibrated_adaptive_length-741e8418 --humaneval-root /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff --current-dataset /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff/data/HumanEval-SingleLineInfilling.jsonl.gz --benchmark-name single-line --manifest-jsonl analysis_outputs/baseline_manifests_20260731_v1/cal_singleline_rest_nonfrozen_smoke12_manifest.jsonl --full-manifest-jsonl analysis_outputs/baseline_manifests_20260731_v1/cal_singleline_rest_nonfrozen_manifest.jsonl --output-dir outputs_clean/official_cal_singleline_20260731_v1 --arm official_cal_primary --smoke-cases 12
```

Full command（smoke+resume-noop gate 通过后，在独立 tmux 立即启动）：

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface /home/shx/miniconda3/envs/dllm_env/bin/python experiments/p1_official_cal_adapter.py --official-cal-root /home/shx/.cache/dllm_infilling/calibrated_adaptive_length-741e8418 --humaneval-root /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff --current-dataset /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff/data/HumanEval-SingleLineInfilling.jsonl.gz --benchmark-name single-line --manifest-jsonl analysis_outputs/baseline_manifests_20260731_v1/cal_singleline_rest_nonfrozen_manifest.jsonl --full-manifest-jsonl analysis_outputs/baseline_manifests_20260731_v1/cal_singleline_rest_nonfrozen_manifest.jsonl --output-dir outputs_clean/official_cal_singleline_20260731_v1 --arm official_cal_primary --auto-full
```

实际 launcher：smoke=`bash scripts/manual_launch_official_cal_singleline_20260731.sh smoke`；full=`tmux new-session -d -s cal_singleline_838_20260731 'cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1 && bash scripts/manual_launch_official_cal_singleline_20260731.sh full'`。

Success gate：smoke=`12/12` unique exact keys，missing/duplicate/error/failure=`0`，evaluator正常，forward/token/wall/memory字段完整，resume no-op writes=`0`，frozen=`sealed/0`。Full gate=`838/838`、143 clusters、同样零错误与完整账本。

Kill criteria：source/checkpoint/config/hash/population不一致；任何 failure journal；forward/token accounting错误；OOM/ECC；已有无关 GPU process。Budget：smoke约2–5分钟（含模型加载）；full约55–70分钟。运行中只读 progress/ETA/OOM/ECC/failure，不读 partial accuracy。

Fresh verification：targeted CAL/builder tests `OK`；adapter `py_compile`、launcher `bash -n`、CLI help、SingleLine certificate/hash preflight、`git diff --check` 全部通过。`reviewer_gate_disabled`，local diff review无 blocker。2026-07-31T07:10:13Z GPU0 仍有外部 PID `755980/810890/818373`，因此 smoke 尚未启动、output 未创建、进度=`0/12`；不得抢占或 kill。

## 2026-07-31 CAL MultiLine 4,990 解盲（覆盖上一动作）

Action name：`CAL-PHASE1A-MULTILINE-4990-UNBLIND`。

状态：`completed_result_ready_for_focused_commit`。

准确标签：`official-source CAL, initial length 32, on the 4,990-row / 143-cluster project-non-frozen CAL-Rest common subset`。

Reviewer motivation：确认现有 raw 确实来自 pinned official CAL primary arm，并在不伪造 fixed32 paired control 的前提下报告绝对 accuracy、grouped uncertainty 和完整成本。

Exact commands：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python - <<'PY'
# Read all 4,990 canonical rows only after analyzer commit/push; audit exact keys,
# arm/config/source/evaluator revisions, failures, and forward/token fields.
PY
/home/shx/miniconda3/envs/dllm_env/bin/python -m analysis.baseline_grouped_analyzer --manifest analysis_outputs/official_cal_corrected_protocol_20260715_v1/cal_rest_common_manifest.jsonl --method official_cal_primary=outputs_clean/official_cal_primary_20260715_sprint_v1/official_cal_primary_raw.jsonl --expected-rows 4990 --expected-clusters 143 --bootstrap-replicates 10000 --seed 20260731 --output-dir analysis_outputs/official_cal_multiline_4990_grouped_20260731_v1
```

GPU=`none`。Inputs：pinned CAL `741e8418…`、evaluator `88062ff…`、LLaDA checkpoint cache revision `0f2787f…`。Success gate：4990 unique exact manifest rows、143 clusters、0 missing/duplicate/error/failure-journal、arm/config exact、cost fields complete、forward accounting守恒、10,000 cluster bootstrap成功。Kill criteria：任一 revision/config/key/count/cost 不一致；此时不得按 official CAL primary arm解释。

Comparison boundary：同 4,990 normalized keys 上没有 completed `official_fixed32` 时，不计算 paired delta/paired CI。Fixed64 也不称为 equal-compute。

Result：row Pass@1=`1643/4990=32.9259%`；143-cluster task-macro=`27.8471%`，95% CI=`[24.3313%,31.2715%]`；integrity 与成本 gate 全部通过。正式报告=`docs/paper_agent/experiments/20260731_official_cal_multiline_4990_result.zh.md`。首次 direct-script invocation 因 Python module path 失败，使用不改变统计实现的 `python -m analysis.baseline_grouped_analyzer` 成功完成。

## 2026-07-31 当前最小动作（覆盖历史 action）

Action name：`BASELINE-CLOSURE-PHASE0-PROTOCOL-ANALYZER-FREEZE`。

状态：`verified_ready_for_focused_commit_outcome_blind`。

阶段：external baseline closure Phase 0。用户已明确授权补全/规范化 external baselines，但排除 M5、M1--M4、PPT、ExecRepoBench final benchmark 和 frozen controller test。本动作先冻结 population/protocol matrix、真正可执行组合的 immutable manifest、统一 grouped analyzer 与 synthetic fixtures；在 focused commit/push 成功前，不读取 official CAL 4,990 的 final `passed` outcome，也不启动任何 full run。

Reviewer motivation：外部 baseline 只有在 population、decoder、sampling、seed、evaluator、candidate keys、统计单位和成本账本一致时才能支持直接比较；否则必须分离为 literature anchor、official-source/paper-protocol reproduction on project non-frozen subset 和 common-protocol local comparison。

涉及文件：

- `docs/paper_agent/baseline_population_and_protocol_matrix.current.zh.md`
- `docs/paper_agent/baseline_population_and_protocol_matrix.current.json`
- `docs/paper_agent/experiments/20260731_baseline_closure_phase0_protocol_analyzer_freeze.zh.md`
- `analysis/baseline_grouped_analyzer.py`
- `analysis/build_cal_singleline_rest_manifest.py`
- `tests/test_baseline_grouped_analyzer.py`
- `tests/test_build_cal_singleline_rest_manifest.py`
- `analysis_outputs/baseline_manifests_20260731_v1/`

Exact CPU commands：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_baseline_grouped_analyzer.py tests/test_build_cal_singleline_rest_manifest.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/baseline_grouped_analyzer.py analysis/build_cal_singleline_rest_manifest.py
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/build_cal_singleline_rest_manifest.py --official-singleline /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff/data/HumanEval-SingleLineInfilling.jsonl.gz --official-multiline /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff/data/HumanEval-MultiLineInfilling.jsonl.gz --multiline-common-manifest analysis_outputs/official_cal_corrected_protocol_20260715_v1/cal_rest_common_manifest.jsonl --allowed-singleline-manifest analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/manifest.csv --output-dir analysis_outputs/baseline_manifests_20260731_v1 --smoke-cases 12
git diff --check
```

GPU/env/log/output：CPU-only；GPU=`none`。不创建 GPU log；manifest 输出写入 `analysis_outputs/baseline_manifests_20260731_v1/`。

Success gate：SingleLine CAL-Rest ∩ non-frozen=`838` rows/`143` clusters，Demo=`100`、Rest=`933`，与已有 allowed manifest 一致，frozen intersection 通过既有 `included_not_frozen_controller_test` 证明为零；manifest hash 固定；synthetic analyzer 正确输出 row Pass@1、equal-weight task macro、10,000 cluster bootstrap CI、row/group help-harm、forward/token/wall/memory、动态长度/扩缩/终止分布，并拒绝不同 candidate-key 集合的 paired comparison。

Kill criteria：任何 population/count/hash/source revision 不一致；builder 需要读取两份 sealed frozen 文件；analyzer 允许非同 key 配对；synthetic tests、JSON/CSV parse、forbidden-input 或 diff hygiene 失败。

Known risks：历史 allowed manifest 含 oracle-length metadata，但本动作只使用 `task_id/task_group/source_dataset/frozen exclusion flag` 构造 population，不把 oracle length 写入新 manifest或 analyzer deployable inputs。`reviewer_gate_disabled`，采用 local diff review + fresh verification fallback。

Expected documentation outputs：baseline matrix、Phase 0 brief、immutable manifest summary/hash、后续各 baseline 的准确标签和 blocker 状态；不改变 central claim。

Fresh verification：17 个相关 tests `OK`；`py_compile` 通过；matrix/manifest summary JSON parse 通过；正式 manifest audit=`838 rows/143 clusters`，smoke=`12 rows/12 clusters`，forbidden fields=`0`；`git diff --check` 通过。`reviewer_gate_disabled`，local diff review 未发现 blocker。

权威路线：`docs/paper_agent/ccfa_master_roadmap.zh.md`；M1--M4 唯一当前状态登记：`docs/paper_agent/method_portfolio.current.json`。

<!-- method-portfolio-status: M1=reviewed_not_promoted_v0; M2=reviewed_not_promoted; M3=reviewed_not_promoted_v0; M4=reviewed_not_promoted_v0_multilinecore_cross_source -->

## 2026-07-29 当前最小动作（覆盖历史 action）

Action name：`EXPERIMENT-COMPLETION-AUDIT-20260729`。

状态：`completed_awaiting_next_stage_approval`。正式报告和可执行审批 prompt 已生成；本轮未启动新 GPU 作业。

阶段：Execution Sprint V1 收口与下一阶段审批准备。维护目标是把 M1--M4、official CAL、其他 official reproduction、历史 full/local transfer evidence 的完成状态和科学结论整理为一个 GitHub 可读的中文报告，并提供一份不自动启动新 GPU full 的待审批执行 prompt。审稿人关切是区分 protocol-matched official result、local diagnostic、技术完整性完成但 outcome 未分析、科学 gate 主动停止和真实 protocol blocker。

本动作只读取 compact status/result artifacts，写入 `docs/paper_agent/experiments/20260729_experiment_completion_status_and_next_stage.zh.md` 与 `docs/paper_agent/prompts/20260729_next_stage_approval_prompt.zh.md`；不读取 raw generated code，不打开 frozen test，不启动 GPU，不改变 central claim。Markdown/diff review、路径存在性与 `git diff --check` 已通过；focused commit/push 记录见本轮 Git 历史。输出是一份用户可直接审阅的完整状态矩阵和可批准的下一阶段方案。

## 2026-07-28 当前最小动作（覆盖历史 action）

四个 M1--M4 V0 都已完成当前已授权的 148-group formal decision，且没有 paper primary。official CAL 的 4,990-case full 已完整性完成（`4990/4990`、missing/error=`0/0`；运行期间未读取 partial accuracy）。M4 mandatory `40,632` bank 与 RandomSpanLight context 的交集为 `0/164`，故使用已冻结的 MultiLine-Core manifest；技术完整性通过，但 primary `assembly_without_repair − best_single`=`-12.84pp`、help/harm=`0/19`，不进入 296/927。DreamOn official source 已审计但未启动 GPU smoke；rho-EOS 是 completion-only，不能伪装成 FIM baseline。所有授权 long runs 与 CPU audit 已完成并已 push；保持 raw outputs 不变，只有新的明确 brief 才可启动后续 GPU 作业。

起点证据：`8b348f979f09cda07811e04b7ad3dee60e56373b`

## Action Name

`FAST-SPRINT-01`: official baselines、P2.1 grouped statistics、四个独立候选方法 M1--M4、以及 ExecRepoBench preparation。

## 当前决定与边界

Operational decision 是 `execute_and_monitor`，不是 blocked。M1、M2、M3、M4 是平行的独立候选方法；当前**没有**论文主方法。M1 只因最先完成代码而先进入 smoke，不代表 M2--M4 被降级、放弃或融合。Phase 5 的 `M1-D0` fixed proxy kill 和 `M4-D0/F1` premise 结果保持为历史结论，不能被改写成对四个新 V0 的 kill。

冻结 controller test 始终 sealed，`test_evaluation_count=0`；不得读取 106 个 sealed SingleLine test rows。所有 full run 必须支持 resume/dedup，并在结束后审计 missing/duplicate/error。旧共享 MultiLine candidate bank 已自然完成 `40,632/40,632` 且 final audit 通过；不重启、不重建、不改写 raw。

## 立即执行的工作包

### P1/P2/P4

- `P1.1-OFFICIAL-CAL`：official `NiuHechang/Calibrated_Adaptive_Length@741e8418a88a732b4c92812424d4f03cab1f7b1f` 的 smoke/full integrity 已完成；DreamOn `8a0a549` 的 FIM/evaluator source route 已审计，未来需新的 12-case smoke brief；rho-EOS `69992ca` completion-only，未有 faithful FIM adapter，不进入 smoke/full；LR-DLLM 只保留 blocker audit。local CAL/CAL-lite 不得叫 official CAL。
- `P2.1-GROUPED-STATS`：已在 `c66678a` 完成。`6707` spans / `20121` results 背后是 `148` base-task groups；primary 是 equal-weight base-task macro accuracy，span-micro 只 descriptive。10,000 次 cluster bootstrap、paired wins/losses、group-aware label-swap、分层 CI、cost frontier 和 8-cell intersection 已写入 compact CSV/JSON/Markdown；禁止行级独立显著性。
- `P4.1-EXECREPOBENCH`：首个外部 benchmark 已固定为 ExecRepoBench。pinned provenance 为 dataset `fa61028ce495c9ceff58398b8a7c47b5ae9f5276` 与 Qwen evaluator `33bc6aabd7791ad7b32f7e92104f11f2359ba890`；`experiments/p4_execrepobench_audit.py` 已准备只输出无代码的 repository-grouped six-fill smoke plan。实际数据/evaluator checkout 的 host 网络 fetch 在 2026-07-13 被 approval control plane `422` 阻断，故 evaluator smoke 尚未宣称完成；最终 external result 仍等方法配置冻结后才可开。

### M1.1 Abductive Program-State Bridge

两轮语义/完整性审计与修复已在 `1d9ef3f` 完成：AST/def-use dependency cone、seed-0 fixed64 检测、64-forward null fallback、等 canvas/forwards/cardinality generic、分离 raw outputs、forbidden-input/activation/determinism/resume tests 均已落地。**历史 5079-case MultiLine M1** 已在用户安全中断后成为 `safely_paused_resumable`：原三个 raw 目录 append-only 保留，不读取部分性能，未来仅胜出方法可用原目录与 `--auto-full --selected-method-only-5079` 的 existing-key dedup 恢复。

新的独立 **M1 RandomSpanLight** 已完成并正式复核：stage1=`1332/1332`、generic=`148/148`、dependency-cone=`148/148`，各自 unique 且 0 duplicate/error。预冻结 full-vs-generic fair delta=`-0.68pp`、help/harm=`0/1`，cone targeted activation=`17/148`；结论为 `reviewed_not_promoted_v0`，不得用本 outcome retune 或进入 296/927/5079。历史 5079 继续 `safely_paused_resumable`。

### M2/M3/M4 独立候选线

- `M2 Constraint-Homotopy V0`：已完成独立 `12→148`，full integrity audit 为三臂各 `148/148`、0 missing/extra/duplicate/error、64 forwards/4096 token-forwards、frozen sealed/count=0。正式 grouped result 为 `constraints_activated_no_reliable_grouped_advantage`：schedule 改变候选 hash，但没有可靠 task-group 优势；不自动进入 296/927/5079，也不阻止 M3。详见 `docs/paper_agent/experiments/m2_constraint_homotopy_20260716_randomspanlight_result.zh.md`。
- `M3 Birth-Death Canvas Diffusion V0`：行为保持 pre-launch 修正已在 `8440af1` 完成。M3 现在已正式复核：birth-death − uniform task-macro=`-4.05pp`、help/harm=`8/14`；birth/death mechanism 确实激活且 mean token-forward 更低，但 accuracy point estimate 也更低。结论为 `reviewed_not_promoted_v0`，不进入 296/927/5079；仅保留 equal-forward/non-equal-token 的成本观察。
- `M4 Semantic Particle Assembly V0`：RandomSpanLight offline 148 structural audit 已完成；cost/activation analyzer 已在 `267dda5` 加固。mandatory 40,632 MultiLine bank 的 outcome-blind 148 manifest repair 已完整运行：三臂各 `148/148`、0 missing/duplicate/error，assembly−best=`-12.84pp`、help/harm=`0/19`，故 status=`reviewed_not_promoted_v0_multilinecore_cross_source`。不覆盖既有 raw，不进入 296/927/5079。

四个 V0 的数据路线固定为：12 technical smoke → 148 RandomSpanLight → 296 MultiLine-Core / 927 non-frozen SingleLine development comparison → selected-method-only 5079 MultiLine → method freeze 后 ExecRepoBench。裸 `--auto-full` 不得直接进入 5079。禁止任何 tests/reference/canonical solution/oracle length/task ID/split/passed label 进入 deployable method。

## H200 并行约束

旧共享候选库、M1--M4、official CAL full 都已完整性结束。H200 当前空闲约 `140GiB`、ECC=0，没有本项目 GPU process。DreamOn/rho-EOS 仅做 CPU protocol/compatibility audit，不启动 GPU full。每个方法、每个进程保持独立 output/log directory。

## 证据命名

historical `802/1033` 是 A6000 historical；`796/1033` 是 H200 evidence base。两者都不是 final held-out result，且不得合并成单一结论。

## 2026-07-17 Execution Sprint V1 current state

`codex/ccfa-execution-sprint-v1`（base `ce416c4`）是权威执行分支；`afd3c45` 已 superseded。M1 5079 MultiLine 已安全暂停、可恢复但不自动恢复；M1 RandomSpanLight 新 runner、M2/M3/M4 独立 runner、共同 grouped analyzer 与 CAL adapter 已在 `9335d84` 后继续实现。完整 CPU tests 已通过，frozen test 仍为 sealed/count=0。

真实 GPU 执行已完成 M1/M2/M3 的 148 路线并正式复核；M1/M2/M3 当前都不晋级。official CAL 的 first import fail-stop 发现 pinned evaluator README 明示的 commented execution call；在不修改 checkout 的 documented one-line runtime overlay 后，12-case smoke=`12/12` integrity pass 并自动进入 4,990-case full。partial CAL accuracy 保持未读取；runtime/ETA/GPU 完整性在 `runtime_status.current.json` 记录。SciPy 的审批失败只触发 pinned import-closure 审计，不把研究判为 blocked。
