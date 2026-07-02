# Codex Repository Audit

审计日期：2026-07-02 CST
审计者：Codex
审计范围：Phase 0，只读代码/证据审计；未启动 GPU 实验，未读取全量 raw outputs 内容。

## 1. 仓库状态

实际 Git 仓库位于：

```text
/home/shx/projects/dllm_infilling/git_workspace
```

外层 `/home/shx/projects/dllm_infilling` 有项目文件和一个只读 `.git` 目录，但 `git` 没有把它识别为可用仓库。本轮所有 Git 操作均以 `git_workspace/` 为工作目录。

审计前状态：

- 默认远端分支：`origin/main`
- 默认分支 commit：`2209463 Mark bilingual docs plan complete`
- 当前基线分支：`paper-agent-overnight`
- 当前基线 commit：`e2b20ae630f05c7d33a549252232b9a5c9db9045 docs: publish paper research package`
- 本轮工作分支：`codex/risk-controlled-dynamic-rescue`
- 本轮分支来源：`paper-agent-overnight @ e2b20ae`
- 远端 fetch：已执行 `git fetch --all --prune`；第一次因沙箱网络失败，获批联网后成功。
- 工作区：切分支前干净；写入本审计文档后会出现本轮文档改动。

活跃分支判断：

| 分支 | commit | 日期 | 判断 |
|---|---|---|---|
| `origin/paper-agent-overnight` | `e2b20ae` | 2026-07-02 | 最新活跃研究分支，领先 `origin/main` 26 commits |
| `paper-agent-overnight` | `e2b20ae` | 2026-07-02 | 本地与远端一致 |
| `origin/main` | `2209463` | 2026-05-29 | 默认分支，但不是最新研究状态 |
| `length-score-formula-full-runs` | `ad3682e` | 2026-06-19 | 本地 worktree 分支，早于 `paper-agent-overnight`，其结果已被研究包汇总 |
| `origin/exp/a6000-midcons-longrescue` | `cd55125` | 2026-05-29 | 历史实验分支 |
| `origin/docs/result-archive` | `49851b9` | 2026-05-29 | 历史归档分支 |

结论：不要把 `main` 当作最新进展；当前最新证据和研究文档在 `paper-agent-overnight`。

## 2. 关键文件最新位置

以下文件在 `paper-agent-overnight @ e2b20ae` 上存在，是当前应优先阅读的版本。

| 文件 | 当前路径 | 最新相关提交 |
|---|---|---|
| `ccfa_readiness_assessment.zh.md` | `ccfa_readiness_assessment.zh.md` | `e2b20ae` |
| `paper_agent_dashboard.zh.md` | `docs/paper_agent/paper_agent_dashboard.zh.md` | `e2b20ae` |
| `research_design.current.zh.md` | `docs/paper_agent/research_design.current.zh.md` | `e2b20ae` |
| `experiment_plan.current.zh.md` | `docs/paper_agent/experiment_plan.current.zh.md` | `e2b20ae` |
| `experiment_results.zh.md` | `docs/paper_agent/experiment_results.zh.md` | `e2b20ae` |
| `evidence_snapshot.md` | `docs/paper_agent/evidence_snapshot.md` | `e2b20ae` |
| `run_registry.md` | `docs/results/run_registry.md` | `e2b20ae` |
| `model_generalization_registry.md` | `docs/results/model_generalization_registry.md` | `e2b20ae` |
| `literature_sota_notes.zh.md` | `docs/results/literature_sota_notes.zh.md` | `e2b20ae` |
| `20260701_next_step_brainstorm_after_v8.md` | `docs/paper_agent/experiments/20260701_next_step_brainstorm_after_v8.md` | `e2b20ae` |
| `2026-07-02-post-v8-rescue-quality-and-local-guard-plan.md` | `docs/superpowers/plans/2026-07-02-post-v8-rescue-quality-and-local-guard-plan.md` | `e2b20ae` |
| `length_probe.py` | `expvision_dllm_clean/length_probe.py` | `e2b20ae` |
| `run_lcal_official_bounded_repair.py` | `clean_scripts/run_lcal_official_bounded_repair.py` | `e2b20ae` |
| `run_route2_trace_rescue.py` | `clean_scripts/run_route2_trace_rescue.py` | `e2b20ae` |
| `run_route2_rescue_quality_v5.py` | `clean_scripts/run_route2_rescue_quality_v5.py` | `e2b20ae` |
| `run_cal_lite_lcal_rescue_policy.py` | `clean_scripts/run_cal_lite_lcal_rescue_policy.py` | `e2b20ae` |
| clean config | `expvision_dllm_clean/config.py` | `e2b20ae` |
| legacy config | `expvision_dllm/config.py` | `e2b20ae` |

## 3. 当前结果关系

### LLaDA-Base 主线

| 方法 / run family | Pass@1 | 与前一可信锚点关系 | 解释 |
|---|---:|---|---|
| A6000 control | `787/1033 = 76.19%` | baseline | 本地同硬件控制 |
| `midcons` | `795/1033 = 76.96%` | vs control `+8/0` | 当前 short/medium checkpoint，收益集中在 `9-12`、`13-16` |
| Route2 broad len24 | `801/1033 = 77.54%` | vs midcons `+7/1` | 有 1 个 primary-pass loss，风险较高 |
| Route2 precision len24 | `800/1033 = 77.44%` | vs midcons `+5/0` | 更保守 |
| Route2 precision len32 | `801/1033 = 77.54%` | vs midcons `+6/0` | 当前最干净 Route2 证据 |
| V5.1 anchor selector | `801/1033 = 77.54%` | vs Route2 len32 `0/0` | 诊断候选池；未提升部署选择 |
| V6 short override | `802/1033 = 77.64%` | vs Route2 len32 `+1/0` | 当前 LLaDA-Base 最高 full-run pass count，但只是 selector polish |
| V7 proportional widening | `792/1033 = 76.67%` | vs midcons `+1/-4` | 负结果 |
| V8a proportional CAL score | `786/1033 = 76.09%` | vs midcons `+3/-12` | 负结果 |
| V8b proportional CAL score | `781/1033 = 75.61%` | vs midcons `+7/-21` | 负结果；long wins 变多但 short/medium 损失更大 |
| V8c proportional CAL score | `782/1033 = 75.70%` | vs midcons `+6/-19` | 负结果；reward cap 不是 hard length cap |

关键 true-long 结论：

- `midcons` 的 failed long 中 `90/91 = 98.90%` under-selected。
- Route2 precision len32 触发 `57` 行，true-long precision `61.40%`，vs `midcons` 为 `6/0/795/232`。
- Route2 precision len32 中 `33` 个 triggered failed-long 仍失败，其中 `31/33` 的 rescue length 已经 `>= oracle`。
- Route2 precision len32 仍漏掉 `56` 个 failed-long rows。
- V6 最高总分来自 1 个短候选 override，不改变 `25+` bucket。
- V8b 把 `25+` pass rate 提到 `22.58%`，但代价是 `21` 个 losses，不能作为成功方法。

结论：当前最可信的关系是 `control < midcons < Route2 precision len32 < V6`，但改进幅度小，true-long `25+` 仍未解决。V7/V8 是比例放长路线的负证据。

### V4 到 V8 的证据含义

| 阶段 | 类型 | 主要输出 | 结论 |
|---|---|---|---|
| Trace feature audit v2 | CPU-only diagnostic | `analysis_outputs/trace_feature_audit_v2_20260613_204721/report.md` | trace 有信号，但跨源稳定性不足，decision `diagnostic_only` |
| Route2 V3 error analysis | CPU-only diagnostic | `analysis_outputs/route2_error_analysis_20260617_165806/report.md` | decision `mixed_rescue_quality_and_gate_recall` |
| Discovery V4 | CPU-only signal audit | `analysis_outputs/discovery_v4_signal_audit_20260618_000000/report.md` | decision `route2_polish_only`，没有稳定低风险 V4 signal |
| V5.1 | GPU full, rescue-quality diagnostic | `outputs_clean/full_route2_rescue_quality_v5_anchor_m002/m010_*` | candidate upper bound `9/57`，selected 仍 `801` |
| V6 | GPU full, selector polish | `outputs_clean/full_route2_v6_short_override_gpu2_20260620_124754` | `802/1033`，`+1/0` vs Route2 |
| V7 | CPU audit + GPU full | `analysis_outputs/proportional_length_widening_v7_midcons_20260630/report.md` | full `792/1033`，负结果 |
| V8 | GPU full | `outputs_clean/full_v8a/b/c_*` | 三条均低于 midcons；全局比例奖励拒绝 |

## 4. 关键 run 可复现性

### 已较好可复核的内容

较新的 runner 输出通常包含：

- `config.json`
- `summary.json`
- `results.jsonl`
- 需要 trace 的 run 还包含 `step_traces.jsonl`
- V5/V6 还包含 `candidate_upper_bound.csv`
- 对应日志位于 `logs/paper_agent/`

较好可复核的 run：

| Run | 证据完整性 | 备注 |
|---|---|---|
| Route2 precision len32 | 高 | `config.json`、`summary.json`、非空 traces、日志 `COMMAND_EXIT_CODE=0` |
| V5.1 anchor runs | 高 | 保存多候选 summary、candidate upper bound 和日志 |
| V6 short override | 高 | 保存 config/summary/candidate upper bound 和日志 |
| V7/V8 | 中高 | 保存 config/summary 和日志；是负结果 |
| cross-model local pairs | 中 | 多数有 summary/log；部分 checkpoint 下载路径在 `/tmp` 或依赖 HF mirror |
| A6000 control/midcons | 中 | 有 raw path/summary/report/launcher，但旧 run 缺少统一 manifest |

### 可复现性缺口

当前还缺少统一的 run manifest。多数输出目录没有显式记录：

- Git commit hash；
- 完整 shell command；
- Python/torch/CUDA/driver/env 版本；
- checkpoint resolved local path 和权重校验；
- wall-clock P95；
- GPU 型号、显存峰值；
- forward-call budget 的统一计数；
- 是否使用远端 HuggingFace、`hf-mirror.com` 或本地 `/tmp` 缓存；
- 对旧 run，日志路径与输出路径的机器可读关联。

`JsonlLogger` 会写 `config.json`，但不会自动写 commit、command、environment 或不可覆盖保护 manifest。输出目录使用 timestamp，通常不会覆盖，但 `os.makedirs(..., exist_ok=True)` 不是严格防覆盖机制。

结论：结果指标大多可从 raw outputs/summary 复核；完整执行环境尚未达到论文级 reproducibility manifest 标准。

## 5. Runner gate/action/selector 审计

### `expvision_dllm_clean/length_probe.py`

- 信号：对每个候选 mask length 计算 mask 位置平均最大 token probability。
- score modes：
  - `raw`
  - `length_power`: `raw_score * length^alpha`
  - `length_power_proportional`: `raw_score * length^(alpha + beta * log(length/ref))`
- selector：按 score 选最大；tie 可选 `shorter` 或 `longer`。
- 风险：跨长度比较平均最大概率和人工 length reward 都是 heuristic，当前没有独立校准。

### `clean_scripts/run_cal_lite_lcal_rescue_policy.py`

- primary：LCAL rescue policy + LCAS-v3 stopping。
- gate：
  - strong gate：base selected length 达到 `strong_min_len`。
  - weak gate：base length 在 weak window 内，且 long score ratio、raw ratio、support count 等通过。
- action：进入 weak/strong correction grid，选择更长或修正长度。
- selector：`correction_selection_rule`，常见为 `shortest_supported`，不是 learned selector。
- 记录：保存 LCAL trigger fields、oracle length offline accounting、baseline comparison。

### `clean_scripts/run_lcal_official_bounded_repair.py`

- primary：先跑 LCAL/S3 selection。
- official-CAL probe：只在 `s3_selected <= official_eval_max_s3_len` 时评估。
- gate/action：
  - `official_bounded_repair`: official length 在 `[repair_min_official_len, repair_max_official_len]` 且 delta 在 `[repair_min_delta, repair_max_delta]`。
  - `official_long_suspicion`: official length 超过 suspicion 下界。
  - `official_mid_rescue`: official length、delta、long_ratio、source 同时过阈值。
  - optional `proportional_widening`: 在 base selection 前用比例阈值放宽到 longer near-best candidate。
- selector：规则优先级为 bounded repair，然后 suspicion，然后 mid rescue；触发后直接使用 official selected length。
- 记录：保存 resolved config、trigger reason、proportional widening fields、summary。
- 风险：多个阈值来自同一测试集多轮探索；触发后使用官方长度不是经过校准的 action utility estimate。

### `clean_scripts/run_route2_trace_rescue.py`

- primary：固定为 `default_midcons_settings()`。
- gate：
  - `broad_plateau`: `top1_last <= 0.667969` 且 `max_remaining_plateau_steps >= 16`。
  - `precision_top1_conf`: `top1_median <= 0.464844` 且 `confidence_max <= 0.84375`。
- action：若 trigger，运行 fixed-length rescue，实际 rescue length 为 `max(primary_selected, rescue_length)`。
- selector：trigger 后无条件用 rescue 覆盖 primary；未 trigger 保留 primary。
- 记录：保存 primary/rescue metrics、combined cost、trace features、baseline pairwise。
- 风险：gate 是手写阈值；action 固定 len24/len32；selector 没有估计 harm probability。

### `clean_scripts/run_route2_rescue_quality_v5.py`

- primary/gate：复用 Route2 gate。
- action：对 triggered rows 生成候选集：
  - `cheap`: 常用 `len24_s64`、`len32_s64`、`len32_s96`。
  - `full`: 更大候选集合，成本更高。
- selectors：
  - `consensus_confidence`: text consensus + confidence/top1/gap/remaining ratio。
  - `syntax_aware`: 增加 parse/compile 辅助，claim boundary 不同。
  - `anchor_len32_confidence`: 默认保护 `len32_s64`，只允许非 anchor len32 候选以 margin 替换。
  - `anchor_len32_short_trace_override`: 默认保护 `len32_s64`，只有 short candidate 的 trace gap/top1 同时强过阈值才 override。
  - `oracle_upper_bound`: 仅 offline ceiling，不能部署。
- 记录：保存 candidate details、selector metadata、oracle upper-bound summary。
- 风险：V6 只捕获 1 个额外 win；candidate upper bound `9/57` 暗示 selector-only 空间有限。

### `expvision_dllm_clean/config.py`

- 默认 dataset split 是 `test`。
- 默认 dataset subset 是 `HumanEval-SingleLineInfilling`。
- 默认 seed 是 runner CLI 的 `42`。
- 默认 score mode 在 clean config 是 `raw`，但多数 runner 会覆盖为 `length_power`。

## 6. 数据划分与泄漏风险

当前最大 protocol 风险是 benchmark leakage。

事实：

- 主数据集是 `HumanEval-SingleLineInfilling/test`，共 `1033` rows。
- 现有主要结果、诊断、V4/V5/V6/V7/V8 都围绕这同一批 `1033` rows 反复分析。
- `task_id` 形如 `SingleLineInfilling/HumanEval/116/L0`。同一原始 HumanEval task 可能对应多个 infill locations。
- 现有 strict-split diagnostics 使用 full `task_id` 做 SHA256 folds。

风险：

- full `task_id` fold 可能把同一原始 HumanEval task 的不同 `L*` location 分到不同 folds，导致 grouped leakage。
- 没有冻结的 train/calibration/validation/final-test 四分法。
- V4/V5/V6/V7/V8 的规则和阈值受到同一最终 test set 上反复 error analysis 影响。
- 当前 full-run pass count 仍有工程参考价值，但不能直接作为 held-out controller 结果。

建议：

1. 解析 original task group，例如 `HumanEval/<id>`，建立 grouped split。
2. controller 的 feature/threshold/action selection 不应使用最终 held-out HumanEval test。
3. Phase 1 action-ceiling 可以使用当前 test outputs 做诊断，但必须标注为 ceiling/anatomy，不是 deployable held-out result。
4. Phase 3 risk-controlled controller 必须先冻结 train/calibration/validation/test protocol。

## 7. 结论来源分级

### 相对可信的直接事实

- `paper-agent-overnight @ e2b20ae` 是当前最新研究状态。
- `midcons` 相对 A6000 control 为 `+8/0`。
- Route2 precision len32 相对 `midcons` 为 `+6/0`。
- V6 相对 Route2 precision len32 为 `+1/0`。
- V7/V8 全局比例放长路线为负。
- Route2 triggered failed-long 中大多数已经 rescue length `>= oracle` 仍失败。
- Discovery V4 没有找到可直接启动 GPU full run 的稳定低风险 signal。

### 合理推断

- 当前瓶颈不是单一 canvas length；rescue generation quality、selector、gate recall 同时限制。
- true-long `25+` 不应继续用盲目加长作为默认路线。
- action-ceiling matrix 比继续 V8 beta sweep 更有决策价值。

### 尚未验证

- 是否存在一个固定 conservative schedule 能显著提高 length-sufficient rescue candidate quality。
- 如果 candidate pool 中存在正确答案，当前 verifier-free selector 能否稳定识别。
- 未触发 failed-long rows 是否可以由新的 inference-visible gate 低风险捕获。
- 当前方法在 protocol-matched CAL/LR-DLLM/DreamOn baseline 下是否有竞争力。

### 与早期假设冲突的结果

- “比例式奖励更偏向长答案会提升整体结果”被 V7/V8 full runs 反驳。
- “true-long 主要只差更长 canvas”被 Route2 V3/V5 证据削弱：大量 triggered failed-long 已经 length-sufficient 仍失败。

## 8. 缺失项清单

实验复现缺口：

- 统一 `manifest.json`，记录 command、commit、env、checkpoint、GPU、seed。
- output directory 的严格 no-overwrite 检查。
- 一致的 forward-call accounting。
- P95 wall-clock 和显存峰值。
- bootstrap confidence interval。
- grouped split 文件和 frozen held-out protocol。

研究证据缺口：

- true-long action-ceiling matrix。
- protocol-matched baselines：fixed length、always-len32、compute-matched best-of-two、CAL official、LR-DLLM Stage I/II、DreamOn-style dynamic canvas。
- independent calibration/test separation。
- rescue candidate generation 的上限：更长 canvas、不同 schedule、少量固定 seeds/local refinement 是否能产生正确候选。

## 9. Phase 1 最小实现建议

审计后建议进入 `True-long action-ceiling matrix`，但先做小规模、预注册、dry-run-capable 实现。

推荐 case pools：

- positive controls：Route2/V6 已救回的 rows，例如 `HumanEval/116/L0`、`16/L0`、`34/L0`、`60/L0` 等。
- triggered but failed long：Route2 triggered 且 baseline/Route2 都失败的 `17-24`、`25+` rows，例如 `HumanEval/108/L6`、`11/L6`、`115/L0`。
- missed failed-long：midcons failed-long 且 Route2 未触发 rows，例如 `HumanEval/10/L5`、`104/L2`、`107/L7`。

推荐 actions：

| Action | 部署性 | 目的 |
|---|---|---|
| A: current primary `midcons` | deployable | 当前 baseline |
| B: current Route2 rescue len32 | deployable-ish | 当前 selective rescue |
| C: oracle-sufficient canvas | offline ceiling only | 判断 canvas adequacy ceiling |
| D: oracle-sufficient canvas + fixed conservative schedule | offline ceiling / pilot | 判断 generation schedule 是否限制 rescue quality |

最小代码应支持：

- `--dry-run`：只生成 case/action manifest 和预计成本，不加载模型。
- `--task-ids-csv`：只跑指定 case。
- `--max-cases-per-pool`：pilot 控制规模。
- `--output-dir analysis_outputs/action_ceiling_<timestamp>`。
- 保存 `config.json`、`case_manifest.csv`、`summary.csv`、`report.md`。
- 对 C/D 明确标记 `oracle_used_for_offline_ceiling_only`，防止误写成 deployable method。

Phase 1 决策规则：

- 若 C/D 也几乎不能生成正确候选，停止 true-long length-control 主线，转向 backbone/generation limitation 或 diagnostic paper。
- 若 C/D 有候选但 deployable selector 选不出，优先研究 selector/action selection。
- 若 missed failed-long 中 C/D 有明显 ceiling，优先研究 trigger recall。
- 若 positive controls 不能复现，先修 runner/环境，不做新方法。
