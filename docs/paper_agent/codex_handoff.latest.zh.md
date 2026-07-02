# Codex Handoff Latest

更新日期：2026-07-02 CST

## 1. 当前状态

- 工作目录：`/home/shx/projects/dllm_infilling/git_workspace`
- 分支：`codex/risk-controlled-dynamic-rescue`
- 本轮起点 commit：`40d17e41ce04f4078bb2728f08449905a8defb2c`
- 本轮已提交 commits：`a938e9b`、`ca67f83`、`d6c7ce8`、`af9e709`、`e8b935e`
- 文档闭环 commit：`e8b935e`；最终 push 后以 `origin/codex/risk-controlled-dynamic-rescue` 的 HEAD 为准。
- working tree：最终 push 前应仅有必要的 handoff metadata refresh；push 后应为 clean。
- 当前阶段：Phase 1b `Distinct-Candidate Generation Ceiling` strict 3-case GPU pilot completed
- GPU：`CUDA_VISIBLE_DEVICES=2`，NVIDIA RTX A6000

当前最可信结论：

> 接受上一轮 `positive_control_only` 后，本轮 Phase 1b 只支持 `candidate_diversity_without_correctness`：no-early-commit 在 `113/L3` 产生了不同候选 hash，但 hard cases `85/L0` 与 `113/L3` 没有出现正确候选；multi-seed 没有带来额外候选多样性；trace-remask 执行了真实 remask/refinement，但最终 hash 与 C 相同。当前不建议扩展到 9 cases 或 full benchmark。

## 2. 本轮完成内容

代码与测试：

- `experiments/action_ceiling/action_ceiling_matrix.py`
  - 增加旧 3-case A/B/C/D pilot 的 action-equivalence audit。
  - 报告中将 `canvas_effects` 明确拆分为 `pass_level_canvas_effects` 和 `candidate_level_canvas_effects`。
- `experiments/action_ceiling/distinct_candidate_ceiling.py`
  - 新增 Phase 1b runner：C oracle-sufficient、E no-early-commit、F trace-remask。
  - 支持 action-distinctness smoke gate、固定 seeds `0,1,2`、case/action manifests、run manifest、candidate diversity summary、action equivalence summary。
  - 压缩 JSONL trajectory，避免保存 top1/prob raw histories。
- `tests/test_action_ceiling_matrix.py`
  - 覆盖 early commit 可关闭、F remask/refinement、remask rule 不依赖 verifier label、seed 顺序、candidate hash clustering、Pass@3、output-change 与 correctness-change 分离、action-distinctness stop rule。

结果与文档：

- 旧 pilot 等价性审计：`analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/action_equivalence.{json,md}`。
- Phase 1b 预注册：`docs/paper_agent/20260702_distinct_candidate_ceiling_action.md`。
- Phase 1b compact pilot：`analysis_outputs/distinct_candidate_ceiling_20260702_phase1b_distinct_pilot/`。
- 文献/novelty 边界更新：`docs/paper_agent/literature_sota_notes.zh.md` 与 `docs/results/literature_sota_notes.zh.md`。

## 3. 精确运行方式

旧 action-equivalence 审计：

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/action_ceiling_matrix.py \
  --timestamp 20260702_3case_pilot_gpu \
  --audit-existing-pilot-dir analysis_outputs/action_ceiling_20260702_3case_pilot_gpu
```

Phase 1b GPU pilot：

```bash
CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/distinct_candidate_ceiling.py \
  --timestamp 20260702_phase1b_distinct_pilot
```

必要验证：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile experiments/action_ceiling/action_ceiling_matrix.py experiments/action_ceiling/distinct_candidate_ceiling.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_action_ceiling_matrix.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m json.tool analysis_outputs/distinct_candidate_ceiling_20260702_phase1b_distinct_pilot/run_manifest.json
/home/shx/miniconda3/envs/dllm_env/bin/python -m json.tool analysis_outputs/distinct_candidate_ceiling_20260702_phase1b_distinct_pilot/candidate_diversity_summary.json
/home/shx/miniconda3/envs/dllm_env/bin/python -m json.tool analysis_outputs/distinct_candidate_ceiling_20260702_phase1b_distinct_pilot/pilot_summary.json
/home/shx/miniconda3/envs/dllm_env/bin/python -m json.tool docs/paper_agent/review_manifest.latest.json
git diff --check
```

## 4. 结果

旧 A/B/C/D action-equivalence：

- `116/L0`：B/C/D 同 hash；A 不同。
- `85/L0`：B/C/D 同 hash；A 不同。
- `113/L3`：A/B 同 hash；C/D 同 hash。
- D `steps96` 与 C 配置不同，但在三例中输出完全等价；不能再把单纯加 steps 当作有效 action。

Phase 1b action-distinctness gate：

- task：`SingleLineInfilling/HumanEval/85/L0`
- gate：passed
- hash distinct from C：`false`
- auditable distinct trajectory：`true`
- E early commit disabled：`true`
- E more actual forwards than C：`true`
- F remasked token count：`4`
- F refinement executed：`true`

正式 pilot：

| Task | Role | Unique hashes | C seeds 0/1/2 | E seeds 0/1/2 | F seeds 0/1/2 | Error |
|---|---|---:|---|---|---|---|
| `SingleLineInfilling/HumanEval/116/L0` | positive control | 1 | PASS/PASS/PASS | PASS/PASS/PASS | PASS/PASS/PASS | `None` |
| `SingleLineInfilling/HumanEval/85/L0` | triggered failed-long syntax failure | 1 | FAIL/FAIL/FAIL | FAIL/FAIL/FAIL | FAIL/FAIL/FAIL | `SyntaxError` |
| `SingleLineInfilling/HumanEval/113/L3` | missed failed-long semantic failure | 2 | FAIL/FAIL/FAIL | FAIL/FAIL/FAIL | FAIL/FAIL/FAIL | `UnitTestFailure` |

Action cost, avg seconds including probe:

- C oracle-sufficient：`4.134`
- E no-early-commit：`4.709`
- F trace-remask：`5.230`

Verdict：`candidate_diversity_without_correctness`。

## 5. 研究解释

数据直接支持的事实：

- `116/L0` positive control 仍然稳定通过，说明 pipeline/replay sanity 成立。
- `85/L0` 在 C/E/F、三个 seed 下均为同一 hash，均 `SyntaxError`；E/F 的轨迹操作没有改变最终候选。
- `113/L3` 中 E 产生了与 C/F 不同的候选 hash，但仍为 `UnitTestFailure`；F 与 C hash-equivalent。
- 没有 non-positive-control correct candidate；offline candidate-existence Pass@3 只在 positive control 成立。

合理推断：

- 当前模型/runner 存在有限 trajectory exploration 空间，因为 E 能改变 `113/L3` 候选。
- 当前 E/F action 不足以修复 hard true-long cases；不能把这写成 deployable controller 失败，也不能写成 backbone 最终无能力。
- 由于 multi-seed 未产生多样性，当前 decoder 更接近 deterministic trajectory lock-in，而非简单 seed exploration 不足。

尚未验证：

- 更强的 pre-registered candidate generator、局部代码结构 refinement 或不同 backbone 是否能产生 hard-case 正确候选。
- 按 grouped split 拟合的 deployable trigger/selector 是否有效。
- 当前三例是否统计代表全体 true-long failure。

与原假设冲突：

- “oracle-sufficient canvas + no-early-commit/trace-remask 会在 hard true-long 上产生正确候选”的期待未被支持。

## 6. 阻塞和风险

- Phase 1b 仍是 3-case diagnostic，不是 full benchmark。
- C/E/F 使用 oracle/reference length，只能作为 offline ceiling。
- HumanEval 已被多轮探索使用，仍有 benchmark leakage 风险；后续 controller 必须建立 grouped calibration/test split。
- F remasking 与外部 T2M/RemeDi 方向重合，不能作为原创点。
- 文献边界显示 CAL/LR-DLLM 已覆盖 length calibration/variable-length inference，DreamOn 已覆盖 dynamic canvas；本项目潜在原创性应收敛到 canvas adequacy、rescue adequacy 和 risk-controlled selective action 的联合建模。

## 7. 下一步建议

1. 停止扩展当前 Route2-based generation family 到 9 cases。
   - 科学问题：当前 E/F 的候选探索是否只产生错误变体？
   - 所需代码：无。
   - 预计输出：对 `113/L3` 的两个 hash 做 compact diff/error analysis；对 `85/L0` 的 SyntaxError 做结构定位。
   - 改变方向的结果：若错误显示是局部可修复 syntax pattern，可设计一个新的预注册 code-aware refinement。

2. 若继续 Phase 1，先设计更强但仍小规模的 candidate generator，而不是增加 seeds/steps。
   - 科学问题：失败是 candidate exploration failure 还是 backbone capability limitation？
   - 所需代码：新的有限 action family，运行前写 brief 和 kill criteria。
   - 预计输出：仍限定 3 cases 的 candidate-existence report。
   - 改变方向的结果：hard cases 出现正确候选才进入 trigger/selector 研究。

3. 启动 grouped split protocol。
   - 科学问题：后续 risk-controlled controller 是否能避免 HumanEval leakage？
   - 所需代码：按 `HumanEval/<id>` 分组生成 train/calibration/frozen validation/test split。
   - 预计输出：split JSON、protocol doc、held-out-only evaluation script。

## 8. 文件索引

优先读：

- `analysis_outputs/distinct_candidate_ceiling_20260702_phase1b_distinct_pilot/pilot_report.md`
- `analysis_outputs/distinct_candidate_ceiling_20260702_phase1b_distinct_pilot/pilot_summary.json`
- `analysis_outputs/distinct_candidate_ceiling_20260702_phase1b_distinct_pilot/candidate_diversity_summary.json`
- `analysis_outputs/distinct_candidate_ceiling_20260702_phase1b_distinct_pilot/action_equivalence.json`
- `analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/action_equivalence.md`
- `docs/paper_agent/20260702_distinct_candidate_ceiling_action.md`
- `docs/paper_agent/review_manifest.latest.json`
- `docs/paper_agent/literature_sota_notes.zh.md`

暂时不建议读：

- 全量 `outputs_clean/*/results.jsonl`
- 大 raw traces
- checkpoint/cache 目录
