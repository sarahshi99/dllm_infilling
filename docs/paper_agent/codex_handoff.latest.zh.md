# Codex Handoff Latest

更新日期：2026-07-02 CST

## 1. 当前状态

- 工作目录：`/home/shx/projects/dllm_infilling/git_workspace`
- 分支：`codex/risk-controlled-dynamic-rescue`
- 本轮基线 commit：`5c4e79ab4592d660d20c6876d13005afcd3c6583`
- 本轮代码硬化 commit：`a9ba687e4158bee93799ab76f61f9c1be782454b`
- 本轮 repro command 补丁 commit：`8c2e1b7`
- 本轮 3-case pilot compact 结果 commit：`27b96be`
- working tree：文档更新前仅有待提交文档修改；push 后应为 clean
- 当前阶段：Phase 1 true-long action-ceiling strict 3-case GPU pilot completed
- GPU：`CUDA_VISIBLE_DEVICES=1`，NVIDIA RTX A6000

当前最可信结论：

> 这次 3-case pilot 只支持 `positive_control_only`：positive control 可稳定复现 Route2 win，且 oracle-sufficient C/D 也能通过；但一个 triggered failed-long case 和一个 missed failed-long case 在 oracle-sufficient canvas 与 steps96 下均未恢复。三例结果不支持自动扩大到 9 cases，也不能外推为 true-long 已解决。

## 2. 本轮完成内容

代码修改：

- `experiments/action_ceiling/action_ceiling_matrix.py`
  - 将 `D_oracle_sufficient_conservative` 改名为 `D_oracle_sufficient_steps96`。
  - 修正 oracle canvas cap：`oracle_len > max_canvas_length` 时不再静默标记 sufficient。
  - 增加 per-task stable seed、每个 action 前 reset Python/NumPy/torch/CUDA RNG。
  - 增加 one-case determinism check。
  - 增加 `pilot_results.csv`、`pilot_results.jsonl`、`pilot_summary.json`、`pilot_report.md`、`run_manifest.json`。
  - `run_manifest.json` 记录 branch/commit/command/repro_command/env/GPU/checkpoint/dataset/task IDs/seed protocol/action definitions/status/traceback。
- `tests/test_action_ceiling_matrix.py`
  - 覆盖 oracle cap、D action 命名、summary verdict、CSV/JSONL compact 输出。

实验输出：

- 成功 pilot：`analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/`
- 无效尝试：第一次沙箱内 GPU run 因 HuggingFace dataset cache lock 只读失败，未产生 `pilot_results.*`，未提交。

## 3. 精确运行方式

代码验证：

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile experiments/action_ceiling/action_ceiling_matrix.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_action_ceiling_matrix.py
git diff --check
```

正式 3-case pilot：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/action_ceiling/action_ceiling_matrix.py \
  --timestamp 20260702_3case_pilot_gpu \
  --task-ids-csv SingleLineInfilling/HumanEval/116/L0,SingleLineInfilling/HumanEval/85/L0,SingleLineInfilling/HumanEval/113/L3 \
  --max-pilot-cases 3 \
  --execute-pilot
```

验证 pilot artifacts：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m json.tool analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/run_manifest.json
/home/shx/miniconda3/envs/dllm_env/bin/python -m json.tool analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/pilot_summary.json
```

## 4. 结果

- verdict：`positive_control_only`
- determinism：`deterministic`
- row_count：`12`，只含指定 3 tasks，每个 task 有 A/B/C/D。
- historical replay mismatches：`[]`
- cost：sum `55.49s`，avg `4.624s`，P95 `6.724s`；run wall-clock `68.24s`

| Task | Pool | A primary | B Route2 | C oracle canvas | D oracle canvas steps96 |
|---|---|---|---|---|---|
| `SingleLineInfilling/HumanEval/116/L0` | `positive_control_rescued` | FAIL | PASS | PASS | PASS |
| `SingleLineInfilling/HumanEval/85/L0` | `triggered_failed_long` | FAIL | FAIL | FAIL | FAIL |
| `SingleLineInfilling/HumanEval/113/L3` | `missed_failed_long` | FAIL | FAIL | FAIL | FAIL |

Key summary:

- candidate existence：只在 positive control 的 C/D 中出现正确候选。
- canvas effect：`[]`
- steps96 effect：`[]`
- trigger opportunity：`[]`
- negative cases：`85/L0`、`113/L3` 的 C/D 均失败。

## 5. 研究解释

数据直接支持的事实：

- `116/L0` 的 A/B replay 与历史一致，positive control sanity 通过。
- `85/L0` 中 B 与 C 均为 oracle-sufficient canvas `32`，64 steps 仍失败；D steps96 也失败。
- `113/L3` 是 missed failed-long，B 按历史未触发复用 primary；C/D 使用 oracle-sufficient canvas `44` 仍失败。
- 当前 pilot 没有发现非 positive-control 的新正确候选。

合理推断：

- 这三例中，失败不能归因于简单 canvas 不足；至少这两个 true-long代表例更像 rescue generation/backbone limitation。
- 不应自动扩大到 9 cases；应先由研究者判断是否需要补一个更能区分 generation vs selector 的 action family。

尚未验证：

- 其他 triggered/missed failed-long rows 是否存在 canvas ceiling signal。
- local refinement、不同 fixed schedule、或更强 candidate generator 是否能恢复 `85/L0` 或 `113/L3`。
- 任何 deployable controller 的 held-out improvement。

与原假设冲突：

- “oracle-sufficient canvas + steps96 会在 true-long pilot 中产生新候选”的弱期待未被这三例支持；只有 positive control 成立。

## 6. 阻塞和风险

- 这只是 3-case pilot，不是统计结论。
- C/D 使用 oracle length，只能作为 offline ceiling，不能称为 deployable。
- 当前仍存在 HumanEval test-set 多轮探索风险；未建立 grouped held-out split。
- 首次沙箱内 run 因 `/home/shx/.cache/huggingface/...lock` 只读失败；复现 GPU run 需要可写 HF cache 或授权沙箱外执行。
- 旧 full runs 仍缺少统一 manifest、forward accounting、VRAM/P95。

## 7. 下一步建议

1. 暂停 9-case expansion，先由研究者决定是否接受 `positive_control_only` 作为 stop signal。
   - 科学问题：当前 action family 是否已足够说明 true-long pilot 缺少 candidate existence？
   - 所需代码：无新增；阅读 `pilot_report.md` 和 `pilot_results.jsonl`。
   - 改变方向的结果：若研究者认为三例代表性不足，可批准 9-case pilot；否则转向新 candidate generator。

2. 若继续 Phase 1，应先设计一个不只是 steps96 的 D action。
   - 科学问题：失败是 backbone capability 还是当前 denoising action 太弱？
   - 所需代码：新增一个预注册 local refinement 或 finite schedule action。
   - 预计输出：同样的 action matrix report。
   - 改变方向的结果：若新 action 仍无候选，应停止 true-long length-control 主线。

3. 建立 grouped split 与 controller calibration protocol。
   - 科学问题：后续 risk-controlled controller 是否可避免 benchmark leakage？
   - 所需代码：按 `HumanEval/<id>` group 划分 train/calibration/frozen validation/test。
   - 预计输出：machine-readable split JSON 和 protocol doc。

## 8. 文件索引

优先读：

- `analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/pilot_report.md`
- `analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/pilot_summary.json`
- `analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/run_manifest.json`
- `analysis_outputs/action_ceiling_20260702_3case_pilot_gpu/pilot_results.csv`
- `docs/paper_agent/review_manifest.latest.json`
- `docs/paper_agent/codex_repository_audit.zh.md`
- `analysis_outputs/action_ceiling_20260702_dryrun/report.md`

暂时不建议读：

- 全量 `outputs_clean/*/results.jsonl`
- 大 raw traces
- checkpoint/cache 目录
