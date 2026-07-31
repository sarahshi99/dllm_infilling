# DreamCoder controls under DreamOn sampling/decoder

日期：2026-07-31 UTC

## Scope / label

准确标签：**DreamCoder Fixed4/8/16/32/64 under DreamOn sampling/decoder**。这是 common-protocol local control，不是 DreamOn、official CAL、DreamOn equal-compute control 或新的方法。

每个 FixedL 使用 `Dream-org/Dream-Coder-v0-Base-7B@2346ccd3be517d0d314152b988a3b9bafa7d6d63`，直接调用 pinned `DreamLM/DreamOn@8a0a54918412eda9402a327646f7f067f7160ec8` 的 `MDMGenerator`。固定语义为 `min_gen_len=max_gen_len=L`；released source 的 EOS contraction 保留，因此“FixedL”指固定初始/最大 canvas，不保证最终 decoded token 数恒等于 L。

## Frozen protocol

- population：SingleLine project non-frozen `927 rows / 148 clusters`；smoke=`12 rows / 12 clusters`；复用已冻结 manifests/hashes。
- lengths：`4/8/16/32/64`；steps=`256`；temperature=`0.2`；top-p=`0.9`；entropy remasking；source expansion logic 保留但受 `max_gen_len=L` 约束。
- evaluator：HumanEval-Infilling `88062ff9859c875d04db115b698ed4b0f0395170`。
- seed：global seed `42`；每个 FixedL 复用同一 row 对应 DreamOn dynamic minL/max64 arm 的 deterministic seed key。
- raw append-only；successful candidate keys dedup；failure journal独立；progress manifest每行更新。
- frozen test=`sealed`，`test_evaluation_count=0`。

## Exact commands

CPU preflight（不加载GPU模型）：

```bash
PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages \
HF_MODULES_CACHE=/tmp/hf_modules_dreamon_20260731 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/dreamon_singleline_adapter.py \
  --source-root /tmp/dllm_infilling_protocol_audit_20260728/DreamOn \
  --evaluator-root /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff \
  --model-snapshot /home/shx/.cache/huggingface/hub/models--Dream-org--Dream-Coder-v0-Base-7B/snapshots/2346ccd3be517d0d314152b988a3b9bafa7d6d63 \
  --model-profile dreamcoder_fixed \
  --manifest-jsonl analysis_outputs/baseline_manifests_20260731_v1/lrdllm_common_singleline_technical_smoke12_manifest.jsonl \
  --full-manifest-jsonl analysis_outputs/baseline_manifests_20260731_v1/lrdllm_common_singleline_nonfrozen_manifest.jsonl \
  --manifest-summary-json analysis_outputs/baseline_manifests_20260731_v1/lrdllm_common_manifest_summary.json \
  --output-dir outputs_clean/dreamcoder_dreamon_fixed_singleline_20260731_v1 \
  --min-gen-len 64 --mode smoke --seed 42 --progress-every 1 --preflight-only
```

每个 length 的 smoke / resume：

```bash
ALLOW_SHARED_GPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash scripts/manual_launch_dreamcoder_dreamon_fixed_20260731.sh smoke <L>
```

full 独立 tmux：

```bash
tmux new-session -d -s dreamcoder_dreamon_fixed<L>_sl927_20260731 \
  'cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1 && export ALLOW_SHARED_GPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; exec bash scripts/manual_launch_dreamcoder_dreamon_fixed_20260731.sh full <L>'
```

GPU=`physical index 0 / NVIDIA H200 NVL`；env=`dllm_env`，offline model/source/evaluator caches。

## Paths / budget

- output：`outputs_clean/dreamcoder_dreamon_fixed_singleline_20260731_v1/`
- log：`logs/paper_agent/20260731_dreamcoder_dreamon_fixed<L>.log`
- budget：5×12 smoke，gate后5×927 full；每个进程约16--24 GiB H200，按实时free memory门控并行，不抢占外部任务。

## Success gate / kill criteria

Success：每个 smoke `12/12` exact unique，missing/duplicate/error=`0`，resume `new_rows_written=0`，forward/token/length accounting完整，frozen=`0`；full只有在本 adapter/tests/action/matrix focused commit/push 后启动。

Kill/stop：source/checkpoint/evaluator/hash不符；candidate seed无法与对应 dynamic arm对齐；OOM/ECC；missing/duplicate/error；forward或length accounting不成立；任何 frozen/evaluator outcome 泄漏。资源型 OOM只停止该arm并保留failure journal，其他独立 baseline继续。
