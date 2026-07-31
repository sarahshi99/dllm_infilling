# DreamCoder MultiLine controls under DreamOn sampling/decoder

日期：2026-07-31 UTC

## Scope / label

准确标签：**DreamCoder Fixed4/8/16/32/64 under DreamOn sampling/decoder, MultiLine**。这是 `DreamOn official-source MultiLine reproduction via benchmark-only adapter` 的 common-protocol local control，不是 DreamOn、official CAL、LR-DLLM equal-compute control 或新的方法。

复用已冻结的 `experiments/dreamon_singleline_adapter.py`；组合 `model-profile=dreamcoder_fixed` 与 `dataset-profile=multiline`。该组合只把相同 pinned `MDMGenerator`、sampling、seed derivation、accounting 和 evaluator应用到 DreamCoder checkpoint 与 MultiLine manifest；每个 FixedL 设 `min_gen_len=max_gen_len=L`，保留 released source EOS contraction。

## Frozen protocol / manifests

- checkpoint：`Dream-org/Dream-Coder-v0-Base-7B@2346ccd3be517d0d314152b988a3b9bafa7d6d63`；cached snapshot无license metadata，发布前保留风险标记。
- source：`DreamLM/DreamOn@8a0a54918412eda9402a327646f7f067f7160ec8`；evaluator=`88062ff9859c875d04db115b698ed4b0f0395170`。
- full：MultiLine project non-frozen `5079 rows / 148 clusters`，manifest SHA256=`b18fe79cda00048b89e0d92983e1b3d90aa82d95c0b17b684847f2962d132f80`。
- smoke：outcome-blind `12 rows / 12 clusters`，SHA256=`5bd2d19ba9b8c48318708c79319e306d2438404d7c13e9bfaf327de6a483ebf9`。
- lengths=`4/8/16/32/64`；steps=`256`；temperature=`0.2`；top-p=`0.9`；entropy remasking；seed=`42`。每个 FixedL 复用对应 DreamOn dynamic minL arm 的 per-row seed key。
- raw append-only，successful candidate keys dedup；failure journal和progress manifest独立。Frozen test=`sealed`，`test_evaluation_count=0`。

## Exact commands

CPU preflight：

```bash
PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages \
HF_MODULES_CACHE=/tmp/hf_modules_dreamon_20260731 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/dreamon_singleline_adapter.py \
  --source-root /tmp/dllm_infilling_protocol_audit_20260728/DreamOn \
  --evaluator-root /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff \
  --model-snapshot /home/shx/.cache/huggingface/hub/models--Dream-org--Dream-Coder-v0-Base-7B/snapshots/2346ccd3be517d0d314152b988a3b9bafa7d6d63 \
  --model-profile dreamcoder_fixed --dataset-profile multiline \
  --manifest-jsonl analysis_outputs/baseline_manifests_20260731_v1/dreamon_multiline_technical_smoke12_manifest.jsonl \
  --full-manifest-jsonl analysis_outputs/baseline_manifests_20260731_v1/lrdllm_common_multiline_nonfrozen_manifest.jsonl \
  --manifest-summary-json analysis_outputs/baseline_manifests_20260731_v1/lrdllm_common_manifest_summary.json \
  --output-dir outputs_clean/dreamcoder_dreamon_fixed_multiline_20260731_v1 \
  --min-gen-len 64 --mode smoke --seed 42 --progress-every 1 --preflight-only
```

Smoke / resume：

```bash
ALLOW_SHARED_GPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash scripts/manual_launch_dreamcoder_dreamon_fixed_multiline_20260731.sh smoke <L>
```

Full 独立 tmux：

```bash
tmux new-session -d -s dreamcoder_dreamon_fixed<L>_ml5079_20260731 \
  'cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1 && export ALLOW_SHARED_GPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; exec bash scripts/manual_launch_dreamcoder_dreamon_fixed_multiline_20260731.sh full <L>'
```

GPU=`physical index 0 / single NVIDIA H200 NVL`；env=`dllm_env`。Output=`outputs_clean/dreamcoder_dreamon_fixed_multiline_20260731_v1/`；log=`logs/paper_agent/20260731_dreamcoder_dreamon_fixed_multiline<L>.log`。Budget=`5×12 smoke + 5×5079 full`；优先 Fixed64 和已完成 DreamOn arm 对应长度，按真实free-memory gate排队，不抢占已有任务。

## Success gate / kill criteria

先 commit/push 本 action、launcher、test 与 matrix；之后每个 smoke 必须 `12/12` exact unique、missing/duplicate/error/failure/accounting=`0`，resume `new_rows_written=0`。只有 gate 通过才启动对应 full。

Kill/stop：source/checkpoint/evaluator/manifest/hash不符；fixed key与对应 dynamic seed key不能配对；OOM/ECC；missing/duplicate/error；forward/token/length accounting失败；任何 frozen/evaluator-outcome泄漏。资源型OOM只阻塞该arm并保留journal；其他 baseline继续。
