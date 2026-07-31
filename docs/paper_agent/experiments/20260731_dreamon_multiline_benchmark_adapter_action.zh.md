# DreamOn MultiLine benchmark-only adapter action

日期：2026-07-31 UTC

准确标签：**DreamOn official-source MultiLine reproduction via benchmark-only adapter**。

## Equivalence boundary

该 adapter 仍直接调用 pinned `DreamLM/DreamOn@8a0a54918412eda9402a327646f7f067f7160ec8` 的同一 `MDMGenerator.batch_generate_with_expand_as_token` 与同一 `decode_one`。MultiLine profile 只改变：

1. dataset population contract：`HumanEval-MultiLineInfilling`；
2. immutable manifest / source-row namespace；
3. evaluator输入行与output/log目录；
4. candidate-key前缀，避免与SingleLine raw冲突。

model/checkpoint、每例prompt/suffix拼接、generator config、sampling、seed derivation、forward/token accounting、completion extraction 和 pinned evaluator调用不分叉。因此这是 benchmark-only adapter，不是新的 DreamOn 算法。Single H200仍只替代官方8进程样本分片；不称 exact official 8-GPU random-stream reproduction。

## Population / protocol

- full：project non-frozen MultiLine `5079 rows / 148 clusters`，manifest SHA256=`b18fe79cda00048b89e0d92983e1b3d90aa82d95c0b17b684847f2962d132f80`。
- smoke：从 immutable full manifest 顺序选取12个不同clusters；SHA256=`5bd2d19ba9b8c48318708c79319e306d2438404d7c13e9bfaf327de6a483ebf9`；不读source outcome、solution/tests/oracle length或sealed files。
- arms：initial=`4/8/16/32/64`，max=`64`，steps=`256`，temperature=`0.2`，top-p=`0.9`，entropy remasking，mask expansion，seed=`42` deterministic per candidate。
- frozen test=`sealed`，`test_evaluation_count=0`。

## Exact commands

Manifest：

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/build_dreamon_multiline_smoke_manifest.py \
  --full-manifest analysis_outputs/baseline_manifests_20260731_v1/lrdllm_common_multiline_nonfrozen_manifest.jsonl \
  --output-dir analysis_outputs/baseline_manifests_20260731_v1
```

CPU preflight：

```bash
PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages \
HF_MODULES_CACHE=/tmp/hf_modules_dreamon_20260731 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/shx/miniconda3/envs/dllm_env/bin/python experiments/dreamon_singleline_adapter.py \
  --source-root /tmp/dllm_infilling_protocol_audit_20260728/DreamOn \
  --evaluator-root /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff \
  --model-snapshot /home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194 \
  --model-profile dreamon --dataset-profile multiline \
  --manifest-jsonl analysis_outputs/baseline_manifests_20260731_v1/dreamon_multiline_technical_smoke12_manifest.jsonl \
  --full-manifest-jsonl analysis_outputs/baseline_manifests_20260731_v1/lrdllm_common_multiline_nonfrozen_manifest.jsonl \
  --manifest-summary-json analysis_outputs/baseline_manifests_20260731_v1/lrdllm_common_manifest_summary.json \
  --output-dir outputs_clean/dreamon_multiline_20260731_v1 \
  --min-gen-len 4 --mode smoke --seed 42 --progress-every 1 --preflight-only
```

Smoke / full：

```bash
ALLOW_SHARED_GPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  bash scripts/manual_launch_dreamon_multiline_20260731.sh smoke <L>

tmux new-session -d -s dreamon_ml5079_min<L>_20260731 \
  'cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1 && export ALLOW_SHARED_GPU=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; exec bash scripts/manual_launch_dreamon_multiline_20260731.sh full <L>'
```

Output=`outputs_clean/dreamon_multiline_20260731_v1/`；log=`logs/paper_agent/20260731_dreamon_multiline_min<L>.log`；GPU=`physical 0 / H200`。

## Gate / kill criteria

Success：manifest=`12 rows/12 clusters`、forbidden fields=`0`、preflight hashes/tokenizer通过；每个smoke=`12/12` exact unique、missing/duplicate/error/failure/accounting=`0`、resume no-op=`0` writes。adapter/tests/action/manifest必须先commit/push，再启动GPU smoke/full。

Kill：dataset/source/checkpoint/evaluator/hash不符；MultiLine profile进入不同per-example algorithm branch；OOM/ECC；missing/duplicate/error；forward/length accounting失败；任何sealed/outcome泄漏。RandomSpan不进入paper-faithful DreamOn表。
