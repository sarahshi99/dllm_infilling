# LLaDA-MoE Download And API Probe

Timestamp: 2026-06-10 19:41 CST

## Purpose

Prepare the next protocol-matched literature-backbone run for `inclusionAI/LLaDA-MoE-7B-A1B-Base`, which appears in LR-DLLM Table 3.

This is a download/API compatibility gate, not a pass-rate experiment.

## Literature Anchor

LR-DLLM Table 3 reports:

| Backbone | Reported baseline single-line | LR-DLLM single-line |
|---|---:|---:|
| LLaDA-MoE | `48.8` | `71.3` |

These are anchors only. A local claim requires a same-backbone local baseline/candidate pair.

## Current Local State

- Existing cache `/tmp/hf_lladamoe_20260609` only contained `refs/main`; no complete model directory existed.
- New local probe path: `/tmp/lladamoe_probe_20260610`.
- New cache path: `/tmp/hf_lladamoe_20260610`.
- `/tmp` free space before full weight download: about `50G`.

## Metadata/API Probe So Far

HF API reports:

- repo: `inclusionAI/LLaDA-MoE-7B-A1B-Base`
- gated: `false`
- architecture: `LLaDAMoEModel`
- `model_type`: `llada`
- weight shards: `3`
- safetensors metadata total parameters: `7,356,880,896`
- safetensors metadata total size: `14,713,761,792` bytes

Local small-file probe:

- `AutoConfig.from_pretrained(..., trust_remote_code=True)` passed.
- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` passed.
- tokenizer `mask_token=<|mask|>`, `mask_token_id=156895`.
- config has no `mask_token_id`, but the local runner resolves from `tokenizer.mask_token_id`, so this is likely compatible.
- empty `AutoModel.from_config(..., trust_remote_code=True)` passed and produced `LLaDAMoEModelLM`.

## Planned Full Download Command

Run in tmux so the download survives interruptions and has a durable log.

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
script -q -e -c "HF_HUB_DISABLE_XET=1 HF_HOME=/tmp/hf_lladamoe_20260610 /home/shx/miniconda3/envs/llmxy/bin/python -m huggingface_hub.commands.huggingface_cli download inclusionAI/LLaDA-MoE-7B-A1B-Base --cache-dir /tmp/hf_lladamoe_20260610 --local-dir /tmp/lladamoe_probe_20260610 --max-workers 1" logs/paper_agent/20260610_1941_lladamoe_full_download.log
```

## Actual Download Status

Update 2026-06-10 21:36 CST:

- The first `huggingface_hub` full-download attempt exited `0` but did not fetch weights because the network path was unavailable.
- A retry with explicit proxy began but was too slow, so it was stopped before completion.
- The active download uses `aria2c` with the configured proxy and writes directly into `/tmp/lladamoe_probe_20260610`.

Active tmux session:

```bash
lladamoe_aria2_20260610_1950
```

Active command:

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
aria2c --all-proxy=http://127.0.0.1:7890 --continue=true --auto-file-renaming=false --allow-overwrite=true --max-connection-per-server=8 --split=8 --min-split-size=16M --dir=/tmp/lladamoe_probe_20260610 --input-file=logs/paper_agent/20260610_1950_lladamoe_aria2_input.txt --summary-interval=60 --console-log-level=notice
```

Active log:

```bash
logs/paper_agent/20260610_1950_lladamoe_aria2_download.log
```

Important caution: `aria2c` preallocates the three shard files. The apparent `14G` total size is not completion evidence. Completion must be judged by the tmux session exiting successfully, no `.aria2` sidecar files remaining, and byte-size validation against `model.safetensors.index.json`.

## Success Criteria

- Download exits `0`.
- `/tmp/lladamoe_probe_20260610` contains all three safetensors shards and `model.safetensors.index.json`.
- Shard file sizes sum to the index/API expected size.
- Local config/tokenizer/model-class probe still passes after full download.
- No GPU experiment is launched until a tiny smoke action brief is written.

## Kill Criteria

Stop and investigate if download fails, shard size is incomplete, model class import fails, mask token is unresolved, `/tmp` free space becomes unsafe, or later GPU smoke shows OOM/verifier/canvas collapse.
