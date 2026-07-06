# H200 Bootstrap Report

Timestamp: `20260705_103617`

## Bootstrap Verdict

`host_h200_available_sandbox_gpu_hidden`

H200 reproduction has not started yet. The default Codex sandbox hides `/dev/nvidia*`, so sandboxed `nvidia-smi` fails and sandboxed PyTorch reports `cuda_available=false`. Follow-up approved host checks outside the sandbox show the H200 is healthy and idle: `nvidia-smi` succeeds, `/dev/nvidia0`, `/dev/nvidiactl`, and `/dev/nvidia-uvm` are present, and `dllm_env` PyTorch reports `cuda_available=true`, `gpu_count=1`, `gpu_names=["NVIDIA H200 NVL"]`.

Detailed correction: `HOST_GPU_VISIBILITY_20260706.md`.

## Git

- Workspace: `/home/shx/projects/dllm_infilling/git_workspace`
- Branch: `codex/risk-controlled-dynamic-rescue`
- HEAD: `2b0662bfe9fdab787a5249dc9cbefea12d683af1`
- Remote: `git@github.com:sarahshi99/dllm_infilling.git`
- Working tree before bootstrap files: untracked `scripts/bootstrap_remote_10_98_36_183.sh`, `scripts/no_flash_attn/`
- GitHub verification: SSH auth and remote branch freshness verified on 2026-07-06 UTC. See `GITHUB_REMOTE_VERIFICATION.md`.

## Environment Summary

- Hostname: `server`
- OS: Ubuntu 22.04.5 LTS
- CPU: AMD EPYC 9J14, 384 logical CPUs
- RAM: 251 GiB
- Disk: `/home/shx/projects/dllm_infilling` has about 3.2T available
- Driver proc version: NVIDIA open kernel module `580.159.03`
- GPU proc identity: NVIDIA H200 NVL, UUID `GPU-c55d478c-6944-066f-7e9a-8af248e4f6f1`
- Sandbox `nvidia-smi`: failed, cannot communicate with driver
- Host/unsandboxed `nvidia-smi`: succeeds; H200 idle
- Sandbox `/dev/nvidia*`: none visible
- Host/unsandboxed `/dev/nvidia*`: `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`
- Python env: `/home/shx/miniconda3/envs/dllm_env`, Python 3.10.20
- PyTorch: `2.5.1+cu121`, CUDA runtime `12.1`, cuDNN `90100`
- Transformers: `4.38.2`
- Accelerate: `1.13.0`
- NumPy: `2.2.6`
- Triton: `3.1.0`

## Artifact Integrity

- Review manifest JSON parses.
- Frozen test lock JSON parses.
- Grouped split manifest JSON parses.
- Frozen test status remains `sealed`.
- Frozen test evaluation count remains `0`.
- Compact artifact hashes: `copied_artifact_hashes.csv`.

- V6 compact summary is present under `git_workspace/outputs_clean/full_route2_v6_short_override_gpu2_20260620_124754/summary.json`; outer copied `outputs_clean/` is not the canonical location for that run.

## Checkpoint

- Model: `GSAI-ML/LLaDA-8B-Base`
- Snapshot: `/home/shx/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/0f2787f2d87eac5eed8a087d5ecd24277e6255b2`
- Revision: `0f2787f2d87eac5eed8a087d5ecd24277e6255b2`
- Config/tokenizer/index hashes recorded in `environment_manifest.json` and `copied_artifact_hashes.csv`.
- Six safetensor shards are present through HF snapshot symlinks; resolved total size is `16031197112` bytes.
- Sampled shard hashes recorded for shard 1 and shard 6.

## Execution Condition

Full H200 baselines, action-bank rebuild, controller V1 replay, true-long replay, and Controller V2 were not started by this bootstrap audit. They should be launched only through approved unsandboxed/escalated GPU commands, not the default sandbox.
