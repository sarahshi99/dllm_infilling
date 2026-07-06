# New Server H200 Bootstrap Audit

更新时间：2026-07-06 UTC

## Verdict

`h200_environment_invalid`

本轮只完成 H200 新服务器 bootstrap 和复制 artifact 完整性审计。由于当前会话中 GPU 环境不可用，未启动任何 1033-row H200 full rerun、action bank rebuild、Controller V1 replay、true-long replay 或 Controller V2。

阻塞原因：

- `/proc/driver/nvidia/gpus/0000:22:00.0/information` 可见 `NVIDIA H200 NVL`，UUID 为 `GPU-c55d478c-6944-066f-7e9a-8af248e4f6f1`。
- `nvidia-smi` 和 `nvidia-smi -L` 均失败：无法与 NVIDIA driver 通信。
- `/dev/nvidia*` 设备节点当前不可见，只有 `/dev/nvidia-caps`。
- `dllm_env` 中 PyTorch 可导入，但 `torch.cuda.is_available() = false`，`torch.cuda.device_count() = 0`。

因此，按迁移协议必须停止 GPU 复现实验。当前结论是环境无效，而不是 H200 数值复现失败。

## Workspace And Git

- 实际仓库根目录：`/home/shx/projects/dllm_infilling/git_workspace`
- 外层目录：`/home/shx/projects/dllm_infilling` 中的 `.git` 是空目录，占位而非有效 Git 仓库。
- Branch：`codex/risk-controlled-dynamic-rescue`
- HEAD：`2b0662bfe9fdab787a5249dc9cbefea12d683af1`
- Remote：`git@github.com:sarahshi99/dllm_infilling.git`
- `git fsck --no-progress`：退出码 `0`，仅报告 dangling blobs。
- 工作区原有未跟踪项：`scripts/bootstrap_remote_10_98_36_183.sh`、`scripts/no_flash_attn/`。未删除、未覆盖。

GitHub 远端验证已恢复：

- `ssh -T git@github.com` 返回 `Hi sarahshi99/dllm_infilling! You've successfully authenticated, but GitHub does not provide shell access.`
- `git ls-remote origin refs/heads/codex/risk-controlled-dynamic-rescue` 返回 `2b0662bfe9fdab787a5249dc9cbefea12d683af1`。
- 远端 source-of-truth branch 与本地 HEAD 一致。
- verification report：`analysis_outputs/h200_bootstrap_20260705_103617/GITHUB_REMOTE_VERIFICATION.md`

## Environment Manifest

机器可读记录：

- `analysis_outputs/h200_bootstrap_20260705_103617/environment_manifest.json`
- `analysis_outputs/h200_bootstrap_20260705_103617/report.md`
- `analysis_outputs/h200_bootstrap_20260705_103617/conda_environment.yml`
- `analysis_outputs/h200_bootstrap_20260705_103617/pip_freeze.txt`
- `analysis_outputs/h200_bootstrap_20260705_103617/copied_artifact_hashes.csv`

摘要：

- Hostname：`server`
- OS：Ubuntu 22.04.5 LTS
- CPU：AMD EPYC 9J14，384 logical CPUs
- RAM：251 GiB
- Disk：项目分区约 3.2T available
- NVIDIA kernel module：`580.159.03`
- Python env：`/home/shx/miniconda3/envs/dllm_env`
- Python：3.10.20
- PyTorch：`2.5.1+cu121`
- Transformers：`4.38.2`
- Accelerate：`1.13.0`
- NumPy：`2.2.6`
- Triton：`3.1.0`
- Tokenizers：`0.15.2`
- CUDA runtime reported by PyTorch：`12.1`
- cuDNN reported by PyTorch：`90100`
- TF32：matmul `false`，cuDNN `true`

Base Conda 环境中的 `torch` 不能导入，报缺少 `libcudnn.so.9`；后续实验必须使用 `dllm_env` 或重新建立固定环境。

## Checkpoint And Dataset

主 checkpoint：

- Model：`GSAI-ML/LLaDA-8B-Base`
- HF snapshot：`/home/shx/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/0f2787f2d87eac5eed8a087d5ecd24277e6255b2`
- Revision：`0f2787f2d87eac5eed8a087d5ecd24277e6255b2`
- Shards：6 个 safetensors shard，resolved total size `16031197112` bytes
- Config SHA256：`5f99fefe855fdb5100bb6cadb57bdb09fae723ad54811f95c00ecacf29d58a6a`
- Tokenizer SHA256：`ee1ef8e5f6d9493ac25480b7b7337ff5d2c1b946190afff18d12c86ca738ae00`
- Tokenizer config SHA256：`6e9f41633217287fcf9a58890efb26e91e905bd6ae2234b534b65e0c36f4dd3c`
- Checkpoint index SHA256：`28b4ec27206e42e7ade630450e6ce618bd197acf34d35120e8e86d2bb910a408`
- Sampled shard hashes：shard 1 `4c0652913997c26851c51d5881918a93b2d502c0eeabcf3fe409800441ebc385`；shard 6 `3a2c91cf7aac23ca84f48d4fdabf58ffd050e9f56b9c94b7962a3d3f5c35e3c6`

Tokenizer probe：

- `mask_token_id = 126336`
- `<|mdm_mask|>` resolves to `126336`
- `tokenizer.mask_token` attribute is `None`

Dataset cache:

- HF repo cache includes `loubnabnl/humaneval_infilling`
- Local datasets cache contains `HumanEval-SingleLineInfilling`

## Copied Artifact Integrity

JSON checks passed:

- `docs/paper_agent/review_manifest.latest.json`
- `analysis_outputs/frozen_controller_20260703_phase2_freeze/test_lock.json`
- `analysis_outputs/grouped_split_20260702_accel2/split_manifest.json`

Frozen test integrity:

- `test_status = sealed`
- `test_evaluation_count = 0`

Hash inventory:

- `analysis_outputs/h200_bootstrap_20260705_103617/copied_artifact_hashes.csv`

The V6 compact summary is present at `git_workspace/outputs_clean/full_route2_v6_short_override_gpu2_20260620_124754/summary.json`. The outer copied `outputs_clean/` is not the canonical location for that run, so the inventory records the repo-relative path.

## Required Next Steps

1. Fix system GPU visibility so `nvidia-smi` works and `/dev/nvidia0`, `/dev/nvidiactl`, and `/dev/nvidia-uvm` are visible.
2. Re-run local CUDA probe in `dllm_env` and require `torch.cuda.is_available() = true`.
3. Keep the focused bootstrap audit commit pushed before any GPU rerun.
4. Only after GPU visibility is valid, start Tier 1 H200 full reruns and subsequent controller work.
