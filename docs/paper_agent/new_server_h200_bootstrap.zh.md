# New Server H200 Bootstrap Audit

更新时间：2026-07-06 UTC

## Bootstrap Verdict

`host_h200_available_sandbox_gpu_hidden`

本轮完成 H200 新服务器 bootstrap、复制 artifact 完整性审计和 GitHub push 验证。默认 Codex 沙箱中 GPU 不可见，但经批准的沙箱外检查确认 host H200 可用。后续 GPU 实验必须使用 approved unsandboxed/escalated execution path。

## Tier 1 H200 Reproduction Status

更新时间：2026-07-07 UTC

H200 core baselines 已全量重跑，compact audit 为：

- `analysis_outputs/h200_repro_audit_20260707_tier1_v2/`
- reproduction verdict：`h200_material_outcome_drift`

结果摘要：

| Run | Old A6000 | H200 | Delta | Paired H200 Wins/Losses |
|---|---:|---:|---:|---:|
| Control | `787/1033` | `787/1033` | `0` | `4/4` |
| Midcons | `795/1033` | `794/1033` | `-1` | `4/5` |
| Route2 | `801/1033` | `795/1033` | `-6` | `3/9` |
| V6 | `802/1033` | `796/1033` | `-6` | `3/9` |
| Local same-protocol CAL | `774/1033` | `769/1033` | `-5` | `4/9` |

研究者已接受当前 H200 rerun 结果作为新的 evidence base；`h200_material_outcome_drift` 继续记录为 reproducibility caveat，但不再作为 Controller V2 的 stop condition。H200 action bank、Controller V1 replay、CPU-only material drift triage 和 Controller V2 validation 已于 2026-07-07 UTC 完成；frozen test 继续保持 `sealed`、`test_evaluation_count=0`。

## H200 Action Bank And Controller V1 Replay（2026-07-07）

Artifacts：

- H200 action bank：`analysis_outputs/controller_action_bank_h200_20260707_tier1_offline/`
- H200 Controller V1 replay：`analysis_outputs/controller_validation_h200_20260707_v1_replay/`
- Old-vs-H200 action/controller audit：`analysis_outputs/h200_repro_audit_20260707_action_bank_v1/`

Action-bank coverage：

- train/calibration/validation tasks：`927`
- action rows：`4635`
- split rows：train `3225`，calibration `775`，validation `635`
- action counts：each of `KEEP_PRIMARY`, `EXPAND_16`, `EXPAND_24`, `EXPAND_32`, `EXPAND_48` has `927`
- test rows：`0`
- H200 pass count：`2506`
- H200 benefit/harm labels non-KEEP：`193/1237`

Controller V1 replay：

- selected controller：`probe_only` + `benefit_only`，threshold `999.0`
- validation result：`89/127`
- interventions：`0`
- wins/losses vs primary：`0/0`
- validation gate：failed
- test decision：`sealed`
- oracle action-bank upper bound：`106/127`，`17` wins / `0` losses

Old-vs-H200 comparison：

- action-bank outcome agreement：`94.95%`
- action-bank candidate-hash agreement：`80.13%`
- old vs H200 V1 selected validation：`90/127` vs `89/127`
- server migration verdict remains：`h200_material_outcome_drift`
- H200 evidence-base accepted：`true`
- H200 material drift no longer blocks Controller V2：`true`

## H200 Material Drift Triage（2026-07-07）

- Output：`analysis_outputs/h200_material_drift_triage_20260707_material_drift_triage/`
- Report：`analysis_outputs/h200_material_drift_triage_20260707_material_drift_triage/report.md`
- Triage verdict before evidence-base acceptance：material drift was material enough to require a decision; this stop condition is now superseded by `docs/paper_agent/h200_evidence_base_decision.zh.md`
- Core public-split flip rows：`53`
- Action-bank flip/label-change rows：`267`
- Row-level test details：suppressed
- Frozen test：`sealed`，`test_evaluation_count=0`

Interpretation：H200 drift remains small in absolute pass count but material for Route2/V6/CAL and action labels. The triage remains an audit artifact; the later evidence-base decision supersedes it as a stop condition and authorizes Controller V2 on H200 train/calibration/validation.

关键环境事实：

- `/proc/driver/nvidia/gpus/0000:22:00.0/information` 可见 `NVIDIA H200 NVL`，UUID 为 `GPU-c55d478c-6944-066f-7e9a-8af248e4f6f1`。
- 默认 Codex 沙箱内：`nvidia-smi` 和 `nvidia-smi -L` 均失败；`/dev/nvidia*` 不可见；`dllm_env` 中 `torch.cuda.is_available() = false`，`torch.cuda.device_count() = 0`。
- 沙箱外/approved host check：`nvidia-smi` 正常，driver `580.159.03`，CUDA driver `13.0`，GPU `NVIDIA H200 NVL` 空闲，`/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 可见。
- 沙箱外 `dllm_env` PyTorch：`torch.cuda.is_available() = true`，`torch.cuda.device_count() = 1`，GPU name `NVIDIA H200 NVL`。

因此，当前结论不是 host H200 invalid，而是 sandbox GPU device-node visibility limited。Tier 1 reruns 已通过 approved unsandboxed GPU command 完成，且出现 material outcome drift。

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
- `analysis_outputs/h200_bootstrap_20260705_103617/HOST_GPU_VISIBILITY_20260706.md`

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

1. Analyze material H200 drift before any Controller V2 work.
2. Decide whether to accept H200 results as the new evidence base or investigate environment/generation drift further.
3. Continue to keep frozen test sealed until validation gate conditions are met.
