# Host GPU Visibility Correction

Status: `host_h200_available_sandbox_gpu_hidden`

Verification date: 2026-07-06 UTC

The earlier bootstrap audit correctly observed that the default Codex sandbox cannot see `/dev/nvidia*` and therefore reports `nvidia-smi` failure and `torch.cuda.is_available() = false`. Follow-up checks outside the sandbox show that the host H200 is healthy and idle.

## Default Sandbox

- `nvidia-smi`: fails with driver communication error.
- `/dev/nvidia*`: not visible.
- `dllm_env` PyTorch: `cuda_available = false`, `gpu_count = 0`.

## Approved Host / Unsandboxed Check

- `nvidia-smi`: succeeds.
- Driver: `580.159.03`.
- CUDA driver reported by `nvidia-smi`: `13.0`.
- GPU count: `1`.
- GPU: `NVIDIA H200 NVL`.
- UUID: `GPU-c55d478c-6944-066f-7e9a-8af248e4f6f1`.
- Memory: `143771 MiB`.
- Running GPU processes: none.
- `/dev/nvidia0`, `/dev/nvidiactl`, and `/dev/nvidia-uvm` are present outside the sandbox.
- `dllm_env` PyTorch outside the sandbox: `cuda_available = true`, `gpu_count = 1`, `gpu_names = ["NVIDIA H200 NVL"]`, CUDA runtime `12.1`.

Conclusion: the H200 host environment is usable, but GPU experiments must be launched through the approved unsandboxed/escalated execution path. The remaining blocker is not the NVIDIA driver; it is the default Codex sandbox's device-node visibility.
