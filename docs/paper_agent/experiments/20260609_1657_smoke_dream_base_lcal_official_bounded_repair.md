# Dream Base LCAL Official Bounded-Repair Smoke

Timestamp: 2026-06-09 16:57 CST

## Purpose

Validate the next literature backbone, `Dream-org/Dream-v0-Base-7B`, after `apple/DiffuCoder-7B-Base` was blocked by HuggingFace mirror metadata resolution. This smoke verifies model loading from the local Git/LFS checkout, Dream-style fixed-canvas generation, verifier execution, and output schema.

## Download / Cache Note

`huggingface_hub` with `HF_ENDPOINT=https://hf-mirror.com` failed for uncached models because mirror metadata/HEAD resolution raised `FileMetadataError`. A Git/LFS path was used instead:

- Remote URL: `https://hf-mirror.com/Dream-org/Dream-v0-Base-7B`.
- Git reported a redirect to `https://huggingface.co/Dream-org/Dream-v0-Base-7B/`.
- Local checkout: `/tmp/dream_base_repo_probe_20260609`.
- LFS files: four safetensors shards, total model size about `15.23GB`.

This is a practical cache workaround, not a protocol result.

## Command

The smoke was launched by watcher script `clean_scripts/wait_and_run_dream_base_smoke_20260609.sh` after GPU 2 became free.

Log:

`logs/paper_agent/20260609_1542_smoke_dream_base_lcal_official_bounded_repair_gpu2_unsandboxed.log`

Output:

`/home/shx/projects/dllm_infilling/outputs_clean/smoke_dream_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_165727`

## Result

| Model | Run | Rows | Pass | Avg sec/sample incl. probe | Selected/oracle |
|---|---|---:|---:|---:|---|
| `/tmp/dream_base_repo_probe_20260609` | `lcal_official_bounded_repair` smoke | `2` | `2/2` | `3.1321` | exact on both rows |

Additional checks:

- `decode_backend`: `dreamcoder_native_diffusion_generate_fixed_canvas`.
- `canvas_format`: `bos_prefix_masks_suffix_eos`.
- `mask_length_source`: `lcal_official_bounded_repair`.
- `official_repair_trigger_count`: `0/2`.
- Verifier ran successfully; both rows passed.

## Interpretation

The Dream-7B runner/API/canvas/verifier contract is compatible enough to launch a local same-backbone full-run pair:

1. `cal_lite` local baseline.
2. `lcal_official_bounded_repair` current method.

No literature claim should be made from the smoke. LR-DLLM Dream-7B `76.7` and DreamOn Dream-7B `88.6` remain literature anchors only.
