# Model Generalization Registry

This registry records compact metadata for local cross-model infilling runs.
Smoke runs are excluded by default; raw `results.jsonl` files remain local.

| Run | Family | Model | Samples | Pass | Rate | Mask Source | Raw Path |
|---|---|---|---:|---:|---:|---|---|
| `full_lcas_v3_dream-coder-base_20260512_154355` | `20260512_114917_lcas_v3_full` | `Dream-org/Dream-Coder-v0-Base-7B` | 1033 | 0 | 0.00% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_dream-coder-base_20260512_154355` |
| `full_cal_lite_base_alpha010_cap24_official_canvas_20260513_232721` | `20260513_dreamcoder_official_full` | `Dream-org/Dream-Coder-v0-Base-7B` | 1033 | 825 | 79.86% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_base_alpha010_cap24_official_canvas_20260513_232721` |
| `full_cal_lite_base_alpha020_official_canvas_20260513_213936` | `20260513_dreamcoder_official_full` | `Dream-org/Dream-Coder-v0-Base-7B` | 1033 | 818 | 79.19% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_base_alpha020_official_canvas_20260513_213936` |
| `full_lcas_v3_dream-coder-instruct_20260512_162945` | `20260512_114917_lcas_v3_full` | `Dream-org/Dream-Coder-v0-Instruct-7B` | 1033 | 1 | 0.10% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_dream-coder-instruct_20260512_162945` |
| `full_cal_lite_instruct_alpha010_cap24_official_canvas_20260513_232724` | `20260513_dreamcoder_official_full` | `Dream-org/Dream-Coder-v0-Instruct-7B` | 1033 | 848 | 82.09% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_instruct_alpha010_cap24_official_canvas_20260513_232724` |
| `full_cal_lite_instruct_alpha020_official_canvas_20260513_213934` | `20260513_dreamcoder_official_full` | `Dream-org/Dream-Coder-v0-Instruct-7B` | 1033 | 785 | 75.99% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_instruct_alpha020_official_canvas_20260513_213934` |
| `full_lcas_v3_llada-instruct_resume_20260512_141851` | `20260512_114917_lcas_v3_full` | `GSAI-ML/LLaDA-8B-Instruct` | 1033 | 817 | 79.09% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_llada-instruct_resume_20260512_141851` |
