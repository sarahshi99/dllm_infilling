# 模型泛化 Registry

该 registry 记录本地 cross-model infilling runs 的紧凑元数据。
默认排除 smoke runs；原始 `results.jsonl` 仍保留在本地。

| 运行 | Family | 模型 | 样本数 | Pass | Rate | Mask Source | 原始路径 |
|---|---|---|---:|---:|---:|---|---|
| `full_lcas_v3_dream-coder-base_20260512_154355` | `20260512_114917_lcas_v3_full` | `Dream-org/Dream-Coder-v0-Base-7B` | 1033 | 0 | 0.00% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_dream-coder-base_20260512_154355` |
| `full_cal_lite_base_alpha010_cap24_official_canvas_20260513_232721` | `20260513_dreamcoder_official_full` | `Dream-org/Dream-Coder-v0-Base-7B` | 1033 | 825 | 79.86% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_base_alpha010_cap24_official_canvas_20260513_232721` |
| `full_cal_lite_base_alpha020_official_canvas_20260513_213936` | `20260513_dreamcoder_official_full` | `Dream-org/Dream-Coder-v0-Base-7B` | 1033 | 818 | 79.19% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_base_alpha020_official_canvas_20260513_213936` |
| `full_lcas_v3_dream-coder-instruct_20260512_162945` | `20260512_114917_lcas_v3_full` | `Dream-org/Dream-Coder-v0-Instruct-7B` | 1033 | 1 | 0.10% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_dream-coder-instruct_20260512_162945` |
| `full_cal_lite_instruct_alpha010_cap24_official_canvas_20260513_232724` | `20260513_dreamcoder_official_full` | `Dream-org/Dream-Coder-v0-Instruct-7B` | 1033 | 848 | 82.09% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_instruct_alpha010_cap24_official_canvas_20260513_232724` |
| `full_cal_lite_instruct_alpha020_official_canvas_20260513_213934` | `20260513_dreamcoder_official_full` | `Dream-org/Dream-Coder-v0-Instruct-7B` | 1033 | 785 | 75.99% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_instruct_alpha020_official_canvas_20260513_213934` |
| `full_lcas_v3_llada-instruct_resume_20260512_141851` | `20260512_114917_lcas_v3_full` | `GSAI-ML/LLaDA-8B-Instruct` | 1033 | 817 | 79.09% | `cal_lite` | `/home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_llada-instruct_resume_20260512_141851` |

解释：Dream-Coder official-canvas runs 的分数较高，但 protocol 与 LLaDA LCAL 主线不同；LCAS v3 Dream-Coder 的 `0/1` pass 结果更可能是 prompt/canvas mismatch，而不是模型能力结论。
