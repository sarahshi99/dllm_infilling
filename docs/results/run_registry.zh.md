# Run Registry

该 registry 记录有意义的本地实验输出，同时避免把原始 `results.jsonl` 提交到普通 git。

完整机器可读版本见 `docs/results/run_registry.json`，英文表格见 `run_registry.md`。当前人类阅读时应重点关注以下 canonical runs：

| 运行 | 状态 | 样本数 | Pass | Rate | 模型 | 作用 |
|---|---|---:|---:|---:|---|---|
| `full_oracle_sl_20260410_145452` | `diagnostic` | 1033 | 900 | 87.12% | `GSAI-ML/LLaDA-8B-Base` | Oracle-length 上界参考，说明长度选择是核心瓶颈。 |
| `full_fixed_sl_20260410_161152` | `diagnostic` | 1033 | 478 | 46.27% | `GSAI-ML/LLaDA-8B-Base` | Fixed-length 下界 baseline。 |
| `full_cal_lite_v1_sl_20260415_194824` | `diagnostic` | 1033 | 718 | 69.51% | `GSAI-ML/LLaDA-8B-Base` | 早期 CAL-lite 大幅提升。 |
| `full_cal_lite_v2_alpha_008_sl_20260417_131256` | `superseded` | 1033 | 770 | 74.54% | `GSAI-ML/LLaDA-8B-Base` | 早期 alpha sweep 的强点。 |
| `full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_gpus23_20260519_175826` | `global_best` | 1033 | 787 | 76.19% | `GSAI-ML/LLaDA-8B-Base` | A6000 前的 union checkpoint。 |
| `full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_163529` | `env_control` | 1033 | 787 | 76.19% | `GSAI-ML/LLaDA-8B-Base` | A6000 control。 |
| `full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_221626` | `candidate` | 1033 | 795 | 76.96% | `GSAI-ML/LLaDA-8B-Base` | 当前 A6000 最佳 checkpoint。 |

Registry 的机器可读条目现包括 63 个 runs。除 canonical runs 外，其他 runs 主要用于记录失败方向、参数敏感性和被 superseded 的实验路线，避免重复消耗 GPU。

DreamOn V2-Hard-v2 scoped runs：

| 运行 | 状态 | Rows | Completed | Pass | Compile | 结论 |
|---|---|---:|---:|---:|---:|---|
| `dreamon_progressive_v2_hard_v2_smoke5_20260801` | `diagnostic_gate_passed` | 5 | 5 | 2 | 3 | smoke 协议门禁通过。 |
| `dreamon_progressive_v2_hard_v2_pilot30_20260801` | `stopped_pilot_gate_failed` | 30 | 25 | 11 | 19 | 5 个 exact cycle，未达到 completion/compile 门禁；full 未启动。 |

DreamOn V3 scoped runs：

| 运行 | 状态 | Rows | Completed | Pass | Compile | 结论 |
|---|---|---:|---:|---:|---:|---|
| `v3_hard_budgeted_pilot30_20260802` | `engineering_pass_quality_fail` | 30 | 30 | 14 | 23 | 官方累计 budget control；Full 未授权。 |
| `v3_hard_budgeted_full642_posthoc_20260802` | `completed_posthoc_diagnostic` | 642 | 642 | 281 | 538 | 用户后续明确授权；不改变原始 Pilot gate，非 held-out。 |
| `v3_hard_budgeted_nonempty_oracle_pilot30_20260802` | `full_authorized` | 30 | 30 | 18 | 25 | oracle structural pilot。 |
| `v3_hard_budgeted_nonempty_oracle_full642_20260802` | `completed_oracle_diagnostic` | 642 | 642 | 280 | 471 | 不可部署、非 held-out。 |
| `v3_pure_newline_blank30_manifest_20260803` | `sealed_posthoc_mechanism_manifest` | 30 | 30 | 3 (A) | 22 (A) | 仅由 A trace 构造；C0/C targeted gate population。 |
| `v3_c0_pure_newline_veto_smoke5_20260803` | `engineering_gate_passed` | 5 | 5 | 1 | 3 | C0 deterministic/resume smoke。 |
| `v3_c_nonconsuming_blankline_smoke5_20260803` | `engineering_gate_passed` | 5 | 5 | 4 | 4 | C deterministic/resume smoke。 |
| `v3_c0_pure_newline_veto_pure30_20260803` | `engineering_pass_quality_fail` | 30 | 30 | 5 | 22 | 未达冻结 6/30 Full 门槛；未跑 Full。 |
| `v3_c_nonconsuming_blankline_pure30_20260803` | `full_quality_gate_passed` | 30 | 30 | 14 | 25 | 自动授权 C Full。 |
| `v3_c0_pure_newline_veto_pilot30_20260803` | `engineering_gate_passed_full_not_authorized` | 30 | 30 | 15 | 24 | regression pilot；C0 Pure30 门槛失败。 |
| `v3_c_nonconsuming_blankline_pilot30_20260803` | `engineering_gate_passed` | 30 | 30 | 17 | 25 | regression pilot。 |
| `v3_c_nonconsuming_blankline_full642_20260803` | `completed_gate_authorized_development_diagnostic` | 642 | 642 | 292 | 541 | 97 exact；C vs A 12/1 wins/losses；非 held-out。 |
