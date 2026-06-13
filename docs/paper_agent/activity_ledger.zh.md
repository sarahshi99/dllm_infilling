# Paper-Agent Activity Ledger

## 2026-06-12 19:31 CST

- action：完成 trace-long-rescue 的 Task 4/5 full trace collection 和 offline route analysis。
- evidence：previous trace 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`；midcons trace 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`；route analysis 目录 `analysis_outputs/trace_long_rescue_llada_base_prev_20260612_192611` 和 `analysis_outputs/trace_long_rescue_llada_base_midcons_20260612_192611`；report `analysis_outputs/trace_long_rescue_report_20260612`。
- result：previous trace run 验证为 `769/1033 = 74.44%`，有 `35257` trace rows；current `midcons` trace run 验证为 `795/1033 = 76.96%`，有 `35768` trace rows。Route 1/2 在两个 trace sources 上均触发 `0` 行；Route 3 因无 trace signal 而停止。所有 routes 的 Gate A/B 均失败。
- interpretation：这是当前 trace-only / risk-controlled route family 的 negative diagnostic evidence。不应从这批 traces 启动 route-specific GPU policy full run。
- next：policy-runner 工作在这里停止；除非新的 action brief 定义更强 trace feature family 或不同 route。

## 2026-06-12 18:10 CST

- action：验证 previous local method full trace run 后，启动 current `midcons` full trace run，并保持它是当前唯一 trace-long-rescue GPU job。
- evidence：previous 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`；current tmux session `trace_llada_base_midcons_20260612`；current 日志 `logs/paper_agent/20260612_full_trace_llada_base_midcons_gpu3.log`；current 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`。
- result：previous trace verification 通过：日志 exit `0`、`1033` 个 valid result rows、`35257` 个 trace rows 且覆盖 `1033` 个 task ids、`summary.json` 存在，pass count 为 `769/1033 = 74.44%`。Current `midcons` early health 通过，在 GPU3 上推进到至少 `26/1033` rows 且 traces 非空。
- next：持续监控 `trace_llada_base_midcons_20260612` 到完成；验证 exit `0`、`summary.json`、`1033` valid rows 和非空 linked traces 后，才运行 offline Route 1/2/3 analysis。

## 2026-06-12 17:17 CST

- action：启动 Task 4 previous local method full trace collection，并保持它是 trace-long-rescue 当前唯一 GPU job。
- evidence：tmux session `trace_llada_base_prev_20260612`；日志 `logs/paper_agent/20260612_full_trace_llada_base_cal_lite_lcas_v3b_gpu2.log`；输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`。
- result：early health 通过。该 run 已加载 `1033` 个任务，创建 `config.json`、`results.jsonl` 和非空 `step_traces.jsonl`，并推进到至少 `204/1033` rows、`6989` trace rows，GPU2 正常工作。
- next：持续监控该 tmux session 到完成；验证 exit `0`、`summary.json`、`1033` valid rows 和非空 traces 后，才允许启动 current `midcons` trace run。

## 2026-06-12 16:59 CST

- action：在启动任何 GPU full trace run 之前，按串行模式完成 trace-long-rescue Task 1/2/3。
- evidence：`docs/paper_agent/experiments/20260611_trace_long_rescue_full_plan.md` 和 `docs/paper_agent/current_action.md` 通过 markdown diff hygiene；新增 `analysis/trace_long_rescue_features.py`、`analysis/analyze_trace_long_rescue_routes.py`、`analysis/print_trace_long_rescue_report.py` 和 `tests/test_trace_long_rescue_features.py`。
- result：focused verification 通过：`/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_long_rescue_features.py` 输出 `Ran 6 tests` / `OK`；三个 trace-long-rescue analysis scripts 的 py_compile 通过；Task 1/2/3 相关文件的 `git diff --check` 通过。
- interpretation：offline trace analysis tooling 已就绪，但尚未创建 route-specific policy runner，也尚未启动 GPU trace run。
- next：先打印 exact command/log/output/success/kill criteria，然后只启动 GPU `2` 上的 previous local method full trace run；完成 fresh verification 后才启动 GPU `3` 上的 current `midcons` trace run。

## 2026-06-11 14:42 CST

- action：监控并收口完整 `inclusionAI/LLaDA-MoE-7B-A1B-Base` local same-backbone pair，生成 pairwise/bucket analysis，并同步到 GitHub 可读实验文档。
- evidence：baseline 日志 `logs/paper_agent/20260611_1126_full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared.log` 和 candidate 日志 `logs/paper_agent/20260611_1126_full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared.log` 都以 `COMMAND_EXIT_CODE="0"` 结束。Baseline 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112719` 和 candidate 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112740` 都有 `1033` 个 valid rows、`0` 个 malformed rows 和 `summary.json`。Pairwise analysis 为 `analysis_outputs/lladamoe_full_pair_20260611_1438`。
- result：candidate `801/1033 = 77.54%`，local `cal_lite` LCAS-v3b baseline `777/1033 = 75.22%`；pairwise 为 `31` wins、`7` losses、`770` tie-pass、`225` tie-fail；avg sec including probe 为 `10.6107`，baseline 为 `8.7025`。
- diagnostics：oracle bucket delta 为 `<=8 +9`、`9-12 +5`、`13-16 +4`、`17-24 +6`、`25+ 0`。True-long trigger precision 仍偏弱：`official_repair_true_long_precision = 11.54%`，`official_long_suspicion_true_long_precision = 40.00%`，`official_mid_rescue_true_long_precision = 13.04%`；`under_select_rate_17plus = 91.15%`。
- interpretation：这是当前最强 local transfer result，可支持 LLaDA-MoE 上的 local same-backbone improvement claim，但不是 external SOTA claim。文献 anchors 必须继续与 protocol-matched local controls 分开。
- next：没有新 action brief 前不要启动另一个 GPU full run。下一步研究应先设计更强 length signal 或 ablation plan，或明确选择尚未覆盖的 checkpoint variant。

## 2026-06-11 11:30 CST

- action：完成 `inclusionAI/LLaDA-MoE-7B-A1B-Base` smoke gate，并启动 full local same-backbone baseline/candidate pair。
- evidence：下载恢复日志 `logs/paper_agent/20260611_1100_lladamoe_aria2_resume.log` exit `0`；smoke baseline 输出 `/home/shx/projects/dllm_infilling/outputs_clean/smoke_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112427`，candidate 输出 `/home/shx/projects/dllm_infilling/outputs_clean/smoke_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112452`，两者均为 `2/2` 且日志 exit `0`。Full baseline tmux session `lladamoe_full_base_20260611_1126`，输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112719`；full candidate tmux session `lladamoe_full_candidate_20260611_1126`，输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112740`。
- result：full run 正在运行，尚无最终 pass rate。早期健康检查显示 baseline 约 `16/1033`、candidate 约 `11/1033`，GPU2/GPU3 各约 `15-16GB` 显存，无 OOM/import failure。
- diagnostics：LLaDA-MoE 需要 `llmxy` Transformers `4.52.3` 的 `modeling_rope_utils`，但 `dllm_env` 的 `flash_attn_2_cuda` 因 `GLIBC_2.32` 导入失败；当前命令使用 `/tmp/no_flash_attn` shim 隐藏 flash-attn，同时保留 `dllm_env` verifier 依赖。
- next：持续监控两个 tmux sessions 到结束；完成后验证 exit code、`1033` 行、summary，再生成 pairwise/bucket/runtime analysis 和本地/历史/文献对比表。

## 2026-06-10 21:36 CST

- action：继续推进 `inclusionAI/LLaDA-MoE-7B-A1B-Base` download/API gate，并在启动任何 GPU 实验前写好 smoke-pair brief。
- evidence：当前 tmux session 为 `lladamoe_aria2_20260610_1950`；当前日志为 `logs/paper_agent/20260610_1950_lladamoe_aria2_download.log`；download/probe brief 为 `docs/paper_agent/experiments/20260610_1941_lladamoe_download_api_probe.md`；smoke plan 为 `docs/paper_agent/experiments/20260610_2136_lladamoe_smoke_pair.md`。
- result：下载仍在进行中。当前本地 shards 看似已有 `14G`，但这是 `aria2c` 预分配；`.aria2` sidecar 文件仍存在，因此 checkpoint 尚未完整。
- next：继续监控到 tmux session 退出且 `.aria2` 文件消失，然后校验 shard bytes，运行本地 API/load probe，之后才启动两个 2-sample smokes。

## 2026-06-10 19:25 CST

- action：持续监控 `GSAI-ML/LLaDA-1.5` full local same-backbone pair 到完成，验证 row counts 和 exit codes，从 raw rows 重新计算 pairwise/bucket/runtime metrics，并更新 GitHub 可读结果文档。
- evidence：baseline 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_cal_lite_lcas_v3b_gpu2_shared_20260610_172705`；candidate 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_lcal_official_bounded_repair_gpu3_shared_20260610_172720`；pairwise analysis `analysis_outputs/llada15_full_pair_20260610_1923`；最终 brief `docs/paper_agent/experiments/20260610_1735_full_llada15_parallel_baseline_candidate.md`。
- result：candidate 为 `818/1033 = 79.19%`，本地同 backbone `cal_lite` LCAS-v3b baseline 为 `817/1033 = 79.09%`；pairwise 为 `18` wins、`17` losses，net `+1` task。Avg sec including probe 为 `6.6453`，baseline 为 `5.4224`。
- diagnostics：candidate 在 oracle `<=8` 净损失 `6` 个任务，在 `9-12` 持平，在 oracle `>=13` 合计净增 `+7`。Official repair 触发 `110/1033 = 10.65%` 行，但 true-long precision 只有 `10.91%`；`110` 个 triggers 中 `82` 个是 oracle `<=8`。
- interpretation：这是 near-tie / slight local positive，不是强 claim upgrade。当前 official-CAL repair family 仍然不是精确的 true-long detector。
- next：继续 literature-backbone matrix，下一个为 `inclusionAI/LLaDA-MoE-7B-A1B-Base`；先通过已配置 proxy/mirror 下载并 probe checkpoint，然后跑 tiny baseline/candidate smokes，通过后才启动 full pair。

## 2026-06-10 17:35 CST

- action：在两个 2-sample smokes 通过后，启动 full `GSAI-ML/LLaDA-1.5` local same-backbone pair。
- evidence：full-run brief `docs/paper_agent/experiments/20260610_1735_full_llada15_parallel_baseline_candidate.md`；baseline log `logs/paper_agent/20260610_1735_full_llada15_cal_lite_lcas_v3b_gpu2_shared.log`；candidate log `logs/paper_agent/20260610_1735_full_llada15_lcal_official_bounded_repair_gpu3_shared.log`。
- result：早期健康检查通过。两边都加载了本地 LLaDA-1.5 shards，读取 `1033` 个 HumanEval-SingleLineInfilling tasks，并开始 decode。虽然卡上已有共享任务，加载后 GPU2/GPU3 总显存仍约 `25.8GB` / `25.9GB`，没有 OOM。
- next：持续监控两个 tmux sessions 到结束，然后做 row-count/JSON sanity、raw-row pairwise/bucket/runtime analysis；在此之前不解释性能 claim。

## 2026-06-09 21:55 CST

- action：完成 `GSAI-ML/LLaDA-1.5` metadata/API/local-weight probe，并把结果写入文档。
- evidence：本地模型路径 `/tmp/llada15_probe_20260609`；probe brief `docs/paper_agent/experiments/20260609_2105_llada15_download_api_probe.md`；当前行动说明 `docs/paper_agent/current_action.md`。
- result：通过 proxy 直连 HuggingFace 可用；`hf-mirror.com` 对该 repo 会跳回 HuggingFace，且 `huggingface_hub` mirror 模式失败。6 个 safetensors shards 已全部下载，并按 HF API/index 做 byte-size 校验，总大小 `16,031,197,144` bytes。Local `AutoConfig`、`AutoTokenizer`、`AutoModel.from_config` 和 CPU/local checkpoint loading 均通过。LLaDA-1.5 使用 `LLaDAModelLM`、`model_type=llada`，config `mask_token_id=126336`，tokenizer 中 `<|mdm_mask|>` 也解析为 `126336`；但 `tokenizer.mask_token` 本身是 `None`。
- caveat：这不是 GPU smoke，也不是性能结果。最新 `nvidia-smi` 显示 GPU `2` 和 `3` 被占用，且当前 sandboxed Python 报告 `torch.cuda.is_available() == False`。
- next：等 GPU `2` 或 `3` 释放且允许沙箱外 CUDA/verifier 执行后，先跑 tiny candidate 和同 backbone `cal_lite` baseline smokes，再考虑 full LLaDA-1.5 pair。

## 2026-06-09 21:05 CST

- action：将刚完成的 DiffuCoder-Base full local pair 同步到恢复入口，并为下一个 literature backbone `GSAI-ML/LLaDA-1.5` 写入行动说明。
- evidence：DiffuCoder full-pair brief `docs/paper_agent/experiments/20260609_1925_full_diffucoder_base_parallel_baseline_candidate.md`；下一步行动说明 `docs/paper_agent/current_action.md`；LLaDA-1.5 probe brief `docs/paper_agent/experiments/20260609_2105_llada15_download_api_probe.md`。
- result：DiffuCoder-Base candidate 为 `839/1033 = 81.22%`，本地同 backbone `cal_lite` baseline 为 `838/1033 = 81.12%`，pairwise 为 `25` wins、`24` losses。这仍是 near-tie / slight local positive，不是强 bounded-repair claim。
- next：运行使用 proxy 的 LLaDA-1.5 metadata/download probe，检查 config/tokenizer/mask token；只有 API contract 兼容后才启动 tiny GPU smoke。

## 2026-06-09 18:27 CST

- action：持续监控 `Dream-org/Dream-v0-Base-7B` full baseline/candidate pair 直到完成，验证 row count/schema/backend/canvas，从 raw `results.jsonl` 重新计算 same-backbone pairwise/bucket/runtime metrics，并更新 GitHub 可读实验文档。
- evidence：baseline 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_170219`；candidate 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_170219`；两者均有 `1033` 行、`summary.json`、`0` 个 malformed JSON rows，并使用预期 backend/canvas。
- result：candidate 为 `803/1033 = 77.73%`，同 backbone 本地 cal_lite baseline 为 `802/1033 = 77.64%`，pairwise `28` wins、`27` losses，net `+1` task。Avg sec including probe 为 `3.7337`，baseline 为 `3.6494`。这是 near-tie / slight local positive，不是强 claim。
- diagnostics：oracle `>=17` long buckets 合计多 `+3` tasks，但 `9-12` 和 `13-16` 合计少 `-5` tasks。Repair 在 `101/1033 = 9.78%` 行触发，但 triggered rows 中只有 `13/101 = 12.87%` 是 true-long；大多数 triggers 是 short。
- literature positioning：candidate 高于 LR-DLLM Dream-7B single-line `76.7`，DreamOn Dream-7B `88.6` 是 training-based 且明显更高。以上只是 anchors，不是 protocol-matched claims。
- next：继续 backbone matrix，下一个为 `apple/DiffuCoder-7B-Base`。先使用 HuggingFace 镜像或 Git/LFS 做 download/API probe，再做 tiny smoke；runner/canvas/verifier contract 通过前不启动 full run。

## 2026-06-09 14:35 CST

- action：检查已完成的 DreamCoder Base/Instruct full runs，从 raw `results.jsonl` 重新计算 same-backbone pairwise/bucket/runtime metrics，并更新 GitHub 可读实验文档。
- evidence：Base 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327`；Instruct 输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_123359`；两者均有 `1033` 行和 `summary.json`。完成后的 GPU check 显示没有运行中的 GPU 进程。
- result：Base candidate 为 `832/1033 = 80.54%`，同 backbone 本地 baseline 为 `825/1033 = 79.86%`，pairwise `27` wins、`20` losses，net `+7` tasks。Instruct candidate 为 `834/1033 = 80.74%`，同 backbone 本地 baseline 为 `848/1033 = 82.09%`，pairwise `21` wins、`35` losses，net `-14` tasks。Base 是小幅 local positive；Instruct 是 negative transfer evidence。
- literature positioning：Base 高于 CAL DreamCoder-Base anchors（`70.2` average、`76.2` best shown），低于 LR-DLLM DreamCoder-7B `81.6`；DreamOn DreamCoder `92.1` 是 training-based。以上只是 anchors，不是 protocol-matched claims。
- next：继续 backbone matrix，下一个为 `apple/DiffuCoder-7B-Base`。使用 HuggingFace 镜像，先在 GPU 2/3 上做 download/API probe 和 tiny smoke；runner/canvas/verifier contract 通过前不启动 full run。

## 2026-06-09 12:12 CST

- action：为 `Dream-org/Dream-Coder-v0-Base-7B` 启动 official-canvas LCAL/S3 + official-CAL bounded-repair smoke，并排查运行环境。
- evidence：第一次 smoke 日志 `logs/paper_agent/20260609_1203_smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2.log` 显示模型加载成功但 `datasets` 在只读 `/home/shx/.cache/huggingface/datasets` 写 lock 失败；复制 HF module/dataset cache 到 `/tmp` 后，dataset-load probe 成功读出 `2` 条任务。第二次 smoke 加载模型和任务后，在 HumanEval verifier 的 `multiprocessing.Manager()` 处因 sandbox 禁止 socket/listener 失败；sandbox 内 CUDA probe 显示 `torch.cuda.is_available() == False`。
- result：这是 environment/sandbox blocker，不是 DreamCoder 方法结果。当前没有有效 pass rate 或 runtime。
- next：需要用户显式批准沙箱外执行，先跑 GPU2 上的 2-sample Base smoke；通过后再启动 Instruct smoke/full run，不能把这两个 failed sandbox output 用于论文比较。

## 2026-06-09 12:31 CST

- action：用户批准后，在沙箱外完成 DreamCoder Base 和 DreamCoder Instruct 2-sample smoke，并准备 full parallel runs。
- evidence：Base 输出 `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_122358`，Instruct 输出 `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_122945`；两者均有 `2` 行 JSON、`summary.json`、tier1/2/3 verifier、`lcal_v3`/`official_cal` metadata。Base/Instruct smoke 均为 `2/2` pass，且对各自 baseline 前两题均为 `2` tie-pass。
- result：official-canvas adapter 的 runner/schema/verifier/GPU-runtime sanity gate 通过；这不是 full performance result。已写 full run action brief，计划 Base 用 GPU2、Instruct 用 GPU3。
- next：启动两个 full `1033` sample runs，完成后分别对同 backbone baseline 和 literature anchors 做结果整理。

## 2026-06-09 12:34 CST

- action：启动 DreamCoder Base/Instruct full parallel runs。
- evidence：Base tmux session `dreamcoder_base_full_20260609_1231`，输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327`，日志 `logs/paper_agent/20260609_1231_full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed.log`；Instruct tmux session `dreamcoder_instruct_full_20260609_1231`，输出 `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_123359`，日志 `logs/paper_agent/20260609_1231_full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed.log`。
- result：早期健康检查通过；Base 已观察到 `15` 行，Instruct 已观察到 `7` 行；GPU2/GPU3 均在 `95%+` util、约 `15-16GB` 显存，无早期 OOM。
- next：等待完成后做 row-count/JSON sanity、读取 `summary.json`、对各自 same-backbone baseline 做 pairwise/bucket/runtime analysis，并整理文献锚点对照。

## 2026-06-09 11:04 CST

- action：检查已完成的 `GSAI-ML/LLaDA-8B-Instruct + midcons` run，并设计 literature-backbone rerun matrix。
- evidence：candidate summary 位于 `/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834/summary.json`；历史 baseline summary 位于 `/home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_llada-instruct_resume_20260512_141851/summary.json`；同时核对了 CAL、LR-DLLM、DreamOn 的 PDF 表格。
- result：candidate 为 `815/1033 = 78.90%`，低于同 backbone 历史 baseline `817/1033 = 79.09%`；pairwise 为 `17` wins、`19` losses。已写入 `docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md`。
- next：不升级 claim；下一项实现目标是 DreamCoder official-canvas bounded-repair adapter，先定义 smoke-run criteria，再考虑 full GPU run。

## 2026-06-04 20:31 CST

- action：按用户确认启动 `GSAI-ML/LLaDA-8B-Instruct` + 当前 `midcons` bounded-repair full run，使用 GPU `2,3`。
- evidence：第一轮直连 `huggingface.co` 启动因网络不可达在模型加载前被 Ctrl-C 中断；随后用 `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1` 重新启动。日志 `logs/paper_agent/20260604_2022_llada_instruct_midcons_full_hfmirror.log` 显示模型 checkpoint shards 已加载，`Loaded 1033 tasks`，输出目录为 `/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834`。
- result：在该时间点，run 尚未结束；最近检查时 `results.jsonl` 已有 `62` 行，GPU 2/3 正在使用。该 interim state 已被上方 2026-06-09 final result 覆盖。
- next：等待 full run 完成后做 row-count/JSON sanity check、读取 `summary.json`、和 LLaDA-Instruct 历史 baseline `817/1033 = 79.09%` 做 pairwise/bucket analysis，并更新 results/dashboard/checkpoint。

## 2026-06-04 15:40 CST

- action：对 strict-split probe-score diagnostic 做 final focused verification。
- evidence：`/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_analyze_probe_curve_split_score.py`、`py_compile`、`analysis/analyze_probe_curve_split_score.py`、JSON assertions、`git diff --check`。
- result：fresh verification 通过：`Ran 3 tests` / `OK`，compile exit `0`，audit regeneration 重现 `strict_heldout_pass=False heldout_triggers=63 short_risk=22.22%`，JSON assertions 确认 `cross_validation.aggregate_heldout` 关键 metrics，diff hygiene 通过。第一次 JSON assertion 因 schema path 写错失败，root cause 是断言命令错误，不是 audit metrics 改变。
- next：当前结果继续作为 negative evidence；下一步默认做 CPU-only conservative high-precision score kill-test，GPU work 继续 blocked。

## 2026-06-04 15:31 CST

- action：恢复长期 paper-agent session 后，核对 dashboard、checkpoint、experiment results 和 overnight log 的 verification 状态。
- evidence：`AGENTS.md`、`docs/paper_agent/research_agent_protocol.md`、`git status --short --branch`、`pause_checkpoint.current.md`、`paper_agent_dashboard.zh.md`、`experiment_results.zh.md`、`open_questions.zh.md`、`activity_ledger.zh.md` 尾部，以及 verification 相关 `rg` 命中。
- result：确认 dashboard 中 `verification-before-completion=completed` 对应较早 probe-curve audit 的 `Ran 8 tests` / `OK`；strict-split diagnostic 只有 focused `Ran 3 tests` / `OK`，仍需 final milestone-level verification、diff review、commit、push。已修正 checkpoint workflow 表述；没有启动 GPU 或新实验。
- next：先运行 strict-split diagnostic 的 final focused verification，再决定是否做 CPU-only conservative high-precision score kill-test；GPU work 继续 blocked。

## 2026-06-01 01:50 CST

- action：以低 token 模式恢复，并将已过期的 pause checkpoint 与当前 dirty files 对齐。
- evidence：`git status --short --branch`、`docs/paper_agent/pause_checkpoint.current.md`、`docs/paper_agent/paper_agent_dashboard.zh.md`、`docs/paper_agent/evidence_snapshot.md`、`docs/paper_agent/experiment_plan.current.en.md`、`docs/paper_agent/open_questions.en.md`，以及 `overnight_log.*.md` 尾部。
- result：确认未提交的 strict-split probe diagnostic 是当前计划中的 CPU-only 下一步；没有启动 gstack workflow 或 GPU experiment。
- next：完成 strict-split diagnostic evidence，并记录它是否通过 offline GPU gate。

## 2026-06-01 01:52 CST

- action：运行 strict-split probe-score 单元测试，并从已有 A6000 `midcons` result 生成 CPU-only audit。
- evidence：`tests/test_analyze_probe_curve_split_score.py`、`analysis/analyze_probe_curve_split_score.py`、`docs/paper_agent/probe_curve_split_score_audit.json`、`docs/paper_agent/probe_curve_split_score_audit.md`。
- result：测试通过；aggregate held-out gate 未通过，结果为 `63` triggers、`47.62%` true-long precision、`32.97%` failed-long recall、`22.22%` short-risk、`7.94%` current-pass risk。
- next：双语记录该 negative result，并保持 GPU work blocked。
