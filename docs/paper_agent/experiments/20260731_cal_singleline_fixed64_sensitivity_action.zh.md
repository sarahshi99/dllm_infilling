# CAL SingleLine Fixed64 sensitivity action

日期：2026-07-31 UTC。

Action name：`CAL-PHASE1B-SINGLELINE-FIXED64-SENSITIVITY`。

准确标签：**project Fixed64 internal control in the pinned CAL decoder family on the 838-row / 143-cluster project-non-frozen SingleLine CAL-Rest subset**。

该 arm 是标准固定长度敏感性/跨协议参考，不是 official CAL、DreamOn、LR-DLLM 或 DAEDAL；没有实际 matching forward/token budget 时，不称 equal-compute 或 compute-matched。CAL 的中心同协议控制仍是 `official_fixed32`。

## Frozen protocol

- source：`NiuHechang/Calibrated_Adaptive_Length@741e8418a88a732b4c92812424d4f03cab1f7b1f`；evaluator=`88062ff9859c875d04db115b698ed4b0f0395170`；model=`GSAI-ML/LLaDA-8B-Base@0f2787f2d87eac5eed8a087d5ecd24277e6255b2`。
- population：`838` rows / `143` clusters；manifest=`analysis_outputs/baseline_manifests_20260731_v1/cal_singleline_rest_nonfrozen_manifest.jsonl`，SHA256=`52ef81385984a362fee8729c52cbfa7480265582cabb8250d3ffc27b0aa59af0`；smoke12 SHA256=`56559f3f83ba1ce84c9e03622c5ced6e2ed8d2fa08145a1a64dc0cae0885caa1`。
- arm=`project_fixed64_internal`；initial/max length=`64/64`；span=`1`；dstep=`-1`；bias=`false`；temperature/CFG=`0/0`；oracle=`false`；steps/block length=`64/64`；seed=`42`。
- GPU：physical H200 index `0`，`CUDA_VISIBLE_DEVICES=0`。按用户 2026-07-31 并行授权允许与其他项目实验共享，但仅在真实 free memory ≥`20,000 MiB`、ECC=`0` 时启动；不 kill/preempt。
- output=`outputs_clean/official_cal_singleline_20260731_v1/`；canonical raw=`project_fixed64_internal_raw.jsonl`；log=`logs/paper_agent/20260731_official_cal_singleline.log`。

## Exact commands

Technical smoke：

```bash
bash scripts/manual_launch_official_cal_singleline_20260731.sh smoke project_fixed64_internal
```

Resume no-op：重复同一 smoke command，要求 `new_rows_written=0`。

Full：

```bash
tmux new-session -d -s cal_fixed64_sl838_20260731 'cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1 && bash scripts/manual_launch_official_cal_singleline_20260731.sh full project_fixed64_internal'
```

## Gates and budget

- smoke：`12/12` exact unique，missing/duplicate/error/failure=`0`，forward/token/wall/memory字段完整，resume幂等，frozen=`sealed/0`。
- full：`838/838`、143 clusters，零 missing/duplicate/error/failure，账本守恒；完成前只读 progress/ETA/OOM/ECC/failure，不读 partial accuracy。
- kill/block criteria：source/checkpoint/config/manifest hash偏差；任一 failure journal/accounting error；OOM/ECC；free memory不足。OOM只阻塞本 arm，不终止其他 baseline。
- 预算：smoke约2–8分钟；full在当前并发H200下先按45–90分钟规划，以实时完整性 ETA 为准。

本 action brief 与 matrix 状态必须先 commit/push，再启动 smoke。
