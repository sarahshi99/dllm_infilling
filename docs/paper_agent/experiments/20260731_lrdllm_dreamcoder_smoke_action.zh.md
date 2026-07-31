# LR-DLLM DreamCoder Technical→Mechanism Smoke Action Brief

日期：2026-07-31 UTC

状态：`adapter_manifests_preflight_verified_smoke_pending_gpu_blocked_after_CAL_priority`

准确标签：**paper-guided, author-unverified reimplementation of LR-DLLM**。

## Frozen protocol

- paper：arXiv `2602.07546v1`；截至 2026-07-31 未找到作者代码或 license。
- backbone：`Dream-org/Dream-Coder-v0-Base-7B@2346ccd3be517d0d314152b988a3b9bafa7d6d63`。
- checkpoint hashes：config `3c180c6d...08c9`；weight index `998a0781...5bdb`。
- checkpoint license：当前 cached snapshot 无 license 文件或 metadata；状态为 `external metadata verification pending`，不把 checkpoint/source 复制进发布 artifact。
- evaluator：HumanEval-Infilling `88062ff9859c875d04db115b698ed4b0f0395170`，使用 upstream README 说明的 in-memory single-line enablement overlay；checkout 不修改。
- confidence：raw-logit masked-position mean negative entropy；Stage I 全 `{1,2,4,8,16,32,64,128}` OLS；Stage II `{L-1,L,L+1}`；top-p=`0.9`、temperature=`0.2`、MAX_GEN=`128`。
- commit sampling：独立 forward；search/decode/total 不双计。
- Fixed64 control：相同 candidate keys、canvas、sampling、row seed、evaluator和逐 token left-commit decoder；只关闭 Stage I/II 动态长度搜索。它是 common-protocol Fixed64，不宣称 equal-compute。

## Immutable populations

- SingleLine full：927 rows/148 clusters，SHA256=`84b86f059c4c479a1e45591bb038411fc5c16424c1e78fb81ebf038617d74269`。
- technical smoke：12 rows/12 clusters，SHA256=`c1dc18fac3fd67ba73b11a197e547b4d6c30dc60b1ad018d672447c3d2885280`。
- mechanism smoke：64 rows/64 clusters，四个 prompt+suffix context-length buckets 各16，SHA256=`b1b27a59c724a957814b1878022b7ba1523eaaa55a9fabc7fe7fa2016060508f`。选择不使用 oracle/reference/tests/outcome。
- RandomSpan full：1480/148，SHA256=`474f7a2740027ff3b709ed7ab8b4f3c68125cda10688881f204d9dd405d2073e`。
- MultiLine full：5079/148，SHA256=`b18fe79cda00048b89e0d92983e1b3d90aa82d95c0b17b684847f2962d132f80`。
- summary SHA256=`520c74cec7c2f98d89d982b510c22ace3b06a57f372a11cc2c734521eda5faf4`；`sealed_files_opened=false`、`test_evaluation_count=0`。

## Exact execution

GPU：physical `0`，`CUDA_VISIBLE_DEVICES=0`。默认 launcher 发现已有 PID 会退出3；2026-07-31 用户追加授权允许并行时，必须显式设置 `ALLOW_SHARED_GPU=1`，记录 pre-existing PIDs，并在启动后核验显存/ECC。仍不抢占、不 kill。

Technical smoke：

```bash
bash scripts/manual_launch_lrdllm_dreamcoder_20260731.sh lrdllm_primary technical-smoke singleline
```

Resume no-op：原命令重复一次，要求 `new_rows_written=0` 且 raw 行数不变。

Mechanism smoke（technical gate 通过后）：

```bash
tmux new-session -d -s lrdllm_dc_mech64_20260731 'cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1 && bash scripts/manual_launch_lrdllm_dreamcoder_20260731.sh lrdllm_primary mechanism-smoke singleline'
```

Fixed64 technical control：

```bash
bash scripts/manual_launch_lrdllm_dreamcoder_20260731.sh fixed64_common_protocol technical-smoke singleline
```

Output：`outputs_clean/lrdllm_dreamcoder_singleline_20260731_v1/`。Logs：`logs/paper_agent/20260731_lrdllm_dreamcoder_singleline_<arm>.log`。Raw append-only，failure journal独立，progress/run manifest可恢复。

## Budget and gates

- technical 预算：模型加载后最多45分钟；若加载完成后10分钟没有首行 progress、OOM/ECC、failure journal非零或 accounting错误则停止该 baseline。
- mechanism 预算：最多4小时；只读取完整性、进度、ETA、OOM/ECC、failure journal和机制 gate，不读取 partial accuracy。
- technical gate：12/12 exact unique，0 missing/duplicate/error/failure；resume no-op；forward/token ledger守恒；evaluator正常；frozen=`sealed/0`。
- mechanism gate：64/64 exact unique；上述技术 gate全部满足；expansion和contraction都真实发生；无 128-move guard；终止和成本可解释；resume幂等。
- mechanism gate 后才允许依次 full：SingleLine 927 → RandomSpan 1480 → MultiLine 5079；每项需单独 tmux/action checkpoint，不输出合并 Mean。

Current preflight：22 个 LR core/builder/adapter tests通过；`py_compile`、CLI help、tokenizer/special-id load、checkpoint/evaluator/hash/population certificate和immutable resume no-op通过。GPU smoke尚未启动，进度0/12。
