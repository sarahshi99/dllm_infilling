# LR-DLLM DreamCoder Fixed64 RandomSpan/MultiLine action

日期：2026-07-31 UTC。

Action name：`LRDLLM-DREAMCODER-FIXED64-RANDOMSPAN-MULTILINE`。

目的：补齐 `paper-guided, author-unverified reimplementation of LR-DLLM` 在 DreamCoder RandomSpan 与 MultiLine 上的相同 manifest、sampling、row seed、evaluator和逐 token left-commit Fixed64 common-protocol controls。Fixed64 是标准固定长度 baseline，但不称 LR-DLLM equal-compute 或 compute-matched。

## Frozen inputs

- backbone=`Dream-org/Dream-Coder-v0-Base-7B@2346ccd3be517d0d314152b988a3b9bafa7d6d63`；evaluator=`88062ff9859c875d04db115b698ed4b0f0395170`。
- decoder：Fixed64；temperature=`0.2`；top-p=`0.9`；seed=`42`；相同 canvas、special-token、evaluator和每 candidate deterministic row seed；search forwards=`0`，decode forwards为64次逐 token commit。
- RandomSpan：`1480` rows / `148` clusters；manifest SHA256=`474f7a2740027ff3b709ed7ab8b4f3c68125cda10688881f204d9dd405d2073e`。
- MultiLine：`5079` rows / `148` clusters；manifest SHA256=`b18fe79cda00048b89e0d92983e1b3d90aa82d95c0b17b684847f2962d132f80`。
- SingleLine Fixed64 technical smoke=`12/12`、resume no-op、full=`927/927`已通过，故不为相同实现重复构造非SingleLine smoke manifest；两个 full 依靠各自 exact-key/forward/failure audit fail-stop。
- frozen test=`sealed`，`test_evaluation_count=0`。

## Exact commands

```bash
tmux new-session -d -s lrdllm_fixed64_randomspan1480_20260731 'cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1 && ALLOW_SHARED_GPU=1 bash scripts/manual_launch_lrdllm_dreamcoder_20260731.sh fixed64_common_protocol full randomspan'

tmux new-session -d -s lrdllm_fixed64_multiline5079_20260731 'cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1 && ALLOW_SHARED_GPU=1 bash scripts/manual_launch_lrdllm_dreamcoder_20260731.sh fixed64_common_protocol full multiline'
```

Outputs：`outputs_clean/lrdllm_dreamcoder_{randomspan,multiline}_20260731_v1/`。Logs：`logs/paper_agent/20260731_lrdllm_dreamcoder_{randomspan,multiline}_fixed64_common_protocol.log`。

## Gates

- 各数据单独达到 `1480/1480` 或 `5079/5079`，missing/duplicate/error/failure=`0`；forward/token/wall/memory账本守恒；resume幂等。
- 只在对应 LR primary 与 Fixed64 两臂同 normalized candidate keys 完整后做 paired row/cluster help-harm和paired CI；三个数据不输出合并 Mean。
- 运行中只读 progress/ETA/OOM/ECC/failure，不读 partial accuracy。
- OOM/ECC或failure journal仅阻塞该 arm；不 kill/preempt其他任务。

本 brief 必须先 commit/push，再启动两个 tmux。
