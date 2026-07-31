# DreamOn official-source SingleLine execution action

日期：2026-07-31 UTC（Friday, July 31, 2026）。

Action name：`DREAMON-OFFICIAL-SOURCE-SINGLE-H200-SINGLELINE`。

准确标签：**DreamOn official-source single-H200 reproduction**。它使用 released training-based checkpoint，不与 training-free 方法混栏；single-H200只替代官方8进程样本分片，不改变每例 generator。由于公开 source 未固定 seed，本项目预注册 deterministic per-candidate seed-42 派生，使 resume/样本并行不改变每例随机性；因此不得称为 exact official 8-GPU topology/random-stream reproduction。

## Source/checkpoint reconciliation

- source=`DreamLM/DreamOn@8a0a54918412eda9402a327646f7f067f7160ec8`；generator SHA256=`7709f1ef...18858`；evaluate SHA256=`d8cf3ce8...da57c`；source license=`Apache-2.0`。
- checkpoint=`Dream-org/DreamOn-v0-7B@8ccc74750e43177327f29dab9e91882ba759e194`；config SHA256=`c02a9899...e8d3`；weight index SHA256=`998a0781...5bdb`；license=`apache-2.0`。
- official source shell sweep：initial lengths=`4/8/16/32/64`，max=`64`，steps=`256`，temperature=`0.2`，top-p=`0.9`，entropy remasking，mask expansion，delete EOS，batch=`1`。
- 论文正文 `Lmax=128` 作为独立 paper-described protocol discrepancy登记；本 action 只执行 released official-source max64协议，不冒充未运行的 paper-exact max128配置。
- official `torchrun --nproc_per_node 8` 的 source route仅按 rank切分样本并以DDP包装同一模型；本地直接调用 pinned `MDMGenerator`，每例算法不变。

## Population and safety

- SingleLine project-non-frozen=`927` rows / `148` clusters；复用已冻结 common population manifest `analysis_outputs/baseline_manifests_20260731_v1/lrdllm_common_singleline_nonfrozen_manifest.jsonl`，SHA256=`84b86f059c4c479a1e45591bb038411fc5c16424c1e78fb81ebf038617d74269`。
- smoke12复用 `lrdllm_common_singleline_technical_smoke12_manifest.jsonl`，SHA256=`c1dc18fac3fd67ba73b11a197e547b4d6c30dc60b1ad018d672447c3d2885280`；manifest只含safe row id/task/group/fingerprint，不含solution/tests/outcome/oracle length。
- evaluator=`HumanEval-Infilling@88062ff9859c875d04db115b698ed4b0f0395170`；生成只读取prefix/suffix/entry point，canonical solution/tests只进入独立 evaluator。
- frozen test=`sealed`，`test_evaluation_count=0`；RandomSpan不属于paper-faithful表；MultiLine等待benchmark-only adapter equivalence proof。

## Adapter boundary and accounting

- adapter直接导入 pinned `eval.generator.MDMGenerator`；不复制或改写生成算法。
- pinned generator只在顶层导入、此后从不引用`OmegaConf`，而公开repo未提供requirements且当前env无该包；adapter按文件路径加载generator并提供仅满足该未使用符号的import-only shim，同时按官方`evaluate.py`语义实现HF tokenizer wrapper。source hash与“仅一次未使用OmegaConf引用”均fail-stop锁定。
- source函数返回时通过Python profile读取其局部 `expand_budget` 与 `num_generation_tokens`，据此无侵入记录真实 expansion/contraction；不改变控制流或采样。
- source dynamic expansion与denoising在同一forward中交织，因此search forwards=`0`、所有实际model calls登记为decode forwards；同时记录token-forwards、wall、peak memory、final length、remaining masks与termination。
- candidate key按source row与initial arm固定；raw append-only，failure journal独立，resume只跳过已成功exact keys。

## Exact commands

先执行min4技术smoke：

```bash
ALLOW_SHARED_GPU=1 bash scripts/manual_launch_dreamon_singleline_20260731.sh smoke 4
```

重复同命令验证resume `new_rows_written=0`。通过后，min4 full：

```bash
tmux new-session -d -s dreamon_sl927_min4_20260731 'cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1 && ALLOW_SHARED_GPU=1 bash scripts/manual_launch_dreamon_singleline_20260731.sh full 4'
```

其余 `8/16/32/64` 是同一official-source sweep的独立arms；每个先12-case smoke/resume gate，再full，不从outcome中选择“最佳”arm。Output=`outputs_clean/dreamon_singleline_20260731_v1/`；log=`logs/paper_agent/20260731_dreamon_singleline_min<L>.log`。

## Gates and budget

- smoke：`12/12` exact unique，missing/duplicate/error/failure=`0`；evaluator正常；forward/token/length movement守恒；resume no-op；frozen=`sealed/0`。
- full：每个arm `927/927`、148 clusters并单独报告；matched DreamCoder fixed-length controls完成前不做paired delta。
- 运行中只读progress/ETA/OOM/ECC/failure与机制账本，不读partial accuracy。
- OOM/ECC、source/checkpoint/hash偏差、remaining masks、accounting错误或failure journal均fail-stop该arm，不kill/preempt其他任务。
- smoke预算20–60分钟；单arm full预算4–12小时，实际以当前H200并发ETA为准。

本 adapter/tests/launcher/action/matrix 必须先commit/push，再启动任何DreamOn smoke。
