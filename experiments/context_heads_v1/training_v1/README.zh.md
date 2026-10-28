# B 首次训练 pilot：七头统一 T1 teacher-KL，单 epoch

2026-10-28 用户明确要求开始训练，并随后明确允许 B 使用 GPU、共用 GPU0 剩余显存，
单 worker 依次训练小头；先前“不训练/仅 CPU”的限制对此次 pilot 已解除。
其他约束继续：单 Codex 会话、无子智能体，不修改 A/C、旧 decoder/旧训练脚本或思维导图，
不重跑/加载 DreamOn、不采集新数据、不生成 HumanEval 答案，不自动合并 A。

## 已选择的最小训练范围

- 只训练七种原结构：H0-static、Hctx-old-context-only、Hgated-author-family、H1、H2、H3、H4。
  不追加架构、只 target h 消融、GRU last-only 或 attention mean 等网格。
- 相同 A1 冻结缓存：2304 train 状态、768 validation 状态，源题组数为690/233，组交集为0。
  两种来源 Z4/H4-old 各半，global/frontier、single/crossline 和历史长度1/2/3按原盲选分层。
  不按 teacher 效果、代码 passed 或参考答案筛选。不是 3072 个独立源题。
- 每头1 epoch，batch16，每头计划144个 optimizer steps；相同 state-id 排序后、seed20261028
  的同一排列。没有 200k 采集、5 epoch、多种子、新 backbone 或训练中 rollout。
- rank256、lambda1；原初始化 seed20261015，E/W_out normal(0,.02)，适用头的两表随机值相同。
  所有头均从共同随机初始化起训，H1 gamma=1；没有任何头继承旧训练 checkpoint。
- H2 full-memory、H3 learned-attention、H4严格因果 teacher forcing；不启动 Jacobi 真实生成。
  teacher 状态只含合法 committed 同链前缀；provisional 拒绝。
- T=1 full-vocab KL(teacher||student)，student=softmax(old_logits+delta)。AdamW lr3e-4、
  betas(.9,.999)、eps1e-8、weight_decay0、grad norm clip1。没有独享 CE/teacher/DCD loss。
  这是一次固定超参数 pilot，不是已调优公平最终结果。
- FP32、关闭 TF32；CUBLAS_WORKSPACE_CONFIG=:4096:8、确定性算法。旧 h 已按 A 对齐，本模块
  不再 shift。V152064、H3584、tokenizer len151668，按 A 配置固定400个特殊/补齐行。
  所有 teacher/student 保留完整 V；特殊 delta0 不保护其归一化概率。

合理性：已有相同 teacher 缓存足以先验证训练与 KL 趋势，不需要重跑主模型；
比较七头新训后是否比共同旧 logits/各自随机初始更接近 teacher。
风险：单 epoch、单 seed、有限源题组；Hctx容量/校准可解释收益，K2不能证明多词记忆，
KL改善不等于代码正确率或端到端加速。原A反事实覆盖仍 partial，此训练不补齐它。

## 输入隔离与开销

从 A 的 state_metrics.jsonl.gz 只保留 state_id、输入/teacher相对路径与原 hash、group/split/
span/order/来源/history_length。读取原 npz 时仅九个公共字段，以及独立教师包的 fresh_logits；
不读取 x0、reference、tests、passed、branch_logp 或未来实际 token 作为模型输入。
完整同链 committed 前缀、形状/类型、有限FP32、原hash、组隔离均检查；异常会停止，不丢掉“难样本”。
teacher 定义沿用 A：X0 加该目标本链可见前缀，其他链新词不写入。

一次整理为只读 lossless FP32 memmap，逻辑数组共3,913,728,000 bytes，保存在外部 cache，
不提交普通 git。只把每个 batch 的副本送 GPU，缓存/旧 logits/teacher不求梯度。
每次只有一个真实 V 的头；各模型参数和全词表投影成本仍以原 B accounting 表为准，
不能忽略每头约0.31GB fp32大表、梯度及Adam状态。GPU每进程allocator上限8GiB；
实际 allocated/reserved峰值逐头记录，非端到端内存/速度结论。旧头零对照包装器不参与训练。

GPU0为共享H200 NVL，启动前看到其他进程约19GiB驻留；用户已明确同意共用。
不 kill/修改其他进程。CPU线程4，GPU worker1，依次释放各头自身的缓存。
本次 worker 墙时上限3600秒；非有限 loss/gradient、OOM、输入损坏或预算到达会停止并保留checkpoint，
不自动扩数据、加轮次、改batch或重抽样。训练/验证/写盘的实际时间单独记录，不称推理加速。

## 指标与 checkpoint

完整 validation 比较：共同 stale old_logits、每头训练前、训练后 T1 KL/TV；
按 history_length、order、span、来源和 order×history 分桶，并报告源题组等权 mean KL。
不将同一组状态当独立测试题，不以 K2 改善声称多词历史有效。
本次不生成答案；T=.2/top-p=.9实际生成指标、结构调度、GPU推理公平测速另做，未在此启动。

每64步和每头最后一步保存 last.pt（head+optimizer+cursor+plan）；best.pt 是 validation KL
在初始与epoch末两点间较优的一份模型权重。没有在 validation 上调参或多次选择epoch。
若epoch末比初始差，best_step=0会明确保留；训练后的最后权重仍在 last.pt。
同一命令可恢复，检查共同计划一致后从保存cursor继续；最多确定性重放未落盘的63个batch。
CPU小词表已检查实际KL下降和保存Adam状态后恢复与连续训练逐位相等；不外推为跨GPU/kernel位等价。

## 命令、运行位置与恢复

在本 B worktree 根目录：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/shx/miniconda3/envs/dllm_env/bin/python -m unittest experiments.context_heads_v1.training_v1.test_cpu -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 /home/shx/miniconda3/envs/dllm_env/bin/python -m experiments.context_heads_v1.training_v1.run --prepare-only
bash experiments/context_heads_v1/training_v1/launch.sh
```

launch.sh 的实际 GPU 命令：

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /home/shx/miniconda3/envs/dllm_env/bin/python -u -m experiments.context_heads_v1.training_v1.run --device cuda:0 --batch-size 16 --learning-rate 0.0003 --seed 20261028 --threads 4 --memory-limit-gib 8 --checkpoint-every 64 --max-wall-seconds 3600
```

通过 tmux `dreamon_context_heads_train_20261028_v1` 启动 launcher；外部运行目录：
`/home/shx/.cache/dllm_infilling/context_heads_training_20261028_v1/`。
其中 train.log、status.json、plan.json、environment.json、data/manifest.json、results/、
checkpoints/、summary.json 与 exit.json 是真实状态入口。旧 A/B/C ready、锁与旧报告不变。
同名 worker.lock 防止本 pilot 重复 worker，不删除其他锁。

原B及接入补验的payload文件保留；代码和新说明仅在本 training_v1/，项目日志只追加日期项。
提交推送后启动真实训练，source_commit由worker记录。完成后再归档小型结果，不提交checkpoint/
memmap或原始长日志。任何“已启动/已完成”的判断必须来自PID/实际step或summary，不能把此计划当结果。
