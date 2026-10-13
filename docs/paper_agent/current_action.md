# 当前行动：Markov v2 训练结果归档与 K=2 推理实现

## 2026-10-13 最终结论与收尾修复

- 完整10330行、十组各1033，164基础题；16行检查通过，缺失/重复/失败均0；累计GPU作业约0.802小时。没有启动后续实验。
- 训练贡献 D−C：global +9（帮助/伤害15/6），frontier +18（29/11）；路径 C−B：global 0（0/0），frontier −5（6/11）。原 D−B +9/+13 分解为 +9+0 / +18−5，不能归因为路径的正收益。
- 关闭诊断、剔除评测和写盘后，KL头相对匹配K1纯生成速度 global=1.657×、frontier=1.818×（时间减少39.66%/45.01%）；无头K2=1.712×/1.899×。共享H200且配置按行轮换，外部负载仍可能影响计时。
- 真重算 E−D：global +25（28/3）、frontier +41（48/7）；E−C +34/+59。A对/B错人口救回：D=11/55、21/75；E=33/55、59/75，其他人口相对B新增伤害D=6/11、E=1/6。
- 统计限制：global头收益行加权95%区间含零，frontier头收益基础题等权区间跨零；E相对D的global基础题等权区间跨零，frontier两种权重均为正。重复开发人口、单生成种子，不能推广为独立测试结论。
- 下一轮最值得改的是右词修正的近似质量，依据E还能额外救回；E成本接近K1，仍不能当成可加速的正式验证器。建议只写入队列，不启动新方法。
- 后台生成/分析退出0；自动commit前因ledger末尾空行失败。已修复末尾换行和报告对两种权重区间的措辞，保留原生成版本与计时数据，补做focused commit/push。

## 2026-10-13T02:32:58.796268+00:00 K2 诊断后台任务：完成并停止

- 用户后续授权共享 H200；十配置串行、按行轮换，GPU 背景负载另存。共享负载可能影响纯生成耗时，不能称空卡测量。
- 状态 `completed_generation`，退出码 `0`；成功行 `10330/10330`，smoke `passed`，完整性 `passed`，GPU 作业累计 `0.8017h`（上限12h，排队等待不计）。
- 代码 `a55b1fd78795ae5f83845302a76f31780dda8cd2`；报告 `analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/report.zh.md`，全部配对结论见报告。
- 本轮全部完成；没有后续 GPU 作业。方法建议只写入报告，不自动运行。

## 2026-10-13 用户授权共享 H200 后台执行

- 本条取代之前“有其他进程即停止”的资源限制；不抢占、不终止其他进程。等待可用显存至少32GiB后开始，等待阶段不加载模型、不计GPU预算。
- exact command：`tmux new-session -d -s dreamon_markov_k2_diagnosis_20261013 'bash scripts/background_dreamon_markov_k2_diagnosis_20261013.sh > analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/background.log 2>&1'`（diagnosis 工作树内）。
- 47项CPU检查通过。后台顺序：16行开关检查→估时/12h预算→10330行→聚类分析/完整性审计→更新compact记录→focused commit/push→停止。协议差异/预算/异常会保存阻塞，不改参数或范围。
- tmux=`dreamon_markov_k2_diagnosis_20261013`；日志与结果目录=`analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/`。无需持续人工监督。共享背景负载记录保留，报告明确计时限制。
- 旧阻塞产物移入同目录 `before_shared_gpu_authorization/`；原人口、头、种子和方法不变。

## 2026-10-13 DreamOn + Markov K2 诊断：代码已保存，GPU 资源阻塞

- 分支 `codex/dreamon-markov-k2-diagnosis-v1`；实现提交 `66c5c2b`，结果目录 `analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/`。从已审阅 `b8d6e58` 后最新 `0e97f61` 复用，原工作区和旧结果未改动。
- A/B/C/D/E × 两策略 runner/analyzer 已实现；C 为同类同秩零输出投影、lambda=1；E 只填原左词、保留原右位与未来 mask；真正关闭概率/排名/replay/画布诊断，分离 CUDA 同步生成、评测、写盘计时。
- 46 项 CPU 检查通过（含随机采样 RNG/诊断一致性、事务恢复与去重、合成十组完整统计链）；真实模型 16 行检查 **未运行**，正式批次 **0/10330**，完整性 **incomplete_or_invalid**。CPU 检查不是实验完成。
- 已实际执行 `bash scripts/manual_run_dreamon_markov_k2_diagnosis_20261013.sh`，退出 `3` / `blocked_gpu_busy`：唯一 H200 上仍有 6 个其他进程、约 91761 MiB 占用。没有加载模型、消耗 GPU 预算或抢占进程。
- 当前结论收窄：旧 D−B 的 +9/+13 净行数尚未隔离路径贡献，不能全归训练头；旧 wall 不能回答公平纯生成速度；E 恢复能力未测量。旧报告保留作为历史证据，不宣称过去所有收益都是假的。
- 唯一恢复项：空闲 H200 上运行同一入口，先固定稳定 ID 的 16 行开关等价性/估时，通过且不超过累计 12h 后自动全量十组。若资源仍忙，入口退出；不缩小人口，不留下 watcher。
- 下一方法建议等待 D−C / C−B / E−C,D 证据后决定。本轮不启动训练、上下文头、正式验证器、K4/K8 或其他新方法。

<!-- dreamon-markov-k2-implementation-20261010 -->
## 2026-10-12 v2 训练与 K=2 完整评测已完成

- 已完成：v2 prepared-data 与 SQLite bank 审计通过；TV/KL 已从共同零输出初始化完成 pilot/full；验证和外部分布测试均选择 KL、`lambda=1`。这些是分布拟合结果，不是 HumanEval Pass@1。
- 六组生成：`6198/6198`，退出码 `0`；六组各 `1033` 条，基础题组 `164` 个，重复/缺失/错误记录为 `0`。
- 结果：四个无头方案精确复现历史 `951/901/942/877`；全局 KL=`910/1033`，左前沿 KL=`890/1033`。KL 相对对应 K=2 无头分别净增 `9/13`，全局分组 CI 为正，左前沿 CI 跨零；两者仍低于逐词刷新。
- exact command：`CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python experiments/run_dreamon_singleline_order_parallel.py --markov-k2 --source-root /home/shx/projects/dllm_infilling/.worktrees/dreamon-source-8a0a549 --evaluator-root /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff --model-snapshot /home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194 --markov-checkpoint /home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/checkpoints/kl/full/best.pt --markov-lambda 1 --output-dir analysis_outputs/dreamon_markov_k2_full_20261011_v1 --device cuda`
- tmux/log：`dreamon_markov_k2_full_20261011` / `logs/paper_agent/20261011_dreamon_markov_k2_full.log`；汇总脚本为 `analysis/analyze_dreamon_markov_k2_full.py`。
- GitHub：本地实现提交 `25503d8`，SSH 推送仍因 `github.com` 超时未核验；远端已确认的上一提交为 `110229b`。
- 新增用户授权：评测 TV 头。入口已增加明确的 `--markov-head-name TV` 标签，避免把 TV 结果写成 KL；沿用相同 1033 条、模型、采样和逐题 seed。输出目录：`analysis_outputs/dreamon_markov_k2_tv_full_20261012_v1/`，tmux：`dreamon_markov_k2_tv_full_20261012`。
- 行动说明：`docs/paper_agent/experiments/20261010_markov_k2_implementation_action.zh.md`。

### 2026-09-20 机器重启后的 v2 恢复

- action：复用 2026-09-11 已完成的 v2 transition bank，不重建样本库；修复全量 SQLite 审计脚本的 worktree 导入路径和合法 `reference_token_id=-1` sentinel 判断。
- evidence：`transition_bank_summary.json`=`completed`，总数=`240000`；全量审计 `actual_bank_membership_audit_v2.json`=`passed`，errors=`[]`，SQLite quick_check=`ok`，unique sample keys=`240000`，排除交集=`0`，每题每策略上限=`16`。
- previous interruption：旧训练 watcher 在审计入口因 `ModuleNotFoundError: experiments` 退出，未开始 TV/KL，不是样本库或训练数值错误。
- next exact command：在新的 detached worktree 中运行 `bash scripts/manual_continue_dreamon_markov_v2_training_20260910.sh`；脚本将从 bank summary 直接进入已通过的 audit 后恢复 common init/microbatch/TV/KL。
- GPU policy：沿用 CUDA GPU0，允许与现有任务共享；不 kill、不覆盖 v1、显存不足时由已有 microbatch 探测和 checkpoint 恢复处理。

<!-- markov-training-audit-20260908 -->
## 2026-09-10 当前行动：无基准候选的 v2 数据与训练

- action：在新目录重建 OpenCoder `educational_instruct@7d28f40d...` 数据，先建立题意/实际回答代码/代码 AST/测试的传递关联组，再保守排除包含 HumanEval 候选的整个组；split seed 保持 `20260901`。
- exact data command：`HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python analysis/prepare_dreamon_markov_training_data.py --model-snapshot /home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194 --evaluator-root /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff --result-dir analysis_outputs/dreamon_markov_head_training_20260910_v2 --cache-dir /home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/data --split-seed 20260901`。
- next GPU phases：共享 bank `200000/20000/20000`，随后 TV→释放→KL；H200 单卡串行、`dreamon-repro`、新共同零初始化，旧 v1 权重与优化器不加载。
- success：新实际 bank 无 HumanEval 候选，关联组零跨集，原策略/阶段配额满足；两个头按 `optimizer_batch_v2` 完成既定 pilot/full 验证，至少一个非零修正通过后才打开新外部分布测试并进入六组 SingleLine。
- kill/pause：排除后任一既定 bank 配额不足、两个头均失败、数据或主键完整性错误、或 GPU 被其他任务占用且无法安全串行。禁止缩小规模、重复样本凑数或覆盖 v1。
- outputs：Git 结果目录 `analysis_outputs/dreamon_markov_head_training_20260910_v2/`；服务器数据/权重 `/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/`；运行日志 `logs/paper_agent/20260910_dreamon_markov_v2_k2_eval.log`。

### 2026-09-10 启动前复算审计

- action：从固定 OpenCoder 源数据和 HumanEval 1033 条记录重新计算标准化去重、候选匹配、传递关联组和确定性 split；逐条对照服务器 `prepared_records.jsonl.gz`、完整 split manifest 与 external-test manifest。
- reviewer motivation：防止把数据准备脚本成功退出误当成候选排除、组隔离和 tokenized artifact 均正确。
- exact command：`HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python analysis/audit_dreamon_markov_prepared_data.py --model-snapshot /home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194 --evaluator-root /home/shx/.cache/dllm_infilling/human-eval-infilling-88062ff --prepared-records /home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/data/prepared_records.jsonl.gz --split-manifest analysis_outputs/dreamon_markov_head_training_20260910_v2/split_manifest.jsonl.zst --external-test-manifest analysis_outputs/dreamon_markov_head_training_20260910_v2/external_test_manifest.jsonl.zst --exclusion-audit analysis_outputs/dreamon_markov_head_training_20260910_v2/humaneval_exclusion_audit.json --data-summary analysis_outputs/dreamon_markov_head_training_20260910_v2/data_manifest_summary.json --preparation-audit analysis_outputs/dreamon_markov_head_training_20260910_v2/split_and_dedup_audit.json --output analysis_outputs/dreamon_markov_head_training_20260910_v2/prepared_data_reaudit.json`。
- success：源数据 118278、HumanEval 1033/164；实际 prepared 主键唯一且与 manifest 完全一致；HumanEval 直接候选及其 39 个关联组与 prepared 交集为零；group 零跨 split；token 计数、长度、row seed、词表边界全部一致。
- pause/kill：任一主键、候选排除、group split、token 长度或词表边界不一致；失败时不启动 GPU bank。
- resources：CPU only；不修改源数据或既有 v1/v2 artifact。GPU bank 仍需后续真实生成并做 200000/20000/20000 的 SQLite 成员审计。

### 2026-09-10 GPU transition bank 启动

- action：在独立 detached execution worktree 中，从通过复算审计的 v2 prepared records 生成共享 transition bank；两种轨迹各半，early/middle/late 固定配额，offset=1，每题每策略最多 16。
- exact command：`tmux new-window -t dreamon_markov_v2_k2_eval -n bank-v2 'bash scripts/manual_launch_dreamon_markov_v2_bank_20260910.sh'`。
- GPU/env：物理 H200 GPU 0；`CUDA_VISIBLE_DEVICES=0`；`dreamon-repro`；允许与当前外部低占用进程共享，但不 kill、不抢占或修改对方任务。
- input：`/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/data/prepared_records.jsonl.gz`。
- output：`/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/transition_bank.sqlite`；compact summary 写入 v2 result dir。
- log：`logs/paper_agent/20260910_dreamon_markov_v2_bank.log`。
- success：SQLite 精确达到 train/validation/external=`200000/20000/20000`，每个 policy/stage stratum 达到固定容量，最大每题每策略不超过 16；随后必须执行实际成员审计。
- pause/kill：CUDA/OOM/ECC、模型/词表/官方源码版本不符、任一 split 配额真实不足、数据主键或坐标断言失败。不得重复采样凑数。

### 2026-09-10 bank 后自动训练链

- action：独立 tmux watcher 等待 bank 完整结束，先做 240000 条 SQLite 全量成员审计；通过后建立共同零初始化、测定 microbatch、依次运行 TV 与 KL，并按既有验证规则决定是否打开一次新的外部分布测试。
- exact command：`tmux new-window -t dreamon_markov_v2_k2_eval -n train-v2 'bash scripts/manual_continue_dreamon_markov_v2_training_20260910.sh'`。
- failure isolation：TV 运行错误不会阻止 KL 的一次既定尝试；两个 status 都存在后才 finalize。诊断文件原子写入，恢复时复用完整已有诊断，不重做已完成外部样本。
- shared protocol：同一 bank、共同 seed=42 零输出初始化、rank=256、lr=3e-4、effective batch=128、最多5轮/patience2；pilot 20000/4000，随后重新从共同初始化开始 full。
- outputs：服务器 checkpoint root `/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/checkpoints/`；日志 `logs/paper_agent/20260910_dreamon_markov_v2_training.log`。
- success：实际 bank audit 通过；TV/KL 各完成既定 pilot/full 尝试；若至少一个头有有效非零修正，按 validation 选定一个共同 checkpoint/λ，再执行外部分布测试。两个头都失败则记录负结果，不进入 SingleLine 方法评测。

## 2026-09-10 当前行动：服务器实际成员核查

- action：只读关联已保存的 transition bank、HumanEval 候选 record_id 与现有验证/外部诊断 sample_key；命令为 `/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python analysis/audit_markov_bank_membership.py --result-dir analysis_outputs/dreamon_markov_head_training_20260901_v1 --bank-db /home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260901_v1/transition_bank.sqlite`。
- reviewer motivation：旧报告已撤回“严格隔离”；只有确认候选是否实际进入 full training，才能决定旧 KL v1 权重是否可用于完整 1,033 条开发/机制评估。
- success：生成 `actual_bank_membership_audit.json`，验证 200,000/20,000/20,000 bank 主键数与两类现有诊断完全一致，并给出每个实际入 train 的 HumanEval 候选的 transition 数。
- pause condition：任何 AST 无文档字符串代码匹配候选实际进入 train，或关键主键不一致，则仅实现 K=2 与外部小样本检查，暂停完整 HumanEval；不重训、不删题、不改旧划分。
- resources：CPU/SQLite only；不加载模型、不做教师前向、不修改 bank；输出写入既有 Markov 审计目录。

## 2026-09-08 当前状态：训练完成，协议偏差已确认

- 决策：`iterate`；原训练提交 `32e8004`，修正分支 `codex/dreamon-markov-training-audit-fix-20260908`。
- 完整冻结清单存在跨集重复：代码 1,239 组，题目文本 3,102 组（两者重叠，不可相加）。不能再声称严格隔离。
- 已修复无头报表；保守剔除后的外部测试保留 13,387 条，TV/KL 距离缩小 26.69%/27.79%，是事后敏感性结果，不是新的独立测试。
- 旧权重仍属小批次参考损失均值版本；整批归一化已修复代码但未重训。两条训练候选代码与 HumanEval/13 的标准最大公约数实现匹配，实际是否训练需查服务器 SQLite；其余单断言候选需人工复核。
- 下一步：执行 `analysis/audit_markov_bank_membership.py`；实现固定左前沿双词元与全局 top-2 相邻短链。仅在基准隔离问题澄清后，用原验证选定 KL、λ=1 完成 1,033 条 Pass@1/速度比较。确认污染则保留实现与外部检查，暂停正式 HumanEval，不自行重训。
- 权威报告：`analysis_outputs/dreamon_markov_head_training_20260901_v1/report.zh.md`；原权重、原始诊断与冻结清单保留不动。以下旧日期条目仅作历史记录，不是当前运行指令。

原训练方案保留于 `experiments/20260901_dreamon_markov_head_training_action.zh.md`。本轮未启动任何 GPU 任务。

## 2026-10-13 K2诊断CPU复核完成（追加，旧判断保留）

- 起点HEAD `340170ce6ae44477027dd7e94ec7a9c0f0910384`；SQLite只读与JSONL的10330个payload逐字段一致，原库未修改；全部32字段/完整输出导出约940KiB gzip，可随仓库复算。
- 隔离临时目录不含SQLite/作者数据/模型，且禁止SQLite连接和torch/transformers导入；相对路径分析通过。独立聚合与10000次基础题整组bootstrap（seed20261013）复算，与两份原表/原报告数值表共1588单元格一致；无数值差异。
- 新发现并澄清：原未恢复运行错误0不等于评测无超时。13次timed out全部集中HumanEval/32；共享资源对墙钟超时的影响无法排除，缺CPU负载、评测起止UTC、子进程CPU/退出码/超时来源。仅超时标签未知的敏感性边界不改变原标签，D−C净增范围[9,10]/[17,20]。
- 1052个GPU历史快照全部有6个其他任务，已仅导出匿名数值。1.657×/1.818×只称共享H200观测；轮换不能替代独占计时。旧report保留，新日期勘误另存。
- D−B净+9/+13追回B相对A并行净损失的18%/20%，不混用A对/B错子组救回比例。E仍是固定原右位只填左后重算的机制诊断，不是K1等价、数学上界或成功验证算法。
- 当前复核入口：`analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/review_20261013/INDEX.zh.md`；review_manifest.latest.json已引用，旧manifest快照和旧条目保留。最终归档脚本后续自动纳入紧凑证据并去除GPU进程身份信息。
- 本轮CPU工作到提交推送即停止；没有生成/评测重跑、训练、GPU实验或停止其他进程。后续最小CPU超时检查/独占计时均仅写条件和命令，未排队，不自动推进新方法。

<!-- dreamon-markov-error-audit-20261013-v1-completed -->
## 2026-10-13 DreamOn + Markov 全量错误审计完成（追加）

- 单个持久Codex实际审阅326/326策略行（global144/frontier182），198挖空、100基础题；1220失败输出全部逐组解释；unreviewed/missing_evidence/failed_infrastructure=0。全对1740策略行只保留，未冒充错误归因已审。
- 全部10330原字段/输出/词元与c843847补证gzip精确一致；新32格/ABD八类/对称四格连接通过，复用旧1588格审核与CI。原报告、通过位、历史日期均保留。
- 447失败程序编译定位；773失败取证记录（248次新官方执行，其余同程序证据复用），13历史超时各一次同3秒隔离CPU复评仍超时。bwrap/空环境/只读运行时/seccomp/资源限/关闭网络/凭据不可见，单并发，不使用GPU。760条插桩与原测试结果一致；早期清理故障单独保留并修复，原官方结果不改。
- D-C救44行中37为原语法/缩进失败；D-B伤害6/11，含语法、边界、覆盖、类型和逻辑条件；D失败/E通过28/48，D/E都失败95/95。A/B/D都失败73/78。首次生成分歧326行均unknown，未将最终差异或动作总数倒推成因果。
- 报告/单文件离线HTML/逐条人工记录/coverage完整入口：[审计索引](../../analysis_outputs/dreamon_markov_error_audit_20261013_v1/INDEX.zh.md)。HTML默认326，支持全2066策略行；数据与DOM/实际JS交互验证通过，无原生浏览器视觉验证。
- 分支`codex/dreamon-markov-error-audit-v1`；reviewer_gate_disabled，本地diff+fresh verification。commit/push完成标志在ignored本地completion.json，启动器退出另存runner_exit.json；没有自动重启、子智能体或后续GPU研究。
