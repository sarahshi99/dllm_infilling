# Paper Agent Pause Checkpoint

<!-- dreamon-markov-v2-training-20260910 -->
## 2026-09-10 当前状态：v2 数据复算通过，GPU bank 与训练 watcher 已启动

- 分支/远端：`codex/dreamon-markov-k2-eval-v1@f40241cbb2d8e05217e0ef1db43eb2640cfe409a`，remote 已核对一致。
- 数据复算：`prepared_data_reaudit.json` 状态 `passed`、`errors=[]`。固定 OpenCoder=`118278`，去重后基准排除前=`117346`，prepared=`106491`；split=`86019/10547/9925`，record/source 主键均唯一，`69516` groups、零跨 split。
- HumanEval 排除：`1033/164` 变体/基础题，解析失败/组内差异=`0/0`；直接候选 `88`，排除关联组 `39`、记录 `559`；prepared 实际交集 `0`。该检查不证明基础模型预训练未见过 HumanEval。
- tokenization 复算：全部 `117346` 条重新执行，保留内容逐 token 完全一致，mismatch=`0`；排除=`10846` 条无合法 line middle、`9` 条超过 1024，总计 `10855`，与既有审计一致。
- 恢复修复：transition reservoir 现按源记录事务提交 `progress`，中断回滚后从精确 next index 恢复；训练 curve epoch 幂等，patience 保存当前 epoch 更新后的值，诊断 gzip 原子替换。相关 `27` tests 通过。
- GPU bank：tmux=`dreamon_markov_v2_k2_eval:bank-v2`；execution commit=`3fcf7828fcf00e80dc0e19fb4e5b27a6f9f195c0`；PID=`1583582`；DB=`/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260910_v2/transition_bank.sqlite`；log=`logs/paper_agent/20260910_dreamon_markov_v2_bank.log`。
- 2026-09-10T04:59Z 健康快照：train source progress=`858`，selected transitions=`20248`；H200 util约`75%`、ECC=`0`，无 traceback/OOM。早期 strata 当前占多数；v1 同构造在处理 train `63457` 条后满足全部 early/middle/late 配额，故继续观察，不据启动阶段比例提前停止。
- 自动训练链：tmux=`dreamon_markov_v2_k2_eval:train-v2`；execution commit=`f40241c...`；log=`logs/paper_agent/20260910_dreamon_markov_v2_training.log`。bank 完成后自动执行全量 SQLite 成员审计→共同零初始化→microbatch→TV→KL→验证选头/λ→有非零 deployable gain 时外部分布测试。
- 后续未完成：两个 K=2 推理设计、实现等价性/结构动作测试、六组 1033 条 SingleLine 完整开发/机制评测与统计分析。只有训练门通过才进入完整方法评测。
- 恢复命令：`tmux attach -t dreamon_markov_v2_k2_eval`；快速状态：`tail -n 80 logs/paper_agent/20260910_dreamon_markov_v2_bank.log` 和 `tail -n 80 logs/paper_agent/20260910_dreamon_markov_v2_training.log`。

<!-- markov-training-audit-20260908 -->
## 2026-09-08 当前状态：训练完成，协议偏差已确认

- 决策：`iterate`；原训练提交 `32e8004`，修正分支 `codex/dreamon-markov-training-audit-fix-20260908`。
- 完整冻结清单存在跨集重复：代码 1,239 组，题目文本 3,102 组（两者重叠，不可相加）。不能再声称严格隔离。
- 已修复无头报表；保守剔除后的外部测试保留 13,387 条，TV/KL 距离缩小 26.69%/27.79%，是事后敏感性结果，不是新的独立测试。
- 旧权重仍属小批次参考损失均值版本；整批归一化已修复代码但未重训。两条训练候选代码与 HumanEval/13 的标准最大公约数实现匹配，实际是否训练需查服务器 SQLite；其余单断言候选需人工复核。
- 下一步：执行 `analysis/audit_markov_bank_membership.py`；实现固定左前沿双词元与全局 top-2 相邻短链。仅在基准隔离问题澄清后，用原验证选定 KL、λ=1 完成 1,033 条 Pass@1/速度比较。确认污染则保留实现与外部检查，暂停正式 HumanEval，不自行重训。
- 权威报告：`analysis_outputs/dreamon_markov_head_training_20260901_v1/report.zh.md`；原权重、原始诊断与冻结清单保留不动。以下旧日期条目仅作历史记录，不是当前运行指令。


## 历史记录（已完成）：2026-09-01 Markov 训练启动状态

- Branch/base/current implementation commit: `codex/dreamon-markov-head-training-v1` / `77f0572b1ca4fe031ab6bbf29b3a4d8740f38802` / `987979d`.
- tmux: `dreamon_markov_head_training_v1`; launcher PID=`2209172`; formal bank PID at launch audit=`2211662`.
- Log: `logs/paper_agent/20260901_dreamon_markov_head_training_v1.log`; result dir: `analysis_outputs/dreamon_markov_head_training_20260901_v1/`; local root: `/home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260901_v1/`.
- Current phase: formal shared transition-bank generation on physical H200 GPU 0. Launch audit used/free/util=`16571/126586 MiB/73%`.
- Frozen data: OpenCoder `educational_instruct@7d28f40d579edd7c24402d17d0c7639f991e6f8d`, seed `20260901`, train/validation/external-test=`85426/10738/10776`, external-test manifest SHA-256=`f179a671b37879bd60a0fb47c0c33f30f662a9b8bcade386b51ea0c98b1a1ab9`.
- Prelaunch verification: 33 tests OK; real bank smoke `240/240`; rank-256 parameters=`77,856,768`; micro-batch=`16`, effective batch=`128`; real 32-transition TV loss `0.2163 -> 0.1088`; smoke released GPU to `0 MiB`.
- Automatic sequence: bank -> common init/config -> data commit -> TV -> release audit -> KL -> release audit -> conditional one-time external test -> final audit/commit/push.
- Resume: `tmux attach -t dreamon_markov_head_training_v1`; log tail: `tail -n 100 logs/paper_agent/20260901_dreamon_markov_head_training_v1.log`.
- Scope: no HumanEval generation/results, K=2/K=4, RNN, controller, MultiLine, long block, or backbone training.

## 2026-07-31 DreamCoder MultiLine Matched-Control Freeze Override

- `DreamCoder Fixed4/8/16/32/64 under DreamOn sampling/decoder, MultiLine` 复用现有adapter的`dreamcoder_fixed + multiline`组合；没有新算法分支。
- full=`5079/148`，smoke=`12/12`；launcher=`scripts/manual_launch_dreamcoder_dreamon_fixed_multiline_20260731.sh`，action=`docs/paper_agent/experiments/20260731_dreamcoder_dreamon_fixed_multiline_controls_action.zh.md`。
- freeze commit=`497737e`已push，CPU preflight通过。10个独立tmux watcher已建立：smoke/resume串行`64→4→8→16→32`，full等待全部smoke及对应dynamic/fixed predecessor；统一free≥60GiB连续3次20秒采样才释放。
- Fixed64 smoke queue=`dreamcoder_dreamon_fixed64_ml_smoke_queue_20260731`，pane=`2773397`；其他pane=`2788663/2788704/2788742/2788778/2788816/2788852/2788889/2788929/2788968`。当前GPU余量不足，所有新queue均未加载模型。
- `2026-07-31T17:58:11Z`进度：DreamOn min4=`1277/5079` ETA约3h45m；min8=`916/5079` ETA约3h54m；LR Fixed64=`3448/5079` ETA约1h44m；failure均0，GPU0 free=`14439 MiB`，ECC=0。Frozen test=`sealed/0`。
- LR completion watcher=`lrdllm_multiline_paired_analyzer_wait_20260731`，pane=`2857187`，log=`logs/paper_agent/20260731_lrdllm_multiline_paired_analyzer_wait.log`，output=`analysis_outputs/lrdllm_dreamcoder_multiline_paired_20260731_v1/`。只在exact complete gate后运行10,000-cluster bootstrap paired analyzer。

## 2026-07-31 DreamOn SingleLine Matched-Control Closure Override

- DreamOn min64−DreamCoder Fixed64：row help/harm=`327/25`；cluster=`90/7/51`；task-macro delta=`+21.211pp`，95% CI=`[+15.657,+26.692]pp`。
- DreamCoder Fixed4/8/16/32/64 controls全部`927/927`，missing/duplicate/error/failure/accounting=`0`；五个same-length paired analyses全部完成。
- DreamOn min64 canonical audit=`927/927`且missing/duplicate/error/accounting=`0`，但append-only journal保留1条已恢复的资源OOM，所以其strict progress status仍为`audit_failed`；不要改写为failure=`0`。
- MultiLine min4/min8 full运行，min16/32/64队列；LR MultiLine Fixed64运行。当前日期Friday, July 31, 2026；Frozen test=`sealed/0`。

## 2026-07-31 DreamOn MultiLine Launch Override

- `DreamOn official-source MultiLine reproduction via benchmark-only adapter` 已在commit `487fdaa` push后通过五个12-case smoke/resume gates。
- min4 full运行：tmux/pane/Python=`dreamon_ml5079_min4_20260731/1829390/1829424`，output=`outputs_clean/dreamon_multiline_20260731_v1/`，log=`logs/paper_agent/20260731_dreamon_multiline_min4.log`。
- min8 queue=`dreamon_ml5079_min8_queue_20260731`已于`2026-07-31T17:06:36Z`释放，Python=`2151434`；min16等待min4完成；min32等待min8；min64等待min16。剩余queue只读进度/显存，不占GPU。
- 当前日期是Friday, July 31, 2026；不读partial accuracy。Frozen test=`sealed`，`test_evaluation_count=0`。

## 2026-07-31 DreamOn min32 vs Fixed32 Paired Override

- DreamOn min32−DreamCoder Fixed32：row help/harm=`286/26`；cluster=`89/9/50`；task-macro delta=`+19.060pp`，95% CI=`[+13.661,+24.447]pp`。
- Fixed32 completed=`927/927`，missing/duplicate/error/failure/accounting=`0`。Fixed64与LR MultiLine Fixed64运行。
- delta 为完整系统对比，不隔离动态长度或训练。Frozen test=`sealed/0`。

## 2026-07-31 DreamOn min4 vs Fixed4 Paired Override

- DreamOn min4−DreamCoder Fixed4：row help/harm=`590/3`；cluster=`124/0/24`；task-macro delta=`+58.464pp`，95% CI=`[+53.091,+63.720]pp`。
- Fixed4 completed=`927/927`，missing/duplicate/error/failure/accounting=`0`。Fixed32/64与LR MultiLine Fixed64运行。
- delta 为完整系统对比，不隔离动态长度或训练。Frozen test=`sealed/0`。

## 2026-07-31 DreamOn min16 vs Fixed16 Paired Override

- DreamOn min16−DreamCoder Fixed16：row help/harm=`194/31`；cluster=`81/9/58`；task-macro delta=`+13.943pp`，95% CI=`[+9.618,+18.405]pp`。
- Fixed16 completed=`927/927`，missing/duplicate/error/failure/accounting=`0`。Fixed4/32/64与LR MultiLine Fixed64运行。
- delta 为完整系统对比，不隔离动态长度或训练。Frozen test=`sealed/0`。

## 2026-07-31 DreamOn min8 vs Fixed8 Paired Override

- DreamOn min8−DreamCoder Fixed8：row help/harm=`286/11`；cluster=`111/1/36`；task-macro delta=`+33.243pp`，95% CI=`[+28.390,+38.417]pp`。
- 该 delta 是 released training-based system 相对 base fixed control 的整体差异，不隔离动态长度或训练效应，也不称compute-matched。
- Fixed16/32/64 full与LR MultiLine Fixed64运行；Fixed4在Fixed8完成后改为free≥36GiB门控。Frozen test=`sealed/0`。

## 2026-07-31 LR RandomSpan Paired + DreamCoder Control Launch Override

- LR-DLLM RandomSpan primary−same-key Fixed64：row help/harm=`104/328`；cluster=`19/107/22`；task-macro delta=`-15.135pp`，95% CI=`[-17.905,-12.365]pp`，明确负于 Fixed64。
- DreamCoder Fixed4/8/16/32/64 under DreamOn sampling/decoder：五个12-case smoke/resume gates全部通过。full Fixed64/32/16/8已运行；Fixed4在独立58GiB free-memory gate等待。
- LR MultiLine Fixed64继续运行；不读取partial accuracy。frozen test=`sealed`，`test_evaluation_count=0`。

## 2026-07-31 DreamOn SingleLine Completion Override

- `DreamOn official-source single-H200 reproduction` 的 min4/8/16/32/64 五个 full 均完成 `927/927`。row Pass@1=`88.4574/90.3991/90.7228/91.2621/91.6936%`；148-cluster macro=`78.1111/81.3313/83.1788/83.0372/85.1299%`。
- 五项 canonical audit 均为 missing/duplicate/error/forward-accounting/length-accounting=`0`。min32/min64 各有 1 条已恢复的资源 OOM failure journal；append-only raw 保留，resume/dedup完成。
- DreamCoder DreamOn-matched Fixed4/8/16/32/64 尚未冻结/完成，所以当前只报绝对指标。LR RandomSpan/MultiLine Fixed64 controls继续运行。
- frozen test=`sealed`，`test_evaluation_count=0`；不进入M5/PPT/ExecRepoBench final。

## 2026-07-31 Baseline Results + DreamOn Fulls Override

- CAL SingleLine same-key：primary−official_fixed32 row help/harm=`130/96`，cluster=`47/31/65`，task-macro delta=`+1.898pp`，95% CI=`[-2.233,+5.759]pp`。Fixed64 sensitivity完成，但不称equal-compute。
- DAEDAL SingleLine dynamic−Fixed8 macro=`+0.410pp`，CI=`[-2.062,+2.734]pp`；MultiLine=`+0.455pp`，CI=`[-1.375,+1.791]pp`。两项CI跨0。
- LR-DLLM DreamCoder：SingleLine vs Fixed64 macro=`+0.796pp`，CI=`[-6.746,+8.178]pp`，row help/harm=`246/114`；RandomSpan primary row/macro=`18.3784%`；MultiLine primary row=`31.9945%`、macro=`37.1082%`。RandomSpan/MultiLine Fixed64 controls仍运行。
- DreamOn official-source SingleLine：min4/8/16/32/64 smoke=`12/12`且resume `0` writes；五个927 full均在独立tmux运行。2026-07-31T14:39Z GPU free约9GiB、ECC=0；不添加第8个模型。
- frozen test=`sealed`，`test_evaluation_count=0`；不进入M5/PPT/ExecRepoBench final。

## 2026-07-31 DreamOn Adapter Freeze Override

- DreamOn released checkpoint blocker已解除；`Dream-org/DreamOn-v0-7B@8ccc74750e43177327f29dab9e91882ba759e194`、Apache-2.0和source `8a0a549…` hashes已固定。
- 新增outcome-blind `experiments/dreamon_singleline_adapter.py`、6个unit tests、launcher与action brief。CPU preflight验证SingleLine smoke12/common 927 manifests、source/checkpoint/evaluator/tokenizer IDs；没有加载GPU模型、没有生成/evaluator raw。
- official-source协议固定为min=`4/8/16/32/64`、max=`64`、steps=`256`、temperature=`0.2`、top-p=`0.9`、entropy与mask expansion；paper正文Lmax=128保持独立未执行。随机性固定为per-candidate seed-42派生，不称exact official 8-GPU random stream。
- 下一动作：本commit/push成功后，在真实显存槽上启动min4 12-case smoke；frozen test=`sealed`，`test_evaluation_count=0`。

## 2026-07-31 CAL SingleLine Primary Completion Override

- `official-source CAL, initial length 32, on the 838-row / 143-cluster project-non-frozen SingleLine CAL-Rest subset` 于 `2026-07-31T11:02:34Z` 完成 `838/838`，failure/error=`0/0`。冻结 analyzer：row=`52.3866%`，143-cluster macro=`41.2711%`，CI=`[36.3798%,46.0499%]`。
- 同 key official_fixed32 full仍运行，故暂不做paired comparison。`project_fixed64_internal` action已在commit `5bc4df8` push后，于`2026-07-31T11:02:49Z`启动12-case smoke；它是长度敏感性参考，不称equal-compute。
- Frozen test=`sealed`，`test_evaluation_count=0`。

## 2026-07-31 LR-DLLM SingleLine Completion Override

- `paper-guided, author-unverified reimplementation of LR-DLLM` DreamCoder SingleLine primary 已于 `2026-07-31T10:56:03Z` 完成 `927/927`，failure/error=`0/0`。冻结 analyzer：row=`72.7077%`，148-cluster macro=`60.5238%`，CI=`[54.9216%,65.8574%]`；Fixed64未完成前不做paired comparison。
- 该进程释放显存后，资源 watcher 于 `2026-07-31T10:56:09Z` 启动 DAEDAL MultiLine Fixed8 full；tmux/pane=`daedal_fim_ml4990_fixed8_20260731/3552559`，Python PID=`3717698`。
- RandomSpan/MultiLine LR primary、SingleLine Fixed64、CAL primary/Fixed32、DAEDAL SingleLine dynamic及MultiLine dynamic/Fixed8继续运行。Frozen test=`sealed`，`test_evaluation_count=0`。

## 2026-07-31 Baseline Closure Concurrent Execution Checkpoint（authoritative override）

- branch=`codex/ccfa-execution-sprint-v1`；remote/local launch HEAD=`9f0deb9b11576653fab7f0c75572561e7bdc246f`，是用户指定 `b42ba3038c794ca76d3f5908bd869e0164228a9e` 的 descendant。仅保留两处既知用户 untracked M1/M2 目录，无重叠 dirty。
- CAL MultiLine 4,990 最终标签/结果不变：`official-source CAL, initial length 32, on the 4,990-row / 143-cluster project-non-frozen CAL-Rest common subset`；row=`32.9259%`，task-macro=`27.8471%`，CI=`[24.3313%,31.2715%]`，没有 same-key Fixed32 paired result。
- CAL SingleLine：primary smoke/resume gate通过，primary full tmux=`cal_singleline_838_20260731`（pane/Python=`3110755/3110763`）；Fixed32 smoke=`12/12`、resume no-op=`0` writes，full tmux=`cal_fixed32_sl838_20260731`（`3467135/3467143`）。两臂都只读 progress/integrity，未读 partial accuracy。
- LR-DLLM：标签固定为 `paper-guided, author-unverified reimplementation of LR-DLLM`；technical=`12/12`、mechanism=`64/64`、resume gates通过。Primary full：SingleLine tmux=`lrdllm_dc_sl927_20260731`（`3156939/3156977`）、RandomSpan=`lrdllm_dc_randomspan1480_20260731`（`3447694/3447727`）、MultiLine=`lrdllm_dc_multiline5079_20260731`（`3451834/3451862`）。Fixed64 SingleLine full=`lrdllm_fixed64_sl927_20260731`（`3478035/3478062`）。
- DAEDAL：准确标签=`CAL authors’ DAEDAL FIM adaptation`。SingleLine dynamic tmux=`daedal_fim_sl838_20260731`（`3142712/3142739`）；Fixed8 已完成 `838/838` 并分析：row=`50.8353%`，143-cluster macro=`37.5859%`，CI=`[32.6317%,42.6749%]`。MultiLine dynamic full=`daedal_fim_ml4990_dynamic_20260731`（`3488947/3488975`）；Fixed8 full 资源守候 tmux=`daedal_fim_ml4990_fixed8_20260731`（pane=`3552559`），free≥20GiB且ECC=0时原地启动。
- DreamOn checkpoint 已通过 mirror 完整缓存：`Dream-org/DreamOn-v0-7B@8ccc74750e43177327f29dab9e91882ba759e194`，license=`apache-2.0`，snapshot=`/home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194`。项目内尚无已冻结 DreamOn adapter/manifest，因此不得误报为 smoke-ready。
- 2026-07-31T10:44Z GPU0=`NVIDIA H200 NVL`，8 个项目模型稳定并发，free约`6.1GiB`，util=`100%`，ECC=`0`；外部约70GiB进程已自行退出，无 kill/preempt。
- safety：frozen test=`sealed`，`test_evaluation_count=0`；M5、M1--M4、PPT、ExecRepoBench final均未进入。

## 2026-07-31 Baseline Closure Latest Checkpoint（authoritative override）

- branch=`codex/ccfa-execution-sprint-v1`；remote 已同步到 `96d7a24`，本段文档更新 commit 待生成。
- pushed commits：Phase0 `8d9838d/e7a790b`；CAL 4,990 result `64c49c7`；CAL SingleLine adapter `54f1a98`；LR prereg/core/adapter `a0737dc/92ab200/815a795`；DAEDAL adapter `48661dc`；LR/DAEDAL analyzer aliases `96d7a24`。
- CAL 4,990 label/result：`official-source CAL, initial length 32, on the 4,990-row / 143-cluster project-non-frozen CAL-Rest common subset`；row=`1643/4990=32.9259%`，task-macro=`27.8471%`，CI=`[24.3313%,31.2715%]`；no same-key fixed32。
- CAL SingleLine：manifest=`838/143`；smoke=`0/12`、full=`0/838`。恢复命令：`bash scripts/manual_launch_official_cal_singleline_20260731.sh smoke`。通过完整性 gate 后重复同命令验证 `new_rows_written=0`，再启动 tmux `cal_singleline_838_20260731` full。
- 当前 blocker：GPU0 外部 PID `755980/810890/818373`；2026-07-31T07:10:13Z used/free=`55361/87796 MiB`、util=`53%`、ECC=`0`。不要 kill/抢占。三个 baseline launcher均会因已有 compute PID退出3。
- LR-DLLM：官方代码搜索结论为未找到作者实现；唯一标签=`paper-guided, author-unverified reimplementation of LR-DLLM`。DreamCoder adapter/manifests/preflight完成；SingleLine/RandomSpan/MultiLine formal status=`0/927,0/1480,0/5079`；technical=`0/12`、mechanism=`0/64`。
- DAEDAL：标签=`CAL authors’ DAEDAL FIM adaptation`；SingleLine/MultiLine adapter preflight完成，dynamic/Fixed8 smoke均未运行。CAL source无 LICENSE，禁止复制到发布 artifact。
- DreamOn checkpoint `Dream-org/DreamOn-v0-7B` 不在 cache，独立阻塞 Phase2。
- safety：frozen test=`sealed`，`test_evaluation_count=0`；sealed files未打开；用户 untracked M1/M2目录保持不变；M5/M1--M4/PPT/ExecRepoBench final均未进入。

## 2026-07-31 External Baseline Closure Authoritative Resume Point

Timestamp: 2026-07-31 UTC
Branch: `codex/ccfa-execution-sprint-v1`

### Completed

- Phase 0 matrix/analyzer/SingleLine 838 immutable manifests committed and pushed at `8d9838d`; legacy manifest-key compatibility committed/pushed at `e7a790b` before outcome read.
- official-source CAL MultiLine 4,990 provenance/config gate passed and final outcome was analyzed: row `1643/4990=32.9259%`; 143-cluster task-macro `27.8471%`, 10,000 bootstrap CI `[24.3313%,31.2715%]`.
- No same-key official_fixed32 exists, so no paired delta/help-harm is reported.
- Frozen test remains `sealed`, `test_evaluation_count=0`; M1--M4 and historical raw remain unchanged.

### Running / Next

- No GPU process is active at this checkpoint. Server has one H200 at physical device index `0`; use `CUDA_VISIBLE_DEVICES=0`, at most one project GPU process.
- Next exact action: commit/push CAL 4,990 result docs, then run frozen SingleLine CAL-Rest smoke12. If technical gate passes, immediately launch SingleLine 838 full in an independent tmux.
- DreamOn released checkpoint is missing from local cache; this blocks DreamOn Phase 2 only, not CAL.

### Result paths

- `analysis_outputs/official_cal_multiline_4990_grouped_20260731_v1/`
- `docs/paper_agent/experiments/20260731_official_cal_multiline_4990_result.zh.md`
- `analysis_outputs/baseline_manifests_20260731_v1/cal_singleline_rest_nonfrozen_manifest.jsonl`

### Safety

- frozen controller test=`sealed/0`
- test_evaluation_count=`0`
- no M5/PPT/ExecRepoBench final/M1--M4 action

## 2026-07-28 Execution Sprint V1 Authoritative Resume Point

Timestamp: 2026-07-28 UTC
Branch: `codex/ccfa-execution-sprint-v1` (M4 result and compact CPU audit artifacts are the current authoritative working state; consult `git log` for commit IDs)

### Completed

- M1/M2/M3 formal 148-group decisions are fixed and non-promoted. M1 historical 5079 is `safely_paused_resumable` only.
- official CAL smoke=`12/12`, full=`4990/4990`, missing/error=`0/0`, final integrity complete. Do not claim a 5,715-case reproduction and do not report partial CAL accuracy.
- M4 used the mandated read-only `40,632` MultiLine bank after proving RandomSpanLight visible-context match=`0/164`. Frozen hash-selected 148-group manifest; technical three-arm full passed; fair equal-512 assembly−best=`-12.84pp`, help/harm=`0/19`; status=`reviewed_not_promoted_v0_multilinecore_cross_source`.
- DreamOn `8a0a549` official source audit found a genuine FIM/HumanEval-Infilling dynamic-expansion route but training-based/8-GPU requirements; GPU smoke not started. rho-EOS `69992ca` is completion-only and has no faithful FIM/evaluator route; no GPU run authorized.
- Frozen test remains `sealed`, count=`0`; raw outputs remain uncommitted. Existing untracked M1/M2 analysis directories are user-owned and untouched.

### Current State / Risks

- No paper primary and no eligible 296/927 expansion; do not retune or fuse M1--M4.
- H200 snapshot after M4: util=`0%`, free=`143156 MiB`, ECC=`0`; no project GPU process.
- Claim readiness: `not_ready`. Remaining paper gaps are external baseline outcomes, external benchmark, and a new independently preregistered method—not parameter repair of the four negative V0s.

### Next Safe Actions

1. No authorized GPU action is pending; preserve current raw outputs and compact status.
2. If resuming P1, write a fresh DreamOn 12-case smoke brief that pins evaluator and explicitly labels training-based/8-GPU protocol; do not launch automatically from this checkpoint.
3. Do not create a rho-EOS FIM adapter without a separately audited algorithmic protocol.

### Recommended Resume Prompt

Read `method_portfolio.current.json`, `runtime_status.current.json`, this top section, `paper_agent_dashboard.zh.md`, and `analysis_outputs/p1_dreamon_rhoeos_protocol_audit_20260728_v1/report.zh.md`. Preserve M1--M4 negative results and raw outputs; do not restart CAL or M4. Only proceed to a new GPU experiment after a new user-authorized brief.

## Phase 5 Authoritative Resume Point

Timestamp: 2026-07-12 UTC

### Canonical Resume Override

The current route is no longer another Phase 5 retry. Read `docs/paper_agent/ccfa_master_roadmap.zh.md` and `docs/paper_agent/current_action.md` first. Phase 5 is complete at result commit `52f07457a7974fafa69d59914bbc412a2d83e305`. Active work is P1 official baseline compatibility, P2 grouped statistics over 148 base-task clusters behind 6707 spans, and P4 non-HumanEval benchmark feasibility. M1--M4 are canonical independent methods; only M1-D0/M4-D0/A1 diagnostics have run. Frozen test remains sealed at count zero.

### Completed H200 Run And Scientific Decision

- User-authoritative baseline: `45bead22e3d21daa707be724cf2bdcbbf776592a`.
- The earlier pre-launch state was `infrastructure_blocked / scientific_pending`, not a scientific failure. The manual launcher subsequently created a real GPU process and completed normally.
- Base bank: `1332/1332` rows over `148` allowed tasks; zero missing, duplicate, extra, error, frozen, or malformed rows.
- Alpha auxiliary: `728/728` rows over `91` verified tasks; zero missing, duplicate, extra, or error rows.
- F1–F4 completed. The deterministic combined proxy reached cross-canvas within-task pairwise accuracy `0.6273`, selection net `+15` versus fixed64 and `+14` versus confidence, with no short-bucket net regression.
- The corrected gate failed because grouped-bootstrap delta lower bounds were not strictly positive against every deterministic baseline. Conditional V0 verdict: `killed_corrected_within_task_gate_failed`.
- No supervised-score fallback, heuristic substitution, method fusion, or additional generation was run.
- Frozen test remains `sealed`; `test_evaluation_count=0`.
- Fresh verification: `18` tests, py_compile, artifact assertions, JSON/CSV parse, forbidden-feature audit, exact compact raw-schema scan, and diff hygiene passed.
- Result report: `docs/paper_agent/experiments/20260712_phase5_h200_candidate_bank_result.md`.

Phase 5 decision: `iterate`.

Authoritative branch: `codex/risk-controlled-dynamic-rescue`

Verified starting HEAD after fetch: `b8ae031cf3b430833613aaf29083535edc0738f6` (remote/local match, `0/0`). Ignore older HEAD values below.

2026-07-12 resumed audit correction started from verified remote/local HEAD `3315ed82d42770e3ed7d8ae20e9d5f1570940ff6`. Original F3/V0 was invalid as a training-free selector because pass-trained OOF logistic scores entered selection indirectly. The correction separates supervised diagnostic probes from a fixed deterministic `AST/def-use bridge proxy V0`, adds a corrected within-task/cross-canvas gate, and rejects supervised scores at the V0 boundary. Candidate-bank protocol remains unchanged.

Clean implementation worktree: `.worktrees/phase5-method-falsification`; primary-checkout untracked user files preserved.

Completed:

- Stage A method registry and CCF-A gap ledger.
- Shared-bank/F1–F4/F3 gate/forbidden-feature preregistration.
- Candidate bank, premise falsification, and conditional V0 scaffold.
- Focused verification: `13` tests, py_compile, JSON/CSV parse, sealed-zero audit, raw-code compact scan, diff hygiene.
- Read-only RandomSpanLight audit: `164` source, `148` allowed unique groups, frozen intersection empty, Stage B expected rows `108` smoke / `1332` full.

Historical infrastructure blocker (resolved by the user-authorized manual launch):

- Approved H200 command was rejected before process launch by the approval service: `422 Unprocessable Entity: model not found: codex-auto-review`.
- No GPU process/model load/candidate/evaluator row exists.
- Do not retry or route around without explicit user approval after this notice.
- 2026-07-12 user-authorized exact retry was performed once after correction commit `5aa86c7a3368e8024449ccf421febfbf9eb78dd2` was pushed; it returned the same pre-launch `422`. Candidate rows remain `0`; no process is active. Do not retry again in this session.
- Manual launch entry: `docs/paper_agent/phase5_manual_h200_launch_20260712.md` and `scripts/manual_launch_phase5_h200_candidate_bank.sh`.

Frozen test: `sealed`, `test_evaluation_count=0`.

Operational decision: `completed`.

Scientific decision: `current_proxy_gate_failed`; future work requires a new preregistered iteration.

Phase 5 scaffold commit: `08f01141b8cf6a8e611d55162a3c21d09ca0cf12`.

Completed actions:

1. Exact H200 command completed; smoke/full/alpha audits passed.
2. F1–F4 completed from the full shared bank.
3. Conditional V0 wrote the preregistered killed report after the corrected F3 gate failed.

## Historical H200 Checkpoint Below

Timestamp: 2026-07-06 UTC

## H200 Migration Resume Point

Current branch: `codex/risk-controlled-dynamic-rescue`

Current HEAD: `2b0662bfe9fdab787a5249dc9cbefea12d683af1`

Current phase: H200 new-server bootstrap is corrected before reproduction. Default sandbox GPU checks fail, but approved host/unsandboxed GPU checks pass. Do not resume from the older V8/post-V8 plan; continue H200 Tier 1 reruns through approved unsandboxed/escalated GPU commands.

Latest artifacts:

- `docs/paper_agent/new_server_h200_bootstrap.zh.md`
- `analysis_outputs/h200_bootstrap_20260705_103617/report.md`
- `analysis_outputs/h200_bootstrap_20260705_103617/environment_manifest.json`
- `analysis_outputs/h200_bootstrap_20260705_103617/copied_artifact_hashes.csv`
- `analysis_outputs/h200_bootstrap_20260705_103617/GITHUB_REMOTE_VERIFICATION.md`

Blocking evidence:

- Default sandbox: H200 is visible at `/proc/driver/nvidia/gpus/0000:22:00.0/information`, but `nvidia-smi` fails, `/dev/nvidia*` is hidden, and `dllm_env` reports `torch.cuda.is_available() = false`.
- Approved host/unsandboxed context: `nvidia-smi` succeeds, H200 is idle, `/dev/nvidia0`, `/dev/nvidiactl`, and `/dev/nvidia-uvm` are visible, and `dllm_env` reports `torch.cuda.is_available() = true`, `gpu_count = 1`.
- GitHub SSH auth is now verified by `ssh -T git@github.com`.
- Remote branch freshness is verified: `git ls-remote origin refs/heads/codex/risk-controlled-dynamic-rescue` returns `2b0662bfe9fdab787a5249dc9cbefea12d683af1`, matching local HEAD.

Frozen test status:

- `test_status = sealed`
- `test_evaluation_count = 0`

Next three actions after blocker is fixed:

1. Stage and push the H200 bootstrap correction.
2. Start Tier 1 H200 reruns in new output directories through approved unsandboxed/escalated GPU commands.

## Historical Checkpoint Below

Timestamp: 2026-07-01 15:40 CST

## Current Branch

`paper-agent-overnight`

## Current Phase

The literature-backbone local same-backbone matrix has completed through `inclusionAI/LLaDA-MoE-7B-A1B-Base`. Trace-long-rescue v1, trace feature audit v2, Route2 full follow-ups, Discovery V3/V4, V5/V6 selector polish, V7 proportional widening, and V8 proportional CAL score full runs have all completed. Current LLaDA-Base best follow-up is still V6 short override `802/1033 = 77.64%`. V7 is negative: `792/1033 = 76.67%`, pairwise vs `midcons` `1/4/791/237`. V8 directly changed the CAL-like formula and is also negative: V8a `786/1033 = 76.09%`, V8b `781/1033 = 75.61%`, V8c `782/1033 = 75.70%`. The latest decision is not to continue global proportional length reward as the main policy; any further ratio idea must be local and guarded by an under-selection/risk detector.

## Completed Items

- Initialized bilingual paper-agent docs under `docs/paper_agent/`.
- Built and tested a compact evidence snapshot builder.
- Generated `docs/paper_agent/evidence_snapshot.md` and `.json` from local A6000 raw outputs.
- Implemented and tested `analysis/analyze_probe_curve_long_signals.py`.
- Generated bilingual probe-curve audit docs and JSON.
- Updated plan history through `v3`, including the GPU `2,3` allocation constraint.
- Verified the current milestone with focused unit tests, compile checks, audit regeneration, JSON assertions, and `git diff --check`.
- Entered pause flow and updated the dashboard, logs, evidence snapshot, and this checkpoint.
- Resumed from the stale checkpoint in low-token mode and reconciled the uncommitted strict-split diagnostic files.
- Implemented and tested `analysis/analyze_probe_curve_split_score.py`.
- Generated bilingual strict-split probe-score audit docs and JSON.
- Updated the dashboard, evidence snapshot, experiment results, open questions, activity ledger, and overnight logs with the negative strict-split result.
- Reconciled the checkpoint workflow table with the dashboard/results/log evidence: earlier probe-curve audit verification was completed, while the strict-split diagnostic has focused 3-test verification and still needs final milestone-level verification before commit/push.
- Completed fresh focused verification for the strict-split diagnostic on 2026-06-04: unit test, py_compile, audit regeneration, JSON assertions, and `git diff --check` passed.
- Wrote an Action Brief for the LLaDA-Instruct `midcons` full run to `docs/paper_agent/current_action.md` and `docs/paper_agent/experiments/20260604_2022_llada_instruct_midcons_full.md`.
- Started the LLaDA-Instruct full run. The first launch attempted direct `huggingface.co` access and was interrupted before samples ran; the active run uses `HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`.
- Confirmed the active run loaded model checkpoint shards, loaded `1033` tasks, and created output directory `/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834`.
- Completed the LLaDA-Instruct run: `815/1033 = 78.90%`, below the historical same-backbone LCAS-v3 baseline `817/1033 = 79.09%`; pairwise `17` wins and `19` losses.
- Wrote `docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md` with CAL, LR-DLLM, DreamOn backbone anchors and the next rerun plan.
- Implemented the DreamCoder official-canvas LCAL/S3 + official-CAL bounded-repair adapter in `clean_scripts/run_dreamcoder_official_infilling.py`; `py_compile` passed.
- Wrote the DreamCoder Base smoke action brief to `docs/paper_agent/current_action.md` and `docs/paper_agent/experiments/20260609_1203_smoke_dreamcoder_base_lcal_official_bounded_repair.md`.
- Debugged the smoke environment: copied HF module/dataset caches to `/tmp`, confirmed dataset loading works from `/tmp`, and identified the remaining blocker as sandbox-disabled CUDA plus sandbox-disabled verifier multiprocessing.
- After user approval, ran sandbox-outside DreamCoder Base smoke successfully: `2/2` pass, `2` tie-pass versus its same-backbone baseline over the first two tasks, output `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_122358`.
- Ran sandbox-outside DreamCoder Instruct smoke successfully: `2/2` pass, `2` tie-pass versus its same-backbone baseline over the first two tasks, output `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_122945`.
- Launched and completed full DreamCoder Base/Instruct parallel runs in tmux after smoke success.
- Recomputed DreamCoder same-backbone metrics from raw `results.jsonl`: Base candidate `832/1033 = 80.54%` versus baseline `825/1033 = 79.86%`, pairwise `27` wins and `20` losses; Instruct candidate `834/1033 = 80.74%` versus baseline `848/1033 = 82.09%`, pairwise `21` wins and `35` losses.
- Updated DreamCoder full-run experiment brief, experiment results, dashboards, activity ledgers, and this checkpoint with local baseline and literature-anchor separation.
- Downloaded/probed `Dream-org/Dream-v0-Base-7B` via local Git/LFS checkout under `/tmp`, ran a valid 2-sample smoke, launched a full same-backbone pair, monitored both tmux sessions to completion, and recomputed Dream-7B same-backbone metrics from raw `results.jsonl`: candidate `803/1033 = 77.73%` versus local cal_lite baseline `802/1033 = 77.64%`, pairwise `28` wins and `27` losses.
- Downloaded/probed `apple/DiffuCoder-7B-Base` via proxy/mirror Git/LFS path under `/tmp`, ran a valid 2-sample smoke, launched a full same-backbone pair, monitored both tmux sessions to completion, and recomputed DiffuCoder-Base same-backbone metrics from raw `results.jsonl`: candidate `839/1033 = 81.22%` versus local cal_lite baseline `838/1033 = 81.12%`, pairwise `25` wins and `24` losses.
- Wrote the next action brief for `GSAI-ML/LLaDA-1.5` download/API probe to `docs/paper_agent/current_action.md` and `docs/paper_agent/experiments/20260609_2105_llada15_download_api_probe.md`.
- Completed the `GSAI-ML/LLaDA-1.5` metadata/API/local-weight probe through direct HuggingFace with the configured proxy. All six safetensors shards are present under `/tmp/llada15_probe_20260609` and byte-size checked against HF API/index. Local API/config/tokenizer/model-class inspection passed; CPU/local checkpoint loading passed. GPU smoke was not launched because GPUs `2/3` were occupied and sandboxed Python could not see CUDA.
- Completed the `GSAI-ML/LLaDA-1.5` 2-sample smoke pair and full local same-backbone pair on shared GPUs `2/3`. Baseline `cal_lite` LCAS-v3b is `817/1033 = 79.09%`; LCAL official bounded-repair candidate is `818/1033 = 79.19%`, pairwise `18` wins and `17` losses. This is a near-tie/slight local positive, not a strong claim upgrade.
- Completed `inclusionAI/LLaDA-MoE-7B-A1B-Base` proxy/mirror download recovery under `/tmp/lladamoe_probe_20260610`; all three safetensors shards are present and `safe_open` validation passed.
- Debugged LLaDA-MoE environment compatibility: `dllm_env` Transformers was too old for `modeling_rope_utils`, while the `dllm_env` `flash_attn_2_cuda` extension failed on `GLIBC_2.32`. The active runs therefore use `PYTHONPATH=/tmp/no_flash_attn:/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages` with `/home/shx/miniconda3/envs/dllm_env/bin/python`.
- Completed LLaDA-MoE 2-sample smoke gate. Baseline smoke output `/home/shx/projects/dllm_infilling/outputs_clean/smoke_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112427` is `2/2`; candidate smoke output `/home/shx/projects/dllm_infilling/outputs_clean/smoke_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112452` is `2/2`; both logs ended with `COMMAND_EXIT_CODE="0"`.
- Completed the full LLaDA-MoE local same-backbone pair on GPUs `2/3`. Baseline `cal_lite` LCAS-v3b is `777/1033 = 75.22%`; LCAL official bounded-repair candidate is `801/1033 = 77.54%`, pairwise `31` wins, `7` losses, `770` tie-pass, and `225` tie-fail. Both logs ended with `COMMAND_EXIT_CODE="0"` and both outputs have `1033` valid rows plus `summary.json`.
- Completed trace-long-rescue Task 1/2/3 under the user override to avoid Superpowers skills, subagents, reviewer discovery, and Goal tools. The existing full-trace action brief and current action passed markdown diff hygiene.
- Implemented `analysis/trace_long_rescue_features.py` and `tests/test_trace_long_rescue_features.py`; verified the initial Task 2 test set with `Ran 3 tests` / `OK`.
- Implemented `analysis/analyze_trace_long_rescue_routes.py` and `analysis/print_trace_long_rescue_report.py`; verified the expanded Task 3 test set with `Ran 6 tests` / `OK`, py_compile for all three trace-long-rescue analysis scripts, and `git diff --check`.
- Completed serial Task 4 full trace collection. Previous local method trace run on GPU `2` verified at `769/1033 = 74.44%` with `35257` trace rows; current `midcons` trace run on GPU `3` verified at `795/1033 = 76.96%` with `35768` trace rows.
- Completed Task 5 offline Route 1/2/3 analysis on both full trace outputs. All routes failed Gate A/B; Route 1/2 had `0` triggers and Route 3 stopped without route signal, so no policy runner is justified.
- Implemented and ran CPU-only `trace_feature_audit_v2` under `superpowers:executing-plans`. Final output is `analysis_outputs/trace_feature_audit_v2_20260613_204721`; decision is `diagnostic_only`. Previous source has policy-level candidates, but current `midcons` source is diagnostic-only, so no direct full GPU policy runner is justified.
- Completed two user-approved Route 2 trace-gated long-rescue full follow-up runs using `clean_scripts/run_route2_trace_rescue.py`. Broad plateau output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958` is `801/1033 = 77.54%`, pairwise `7/1/794/231`; precision top1/conf output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958` is `800/1033 = 77.44%`, pairwise `5/0/795/233`.
- Completed the user-requested GPU3-only Route2 precision `len32` full run. The earlier GPU1 partial run was interrupted at about `405/1033` and is excluded. Clean output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516` is `801/1033 = 77.54%`, pairwise `6/0/795/232`; log `logs/paper_agent/20260614_full_route2_precision_len32_gpu3.log` ended with `COMMAND_EXIT_CODE=0`.
- Completed CPU-only Route2 error analysis Discovery V3 under the Superpowers local fallback. Added `analysis/route2_error_analysis.py` and `tests/test_route2_error_analysis.py`; output `analysis_outputs/route2_error_analysis_20260617_165806` reproduces `1033` joined rows, pairwise `6/0/795/232`, `57` triggers, `33` triggered failed-long rows, `56` missed failed-long rows, and `31/33` triggered failed-long rows with rescue length >= oracle. No GPU command was launched.
- Completed Discovery V4 method lineage audit, CPU-only implementation, and full-log run. Added `analysis/discovery_v4_signal_audit.py`, `tests/test_discovery_v4_signal_audit.py`, and `docs/paper_agent/experiments/20260618_discovery_v4_signal_audit.md`; output `analysis_outputs/discovery_v4_signal_audit_20260618_000000` has decision `route2_polish_only`. Focused tests passed with `Ran 5 tests` / `OK`; `py_compile` passed. Two leaked real-data candidates (`true_long`, `triggered_rescue_failure_*`) were caught and filtered before the final result.
- Generated the advisor-facing project summary deck and CCF-A readiness memo requested by the user. Outputs are `docs/paper_agent/presentations/20260618_advisor_project_report.pptx`, `docs/paper_agent/presentations/20260618_advisor_project_report.md`, `docs/paper_agent/presentations/20260618_advisor_project_report.html`, and `ccfa_readiness_assessment.zh.md`. The deck has `22` slides and the memo verdict is `weak_candidate`.
- Completed Route2 V6 short override full run: `802/1033 = 77.64%`, pairwise vs Route2 precision len32 `1/0/801/231`. This is the current LLaDA-Base best follow-up, but only a small selector-polish result.
- Completed proportional length widening V7 CPU audit, expanded-grid GPU smoke, runner grid-boundary fix, guard smoke, and clean full run. Clean full output is `outputs_clean/full_v7_prop_widen_expgrid_gridfix_gpu2_20260701_001430`; result is `792/1033 = 76.67%`, pairwise vs `midcons` `1` win / `4` losses / `791` tie-pass / `237` tie-fail. Proportional promotion count is `10`, true-long promotion count is `0`, long-bucket win is `0`, and short-bucket loss is `2`.
- Completed V8 proportional CAL score formula-level full runs on GPU1. V8a `outputs_clean/full_v8a_propcal_beta002_gpu1_20260701_102654` is `786/1033 = 76.09%`, pairwise vs `midcons` `3/12/783/235`; V8b `outputs_clean/full_v8b_propcal_beta004_gpu1_20260701_120525` is `781/1033 = 75.61%`, pairwise `7/21/774/231`; V8c `outputs_clean/full_v8c_propcal_beta004_cap32_gpu1_20260701_134849` is `782/1033 = 75.70%`, pairwise `6/19/776/232`. V8c cap is a reward cap, not a hard candidate-length cap.

## Current Central Claim

Inference-time length control for DLLM code infilling can safely recover medium-length under-selection by separating medium rescue from true-long detection; however, true-long infilling remains dominated by length underestimation and likely requires a stronger length-modeling signal than the current official-CAL gate family.

## Current Experiment Plan Version

`v8`: proportional CAL score full runs are complete and negative. Do not launch another GPU full run from the global proportional reward family. If continuing the ratio idea, redesign it as a local guarded policy with under-selection detection, raw-confirm/trace/probe risk guards, and explicit success/kill criteria.

## Latest Evidence And Results

- A6000 control: `787/1033 = 76.19%`.
- A6000 `midcons`: `795/1033 = 76.96%`, same-hardware `+8` wins and `0` losses.
- Long buckets unchanged: `17-24 = 20.73%`, `25+ = 16.13%`.
- Long-underestimate sweep: `16776` rules, `0` strict viable.
- Probe-curve audit: `1033/1033` rows have probe-curve features; `0/1033` rows have stopping traces.
- Probe-curve threshold sweep: `4106` single-feature thresholds, `0` strict viable.
- Best single-feature threshold: `long_score_max <= 0.229253`, `63.04%` true-long precision, `31.87%` failed-long recall, `8.70%` short-risk, `2.17%` current-pass risk.
- Interpretation: single-feature probe thresholds are informative but not GPU-safe under the current `5%` short-risk gate.
- Strict-split probe-score audit: `5` deterministic SHA256 task-id folds, `24` features, aggregate held-out `63` triggers, `47.62%` true-long precision, `32.97%` failed-long recall, `22.22%` short-risk, `7.94%` current-pass risk.
- strict_heldout_pass: `False`.
- Interpretation: the simple multivariate probe score is not GPU-safe. GPU work remains blocked until a safer offline signal or trace-enabled smoke rationale exists.
- LLaDA-Instruct cross-model validation: candidate `815/1033 = 78.90%`; historical same-backbone baseline `817/1033 = 79.09%`; interpretation is negative transfer evidence.
- DreamCoder Base smoke: valid sandbox-outside smoke passed, `2/2`, avg total sec including probe `3.1439`, pairwise against same-backbone baseline over first two tasks `0` wins / `0` losses / `2` tie-pass.
- DreamCoder Instruct smoke: valid sandbox-outside smoke passed, `2/2`, avg total sec including probe `3.0555`, pairwise against same-backbone baseline over first two tasks `0` wins / `0` losses / `2` tie-pass.
- DreamCoder Base full run: `832/1033 = 80.54%`; local same-backbone official-canvas cal_lite baseline `825/1033 = 79.86%`; pairwise `27` wins, `20` losses, `805` tie-pass, `181` tie-fail; avg total sec including probe `3.7763` versus `3.7847`. Interpretation: small local positive, not yet a strong claim.
- DreamCoder Instruct full run: `834/1033 = 80.74%`; local same-backbone official-canvas cal_lite baseline `848/1033 = 82.09%`; pairwise `21` wins, `35` losses, `813` tie-pass, `164` tie-fail; avg total sec including probe `3.8472` versus `3.8657`. Interpretation: negative transfer evidence.
- DreamCoder literature anchors: Base is above CAL DreamCoder-Base anchors (`70.2` average, `76.2` best shown) and below LR-DLLM DreamCoder-7B `81.6`; DreamOn DreamCoder `92.1` is training-based. These are anchors, not protocol-matched claims.
- Dream-7B full local pair: candidate `803/1033 = 77.73%`; local same-backbone official-canvas cal_lite baseline `802/1033 = 77.64%`; pairwise `28` wins, `27` losses, `775` tie-pass, `203` tie-fail; avg total sec including probe `3.7337` versus `3.6494`. Interpretation: near-tie/slight local positive, not a strong claim.
- Dream-7B literature anchors: candidate is above LR-DLLM Dream-7B single-line `76.7`, but DreamOn Dream-7B `88.6` is training-based and much higher. These are anchors, not protocol-matched claims.
- DiffuCoder-Base full local pair: candidate `839/1033 = 81.22%`; local same-backbone official-canvas cal_lite baseline `838/1033 = 81.12%`; pairwise `25` wins, `24` losses, `814` tie-pass, `170` tie-fail; avg total sec including probe `3.7538` versus `3.6562`. Interpretation: near-tie/slight local positive, not a strong bounded-repair improvement claim.
- DiffuCoder literature anchors: both local DiffuCoder rows are above CAL DiffuCoder-Base anchors (`68.0` average, `74.8` best shown), but DreamOn DiffuCoder-7B `92.2` is training-based and much higher. These are anchors, not protocol-matched claims.
- LLaDA-1.5 probe: local path `/tmp/llada15_probe_20260609`; architecture `LLaDAModelLM`; `model_type=llada`; config `mask_token_id=126336`; tokenizer `<|mdm_mask|>` resolves to `126336` while `tokenizer.mask_token` is `None`; six shards match expected byte sizes with total `16,031,197,144` bytes. Interpretation: runner family is likely compatible, but this is not a GPU/verifier result and gives no pass rate.
- LLaDA-1.5 full local pair: candidate `818/1033 = 79.19%`; local same-backbone `cal_lite` LCAS-v3b baseline `817/1033 = 79.09%`; pairwise `18` wins, `17` losses, `800` tie-pass, `198` tie-fail; avg total sec including probe `6.6453` versus `5.4224`. Interpretation: near-tie/slight local positive, with short-bucket regression and low official-repair true-long precision (`10.91%`).
- LLaDA-MoE full local pair: candidate `801/1033 = 77.54%`; local same-backbone `cal_lite` LCAS-v3b baseline `777/1033 = 75.22%`; pairwise `31` wins, `7` losses, `770` tie-pass, `225` tie-fail; avg total sec including probe `10.6107` versus `8.7025`. Bucket deltas are nonnegative in every oracle bucket: `<=8 +9`, `9-12 +5`, `13-16 +4`, `17-24 +6`, `25+ 0`. Interpretation: strongest current local transfer result, but slower and still not external SOTA.
- Trace feature audit v2: output `analysis_outputs/trace_feature_audit_v2_20260613_204721`; decision `diagnostic_only`; previous source `policy_candidate`; midcons source `diagnostic_only`. Strongest midcons held-out candidate: `top1_last <= 0.667969 AND max_remaining_plateau_steps >= 16`, with `18` triggers, `9` failed-long, `2` short-risk, `0` current-pass risk, and `0.500` true-long precision. This was useful diagnostic signal, but not enough by itself for automatic full GPU policy execution.
- Route 2 broad plateau full follow-up: output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958`; log `logs/paper_agent/20260613_full_route2_broad_gpu2.log`; `801/1033 = 77.54%` versus `midcons` `795/1033 = 76.96%`; pairwise `7` wins / `1` loss / `794` tie-pass / `231` tie-fail; triggers `73`; trigger true-long precision `53.42%`; avg sec including probe `5.0852`.
- Route 2 precision top1/conf full follow-up: output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958`; log `logs/paper_agent/20260613_full_route2_precision_gpu3.log`; `800/1033 = 77.44%` versus `midcons` `795/1033 = 76.96%`; pairwise `5` wins / `0` losses / `795` tie-pass / `233` tie-fail; triggers `57`; trigger true-long precision `61.40%`; avg sec including probe `5.0945`.
- Route 2 precision len32 GPU3-only follow-up: output `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516`; log `logs/paper_agent/20260614_full_route2_precision_len32_gpu3.log`; `801/1033 = 77.54%` versus `midcons` `795/1033 = 76.96%`; pairwise `6` wins / `0` losses / `795` tie-pass / `232` tie-fail; triggers `57`; trigger true-long precision `61.40%`; avg sec including probe `5.4622`. Bucket net: `<=8 +2`, `9-12 +2`, `13-16 0`, `17-24 +2`, `25+ 0`.
- Interpretation: precision is the cleaner paper-safe incremental result because it has no losses; broad has slightly larger net gain but one short loss. Neither resolves true-long: `17-24` gains only `+1/+2`, and `25+` is unchanged. Among `91` baseline failed-long rows, broad triggers `39` and rescues only `2`; precision triggers `35` and rescues only `1`, so fixed `len=24` rescue quality/length choice remains the key bottleneck.
- Route2 error analysis Discovery V3: output `analysis_outputs/route2_error_analysis_20260617_165806`; report `analysis_outputs/route2_error_analysis_20260617_165806/report.md`; joined rows `1033`; pairwise `6/0/795/232`; triggers `57`; triggered failed-long `33`; missed failed-long `56`; triggered failed-long with rescue length >= oracle `31/33`; dominant bottleneck `mixed_rescue_quality_and_gate_recall`; recommended next path `rescue_generation_quality+gate_recall`.
- Interpretation: continue signal search, but not by blindly increasing rescue length. Next CPU-first work should combine rescue generation/selection audit for triggered long failures and probe-trace fusion for missed failed-long recall.
- Discovery V4 signal audit: output `analysis_outputs/discovery_v4_signal_audit_20260618_000000`; report `analysis_outputs/discovery_v4_signal_audit_20260618_000000/report.md`; joined rows `1033`; true-long `113`; baseline failed-long `91`; decision `route2_polish_only`. Best non-leaking candidate `broad_len24_triggered >= 1` triggers `73` rows, with `4` missed failed-long, `33` triggered rescue-failure, `10` short risk, `1` current-pass risk, `0.534` true-long precision, and `3/5` stable folds; it is rejected.
- Interpretation: no GPU full run should be launched from V4 alone. Keep Route2 precision len32 as conservative polish unless a new rescue generation/selection mechanism is designed.
- V6 short override: output recorded in `docs/paper_agent/experiment_results.zh.md`; `802/1033 = 77.64%`; pairwise vs Route2 precision len32 `1/0/801/231`. Interpretation: current best LLaDA-Base follow-up, but not a true-long solution.
- V7 proportional widening: output `outputs_clean/full_v7_prop_widen_expgrid_gridfix_gpu2_20260701_001430`; log `logs/paper_agent/20260701_v7_prop_widen_full_gridfix_gpu2.log`; `792/1033 = 76.67%`; pairwise vs `midcons` `1/4/791/237`; proportional promoted rows `10`; true-long promoted rows `0`; short promoted rows `1`; long-bucket wins `0`; short-bucket losses `2`; promoted-row avg abs length error worsened from `1.1` to `6.9`. Interpretation: simple ratio widening under current parameters is negative.
- V8 proportional CAL score: outputs `outputs_clean/full_v8a_propcal_beta002_gpu1_20260701_102654`, `outputs_clean/full_v8b_propcal_beta004_gpu1_20260701_120525`, and `outputs_clean/full_v8c_propcal_beta004_cap32_gpu1_20260701_134849`; results are `786/1033 = 76.09%`, `781/1033 = 75.61%`, and `782/1033 = 75.70%`; pairwise vs `midcons` are `3/12/783/235`, `7/21/774/231`, and `6/19/776/232`. Interpretation: direct formula-level proportional reward creates a long/short tradeoff, but the losses dominate.

## Running Or Just-Ended Commands

No V8 proportional CAL score GPU command is running as of 2026-07-01 15:40 CST. The latest completed GPU action was:

- V8a output: `/home/shx/projects/dllm_infilling/git_workspace/outputs_clean/full_v8a_propcal_beta002_gpu1_20260701_102654`
- V8b output: `/home/shx/projects/dllm_infilling/git_workspace/outputs_clean/full_v8b_propcal_beta004_gpu1_20260701_120525`
- V8c output: `/home/shx/projects/dllm_infilling/git_workspace/outputs_clean/full_v8c_propcal_beta004_cap32_gpu1_20260701_134849`
- result: V8a `786/1033 = 76.09%`; V8b `781/1033 = 75.61%`; V8c `782/1033 = 75.70%`
- decision: stop global proportional scoring reward; do not launch another GPU full run without a new action brief.

There is a separate older project process still using GPUs `2/3` (`run_lcal_official_bounded_repair_a6000_v2.py` with experiment name `full_length_score_formula_P04_midconsparams_20260701`). It is unrelated to V7 and should not be interrupted.

Historical note from 2026-06-18: the prior action before later V5/V6/V7 work was documentation synthesis for the advisor report and CCF-A readiness memo:

- PPTX: `/home/shx/projects/dllm_infilling/git_workspace/docs/paper_agent/presentations/20260618_advisor_project_report.pptx`
- Markdown speaker draft: `/home/shx/projects/dllm_infilling/git_workspace/docs/paper_agent/presentations/20260618_advisor_project_report.md`
- HTML speaker draft: `/home/shx/projects/dllm_infilling/git_workspace/docs/paper_agent/presentations/20260618_advisor_project_report.html`
- CCF-A readiness memo: `/home/shx/projects/dllm_infilling/git_workspace/ccfa_readiness_assessment.zh.md`
- Verification: PPTX zip test passed, slide count is `22`, generator `py_compile` passed, and relevant markdown/html/memo files passed `git diff --check`.

Non-research cleanup note: an earlier LibreOffice format-conversion probe for `/tmp/lo_ppt_probe.html` remained stuck under PIDs `2717315`, `2717325`, `2717326`, `2717331`, `2717350`, and defunct child `2717378`. It is unrelated to the generated PPTX because the final deck was created directly by the local Python generator. A host-level kill was attempted but could not be approved because the escalation review service returned `503`; do not treat this as a project experiment.

The latest CPU-only research action before the deck was Discovery V4 signal audit:

- Discovery V4 output: `/home/shx/projects/dllm_infilling/git_workspace/analysis_outputs/discovery_v4_signal_audit_20260618_000000`
- Discovery V4 report: `/home/shx/projects/dllm_infilling/git_workspace/analysis_outputs/discovery_v4_signal_audit_20260618_000000/report.md`
- Discovery V4 final sanity: joined rows `1033`, decision `route2_polish_only`, best non-leaking candidate rejected due to `10` short-risk rows.

The latest GPU action remains the completed GPU3-only Route2 precision len32 follow-up:

- precision len32 log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260614_full_route2_precision_len32_gpu3.log`
- precision len32 output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len32_gpu3_20260614_010516`
- precision len32 final sanity: log ended with `COMMAND_EXIT_CODE=0`; `results.jsonl` has `1033` valid rows; `summary.json` exists; `step_traces.jsonl` is nonempty; result is `801/1033 = 77.54%`, pairwise `6/0/795/232`.

Previous Route 2 full follow-up runs:

- broad log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260613_full_route2_broad_gpu2.log`
- broad output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_broad_plateau_len24_gpu2_20260613_213958`
- broad final sanity: log ended with `COMMAND_EXIT_CODE=0`; `results.jsonl` has `1033` valid rows and `0` malformed rows; `summary.json` exists; `step_traces.jsonl` has `39740` rows.
- precision log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260613_full_route2_precision_gpu3.log`
- precision output: `/home/shx/projects/dllm_infilling/outputs_clean/full_route2_trace_rescue_precision_top1_conf_len24_gpu3_20260613_213958`
- precision final sanity: log ended with `COMMAND_EXIT_CODE=0`; `results.jsonl` has `1033` valid rows and `0` malformed rows; `summary.json` exists; `step_traces.jsonl` has `38872` rows.

The latest CPU-only audit before those GPU follow-ups was `trace_feature_audit_v2`:

- command: `/home/shx/miniconda3/envs/dllm_env/bin/python analysis/trace_feature_audit_v2.py ... --output-dir analysis_outputs/trace_feature_audit_v2_20260613_204721 --folds 5`
- output: `analysis_outputs/trace_feature_audit_v2_20260613_204721`
- result: compact JSON printed `decision=diagnostic_only`, `previous=policy_candidate`, `midcons=diagnostic_only`
- decision at that time: do not automatically start a full GPU policy runner from the audit alone. The later full runs were user-approved follow-up trials.

Completed current `midcons` trace run:

- tmux session: `trace_llada_base_midcons_20260612` exited.
- log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260612_full_trace_llada_base_midcons_gpu3.log`
- output: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_midcons_gpu3_20260612_180846`
- verification: log ended with `COMMAND_EXIT_CODE="0"`; `results.jsonl` has `1033` valid rows; `step_traces.jsonl` has `35768` rows linked to `1033` task ids; `summary.json` exists; pass count is `795/1033 = 76.96%`.

Completed offline route analysis:

- previous analysis: `analysis_outputs/trace_long_rescue_llada_base_prev_20260612_192611`
- midcons analysis: `analysis_outputs/trace_long_rescue_llada_base_midcons_20260612_192611`
- report: `analysis_outputs/trace_long_rescue_report_20260612`
- result: Route 1 and Route 2 trigger `0` rows on both trace sources; Route 3 stops because single-canvas traces plus no Route 1/2 signal do not justify extra multi-canvas cost. Gate A/B both fail for all routes.
- decision: no route-specific GPU policy full run should be launched from this trace batch.

Completed previous local method trace run:

- tmux session: `trace_llada_base_prev_20260612` exited.
- log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260612_full_trace_llada_base_cal_lite_lcas_v3b_gpu2.log`
- output: `/home/shx/projects/dllm_infilling/outputs_clean/full_trace_llada_base_cal_lite_lcas_v3b_gpu2_20260612_170552`
- verification: log ended with `COMMAND_EXIT_CODE="0"`; `results.jsonl` has `1033` valid rows; `step_traces.jsonl` has `35257` rows linked to `1033` task ids; `summary.json` exists; pass count is `769/1033 = 74.44%`.

Completed local verification commands before the GPU launch:

- `/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_trace_long_rescue_features.py`
- `/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/trace_long_rescue_features.py analysis/analyze_trace_long_rescue_routes.py analysis/print_trace_long_rescue_report.py`
- `git diff --check -- analysis/trace_long_rescue_features.py analysis/analyze_trace_long_rescue_routes.py analysis/print_trace_long_rescue_report.py tests/test_trace_long_rescue_features.py docs/paper_agent/experiments/20260611_trace_long_rescue_full_plan.md docs/paper_agent/current_action.md`

Completed LLaDA-MoE full local same-backbone pair:

- Baseline log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260611_1126_full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared.log`
- Baseline output: `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_cal_lite_lcas_v3b_gpu2_nofa_shared_20260611_112719`
- Candidate log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260611_1126_full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared.log`
- Candidate output: `/home/shx/projects/dllm_infilling/outputs_clean/full_lladamoe_lcal_official_bounded_repair_gpu3_nofa_shared_20260611_112740`
- Pairwise analysis: `/home/shx/projects/dllm_infilling/git_workspace/analysis_outputs/lladamoe_full_pair_20260611_1438`
- Final sanity: both logs ended with `COMMAND_EXIT_CODE="0"`, both rows files have `1033` valid rows and `0` malformed rows, both summaries exist, and the pairwise analysis has `1033` common task ids.
- Result: candidate `801/1033 = 77.54%` versus baseline `777/1033 = 75.22%`, pairwise `31/7/770/225`, avg sec including probe `10.6107` versus `8.7025`.

Historical blocked command:

- Log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1203_smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2.log`
- Partial output dirs:
  - `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_20260609_120429`
  - `/home/shx/projects/dllm_infilling/outputs_clean/smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_20260609_120635`
- Status: invalid evidence; do not use for paper comparison.
- Root causes:
  - first attempt: read-only HF dataset cache lock;
  - second attempt: sandbox blocked `multiprocessing.Manager()` listener socket in HumanEval verifier;
  - CUDA probes inside sandbox: `torch.cuda.is_available() == False`.

Required next command after explicit user approval:

```bash
script -q -e -c "DLLM_DISABLE_FLASH_ATTN=1 PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages HF_MODULES_CACHE=/tmp/hf_modules_dreamcoder_20260609 HF_DATASETS_CACHE=/tmp/hf_datasets_dreamcoder_20260609_1205 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_dreamcoder_official_infilling.py --model-path Dream-org/Dream-Coder-v0-Base-7B --max-samples 2 --mask-length-source lcal_official_bounded_repair --baseline-results /home/shx/projects/dllm_infilling/model_generalization_runs/20260513_dreamcoder_official_full/full_cal_lite_base_alpha010_cap24_official_canvas_20260513_232721/results.jsonl --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8" logs/paper_agent/20260609_1212_smoke_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed.log
```

Completed DreamCoder full commands:

- Base log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1231_full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed.log`
- Base output: `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_base_lcal_official_bounded_repair_gpu2_unsandboxed_20260609_123327`
- Base final rows: `1033`
- Instruct log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1231_full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed.log`
- Instruct output: `/home/shx/projects/dllm_infilling/outputs_clean/full_dreamcoder_instruct_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_123359`
- Instruct final rows: `1033`
- GPU status after completion: `nvidia-smi` showed no running GPU processes on 2026-06-09 14:28 CST.

Completed Dream-7B full commands:

- Baseline log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1700_full_dream_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed.log`
- Baseline output: `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_170219`
- Baseline final rows: `1033`
- Candidate log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1700_full_dream_base_lcal_official_bounded_repair_gpu3_unsandboxed.log`
- Candidate output: `/home/shx/projects/dllm_infilling/outputs_clean/full_dream_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_170219`
- Candidate final rows: `1033`
- Final sanity: both rows files have `0` malformed JSON rows, `1033` common task ids, `summary.json`, canvas `bos_prefix_masks_suffix_eos`, and backend `dreamcoder_native_diffusion_generate_fixed_canvas`.

Completed LLaDA-1.5 download/API/local-weight probe:

- Probe doc: `/home/shx/projects/dllm_infilling/git_workspace/docs/paper_agent/experiments/20260609_2105_llada15_download_api_probe.md`
- Local model path: `/tmp/llada15_probe_20260609`
- HF commit observed: `84346fd91ba60252d260022201ad6fc5a3468fb2`
- Network note: direct HuggingFace through proxy worked; `hf-mirror.com` redirected this repo back to HuggingFace and `huggingface_hub` mirror mode failed.
- Shard byte-size check: all six shards match expected sizes; total `16,031,197,144` bytes.
- API note: `LLaDAModelLM`, `model_type=llada`, config `mask_token_id=126336`, tokenizer `<|mdm_mask|>` id `126336`, `tokenizer.mask_token=None`.
- CPU/local load smoke: passed with `torch.bfloat16`; not a GPU smoke.
- GPU status at 2026-06-09 21:53 CST: GPU2 and GPU3 occupied; do not interrupt.

Completed LLaDA-1.5 full local same-backbone pair:

- Baseline log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260610_1735_full_llada15_cal_lite_lcas_v3b_gpu2_shared.log`
- Baseline output: `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_cal_lite_lcas_v3b_gpu2_shared_20260610_172705`
- Baseline final rows: `1033`
- Candidate log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260610_1735_full_llada15_lcal_official_bounded_repair_gpu3_shared.log`
- Candidate output: `/home/shx/projects/dllm_infilling/outputs_clean/full_llada15_lcal_official_bounded_repair_gpu3_shared_20260610_172720`
- Candidate final rows: `1033`
- Pairwise analysis: `/home/shx/projects/dllm_infilling/git_workspace/analysis_outputs/llada15_full_pair_20260610_1923`
- Final sanity: both logs ended with `COMMAND_EXIT_CODE="0"`, both outputs have `summary.json`, both `results.jsonl` files have `1033` valid rows, and both have `1033` common task ids.
- Result: candidate `818/1033 = 79.19%` versus baseline `817/1033 = 79.09%`, pairwise `18/17/800/198`, avg sec including probe `6.6453` versus `5.4224`.

Completed DiffuCoder-Base full commands:

- Baseline log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1925_full_diffucoder_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed.log`
- Baseline output: `/home/shx/projects/dllm_infilling/outputs_clean/full_diffucoder_base_cal_lite_alpha010_official_canvas_gpu2_unsandboxed_20260609_192508`
- Baseline final rows: `1033`
- Candidate log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260609_1925_full_diffucoder_base_lcal_official_bounded_repair_gpu3_unsandboxed.log`
- Candidate output: `/home/shx/projects/dllm_infilling/outputs_clean/full_diffucoder_base_lcal_official_bounded_repair_gpu3_unsandboxed_20260609_192533`
- Candidate final rows: `1033`
- Final sanity: both rows files have `0` malformed JSON rows, `1033` common task ids, `summary.json`, canvas `bos_prefix_masks_suffix_eos`, and backend `dreamcoder_native_diffusion_generate_fixed_canvas`.

Previously completed command:

- tmux session: `llada_instruct_midcons_20260604`
- Log: `/home/shx/projects/dllm_infilling/git_workspace/logs/paper_agent/20260604_2022_llada_instruct_midcons_full_hfmirror.log`
- Output dir: `/home/shx/projects/dllm_infilling/outputs_clean/full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23_20260604_202834`
- Final observed progress: `1033` rows in `results.jsonl`; `summary.json` exists.

```bash
cd /home/shx/projects/dllm_infilling/git_workspace
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py --model-path GSAI-ML/LLaDA-8B-Instruct --baseline-results /home/shx/projects/dllm_infilling/model_generalization_runs/20260512_114917_lcas_v3_full/full_lcas_v3_llada-instruct_resume_20260512_141851/results.jsonl --output-dir /home/shx/projects/dllm_infilling/outputs_clean --experiment-name full_lcal_official_bounded_repair_union_midcons_llada_instruct_off11_13_d3_7_r08_gpus23 --official-eval-max-s3-len 12 --repair-max-s3-len 5 --repair-min-official-len 6 --repair-max-official-len 9 --repair-min-delta 1 --repair-max-delta 8 --suspicion-max-s3-len 5 --suspicion-min-official-len 16 --suspicion-max-official-len 64 --suspicion-min-delta 1 --mid-rescue-max-s3-len 12 --mid-rescue-source base --mid-rescue-min-official-len 11 --mid-rescue-max-official-len 13 --mid-rescue-min-delta 3 --mid-rescue-max-delta 7 --mid-rescue-min-long-ratio 0.8
```

Just-ended strict-split diagnostic commands:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_analyze_probe_curve_split_score.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile analysis/analyze_probe_curve_split_score.py
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/analyze_probe_curve_split_score.py
/home/shx/miniconda3/envs/dllm_env/bin/python -c "import json; p='docs/paper_agent/probe_curve_split_score_audit.json'; d=json.load(open(p)); a=d['cross_validation']['aggregate_heldout']; assert a['strict_heldout_pass'] is False; assert a['trigger_count']==63; assert round(a['short_risk_rate'], 4)==0.2222; assert round(a['current_pass_risk_rate'], 4)==0.0794; assert round(a['true_long_precision'], 4)==0.4762; assert round(a['failed_long_recall'], 4)==0.3297; print('strict_split_assertions_ok', a)"
git diff --check -- analysis/analyze_probe_curve_split_score.py tests/test_analyze_probe_curve_split_score.py docs/paper_agent/probe_curve_split_score_audit.json docs/paper_agent/probe_curve_split_score_audit.md docs/paper_agent/probe_curve_split_score_audit.zh.md docs/paper_agent/current_action.md docs/paper_agent/pause_checkpoint.current.md docs/paper_agent/activity_ledger.en.md docs/paper_agent/activity_ledger.zh.md
```

Observed output:

- strict-split unit tests: `Ran 3 tests` and `OK`
- audit generation: `strict_heldout_pass=False heldout_triggers=63 short_risk=22.22%`
- JSON assertions: `strict_split_assertions_ok` with `trigger_count=63`, `short_risk_rate=0.2222`, `current_pass_risk_rate=0.0794`, `true_long_precision=0.4762`, `failed_long_recall=0.3297`, and `strict_heldout_pass=False`
- diff hygiene: exit `0`

## Modified Files

Intended paper-agent milestone files:

- `analysis/analyze_probe_curve_long_signals.py`
- `tests/test_analyze_probe_curve_long_signals.py`
- `docs/paper_agent/evidence_snapshot.md`
- `docs/paper_agent/experiment_plan.current.en.md`
- `docs/paper_agent/experiment_plan.current.zh.md`
- `docs/paper_agent/experiment_plan.history.en.md`
- `docs/paper_agent/experiment_plan.history.zh.md`
- `docs/paper_agent/experiment_results.en.md`
- `docs/paper_agent/experiment_results.zh.md`
- `docs/paper_agent/open_questions.en.md`
- `docs/paper_agent/open_questions.zh.md`
- `docs/paper_agent/overnight_log.en.md`
- `docs/paper_agent/overnight_log.zh.md`
- `docs/paper_agent/paper_agent_dashboard.en.md`
- `docs/paper_agent/paper_agent_dashboard.zh.md`
- `docs/paper_agent/probe_curve_signal_audit.json`
- `docs/paper_agent/probe_curve_signal_audit.md`
- `docs/paper_agent/probe_curve_signal_audit.zh.md`
- `docs/paper_agent/research_design.current.en.md`
- `docs/paper_agent/research_design.current.zh.md`
- `docs/paper_agent/pause_checkpoint.current.md`
- `analysis/analyze_probe_curve_split_score.py`
- `analysis/trace_long_rescue_features.py`
- `analysis/analyze_trace_long_rescue_routes.py`
- `analysis/print_trace_long_rescue_report.py`
- `tests/test_analyze_probe_curve_split_score.py`
- `tests/test_trace_long_rescue_features.py`
- `docs/paper_agent/activity_ledger.en.md`
- `docs/paper_agent/activity_ledger.zh.md`
- `docs/paper_agent/probe_curve_split_score_audit.json`
- `docs/paper_agent/probe_curve_split_score_audit.md`
- `docs/paper_agent/probe_curve_split_score_audit.zh.md`
- `docs/superpowers/plans/2026-05-31-probe-curve-learned-diagnostic.md`
- `analysis/discovery_v4_signal_audit.py`
- `tests/test_discovery_v4_signal_audit.py`
- `analysis_outputs/discovery_v4_signal_audit_20260618_000000/`
- `docs/paper_agent/experiments/20260618_discovery_v4_signal_audit.md`
- `docs/paper_agent/experiments/20260618_advisor_report_ppt_action.md`
- `docs/paper_agent/presentations/make_20260618_advisor_project_report.py`
- `docs/paper_agent/presentations/20260618_advisor_project_report.md`
- `docs/paper_agent/presentations/20260618_advisor_project_report.html`
- `docs/paper_agent/presentations/20260618_advisor_project_report.pptx`
- `ccfa_readiness_assessment.zh.md`

User/unrelated dirty files to preserve and not stage:

- `AGENTS.md`
- `AGENTS.zh.md`

## Uncommitted Files At Checkpoint Capture

```text
## paper-agent-overnight...origin/paper-agent-overnight [ahead 1]
 M AGENTS.md
 D AGENTS.zh.md
 M docs/paper_agent/evidence_snapshot.md
 M docs/paper_agent/experiment_results.en.md
 M docs/paper_agent/experiment_results.zh.md
 M docs/paper_agent/open_questions.en.md
 M docs/paper_agent/open_questions.zh.md
 M docs/paper_agent/overnight_log.en.md
 M docs/paper_agent/overnight_log.zh.md
 M docs/paper_agent/paper_agent_dashboard.en.md
 M docs/paper_agent/paper_agent_dashboard.zh.md
?? analysis/analyze_probe_curve_split_score.py
?? docs/paper_agent/activity_ledger.en.md
?? docs/paper_agent/activity_ledger.zh.md
?? docs/paper_agent/probe_curve_split_score_audit.json
?? docs/paper_agent/probe_curve_split_score_audit.md
?? docs/paper_agent/probe_curve_split_score_audit.zh.md
?? docs/superpowers/plans/2026-05-31-probe-curve-learned-diagnostic.md
?? tests/test_analyze_probe_curve_split_score.py
```

Captured older pause-state status for the prior probe-curve audit milestone:

```text
## paper-agent-overnight...origin/paper-agent-overnight
 M AGENTS.md
 D AGENTS.zh.md
 M docs/paper_agent/evidence_snapshot.md
 M docs/paper_agent/experiment_plan.current.en.md
 M docs/paper_agent/experiment_plan.current.zh.md
 M docs/paper_agent/experiment_plan.history.en.md
 M docs/paper_agent/experiment_plan.history.zh.md
 M docs/paper_agent/experiment_results.en.md
 M docs/paper_agent/experiment_results.zh.md
 M docs/paper_agent/open_questions.en.md
 M docs/paper_agent/open_questions.zh.md
 M docs/paper_agent/overnight_log.en.md
 M docs/paper_agent/overnight_log.zh.md
 M docs/paper_agent/paper_agent_dashboard.en.md
 M docs/paper_agent/paper_agent_dashboard.zh.md
 M docs/paper_agent/research_design.current.en.md
 M docs/paper_agent/research_design.current.zh.md
?? analysis/analyze_probe_curve_long_signals.py
?? docs/paper_agent/pause_checkpoint.current.md
?? docs/paper_agent/probe_curve_signal_audit.json
?? docs/paper_agent/probe_curve_signal_audit.md
?? docs/paper_agent/probe_curve_signal_audit.zh.md
?? tests/test_analyze_probe_curve_long_signals.py
```

## Known Risks

- Current improvement is small and heuristic; it is not a CCF-A-level central claim yet.
- Long buckets remain unchanged despite `midcons`.
- Current outputs lack stopping traces, so trajectory diagnostics require a trace-enabled smoke run later.
- Single-feature probe-curve thresholds fail the short-risk gate.
- The first simple strict-split multivariate probe score also fails the short-risk gate: `22.22%` held-out short-risk versus the `5%` gate.
- Independent subagent code review is blocked in this environment; only local diff review plus tests were used.
- Future GPU experiments must use cards `2,3` and must not interrupt other jobs.
- Current IDE sandbox cannot produce valid GPU/verifier results for HumanEval experiments; long GPU/verifier runs must be launched outside the sandbox, as the current DreamCoder full runs were.

## Open Questions

- Can constrained, nonlinear, trace-aware, or cross-run probe scoring reduce short-risk to `<=5%` while retaining at least `10` failed-long triggers? The first simple strict-split linear score failed with `22.22%` held-out short-risk.
- Can trajectory features detect true-long under-selection, or is training-time length regularization required?
- Is the paper best framed as a positive method paper, a diagnostic-plus-method paper, or a rigorous negative result motivating dynamic canvas or length regularization?
- For missing backbones without local baselines, should the first full run be treated only as exploratory until a same-backbone local baseline is generated?
- How should paper-critical raw result files be stored if a future run becomes central?

## Workflow / Skill Status

| Workflow / Skill | Status | Evidence | Output files | Notes |
|---|---|---|---|---|
| gstack `/office-hours` | completed | `research_design.initial.*.md` and `research_design.current.*.md` contain the research-community user, need, and minimum publishable contribution mapping; log entry `2026-05-31 12:39 CST` records paper-agent document initialization | `docs/paper_agent/research_design.initial.en.md`, `docs/paper_agent/research_design.initial.zh.md`, `docs/paper_agent/research_design.current.en.md`, `docs/paper_agent/research_design.current.zh.md` | No separate CLI transcript; completion is evidenced by the output research design docs |
| gstack `/plan-ceo-review` | completed | `research_design.current.en.md` includes `CEO-Style Stress Review` covering novelty, importance, reviewer appeal, scope, central claim, weakest assumption, and CCF-A realism | `docs/paper_agent/research_design.current.en.md`, `docs/paper_agent/research_design.current.zh.md` | Current conclusion: credible foothold, not yet a CCF-A claim |
| gstack `/plan-eng-review` | completed | `experiment_plan.current.en.md` includes engineering review, datasets, baselines, metrics, compute budget, reproducibility, failure modes, and kill criteria | `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/experiment_plan.current.zh.md`, `docs/paper_agent/experiment_plan.history.en.md`, `docs/paper_agent/experiment_plan.history.zh.md` | Current version is `v3` |
| Superpowers `brainstorming` | not_started | no evidence found | none | The current direction was carried by the gstack-style design docs; run before any future creative spec change |
| Superpowers `writing-plans` | completed | Current context records the skill was read; `experiment_plan.current.*.md` and history provide the executable plan | `docs/paper_agent/experiment_plan.current.en.md`, `docs/paper_agent/experiment_plan.current.zh.md`, `docs/paper_agent/experiment_plan.history.en.md`, `docs/paper_agent/experiment_plan.history.zh.md` | No separate Superpowers plan file was created |
| Superpowers `systematic-debugging` | completed | The first 2026-06-04 JSON assertion failed with `KeyError: 'strict_heldout_pass'`; inspecting the JSON and script showed the assertion used the wrong schema path | `docs/paper_agent/current_action.md` | Root cause: aggregate metrics live under `cross_validation.aggregate_heldout`; no metric changed |
| Superpowers `verification-before-completion` | completed | Log entries `2026-05-31 21:47 CST` and `2026-05-31 22:11 CST` record completed probe-curve audit verification; 2026-06-04 records strict-split unit test, py_compile, audit regeneration, corrected JSON assertions, and `git diff --check` | `docs/paper_agent/current_action.md`, `docs/paper_agent/paper_agent_dashboard.en.md`, `docs/paper_agent/probe_curve_split_score_audit.md` | Key verification: probe-curve audit `Ran 8 tests` / `OK`; strict-split diagnostic `Ran 3 tests` / `OK` and `strict_heldout_pass=False` |
| Superpowers `requesting-code-review` | blocked | Log entry `2026-05-31 21:47 CST` records that no independent Task/subagent reviewer tool was visible; local diff review plus fresh tests were used as fallback | `docs/paper_agent/overnight_log.en.md`, `docs/paper_agent/overnight_log.zh.md` | Independent reviewer was not completed; residual risk is documented |

## Next Resume: First 3 Actions

1. Preserve unrelated user changes and do not stage unrelated dirty files with paper-agent docs.
2. If preparing a commit, stage only the advisor report files, Discovery V4 files, and updated paper-agent docs that belong to this milestone.
3. Do not launch a GPU full run from Discovery V4; the next research step needs a new action brief around rescue generation/selection quality or principled length modeling.

## Recommended Resume Prompt

```text
Continue the paper-agent work in /home/shx/projects/dllm_infilling/git_workspace on branch paper-agent-overnight in low-token mode. Do not call create_goal/update_goal/get_goal. First read AGENTS.md, git status, docs/paper_agent/pause_checkpoint.current.md, docs/paper_agent/current_action.md, paper_agent_dashboard.zh.md, and the tail of activity_ledger.zh.md. The advisor report deck is at `docs/paper_agent/presentations/20260618_advisor_project_report.pptx`, with speaker Markdown/HTML beside it, and the CCF-A memo is `ccfa_readiness_assessment.zh.md`. Discovery V4 final decision is `route2_polish_only`; do not launch a GPU full run from it. Preserve unrelated user changes. If continuing research after the advisor report, write a new action brief for rescue generation/selection quality or principled length modeling before any GPU work. Do not claim SOTA; keep literature anchors separate from local protocol-matched evidence.
```
