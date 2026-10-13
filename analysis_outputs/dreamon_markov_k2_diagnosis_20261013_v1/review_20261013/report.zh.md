# K2 诊断 CPU 独立复核报告（2026-10-13）

本轮起点 HEAD 为 `340170ce6ae44477027dd7e94ec7a9c0f0910384`，实际核对一致，未回滚。只读取已经存在的成功记录；没有运行模型、头、生成器、代码评测或任何GPU实验，没有停止其他进程。

## 复核结论

完整性为10×1033=10330行、164基础题；缺失、重复、未恢复的已观察运行失败均为0。SQLite只读payload与原per_case JSONL逐字段完全一致。导出原32字段、完整输出文本与token数组，gzip约940KiB；没有从summary展开或补造结果。

隔离临时目录只复制两个CPU分析源码与evidence文件夹，cwd设为无关目录；SQLite连接、torch和transformers导入均被显式禁止。相对路径证据可重建全部统计，随后另一份独立计算以math.fsum/statistics重算聚合，并按注册的NumPy PCG64基础题整组抽样复算区间。已比较1588单元格：原summary/paired与两种复算、原报告三个数值表，差异0。

浮点比较使用原CSV完整精度，允许不同求和顺序≤1e-14×max(1,|原值|)的机器舍入；整数计数精确相等；CI端点额外核对原精度正/负/精确零，不依据显示的0.00判断。见[逐单元格比对](all_cell_comparison.csv)与[审计](reproduction_audit.json)。

## 结果与分母

|策略|A|B|C|D|E|D相对A共享速度观测|
|---|---:|---:|---:|---:|---:|---:|
|global_confidence|951/1033|901/1033|901/1033|910/1033|935/1033|1.657×|
|left_to_right_frontier|942/1033|877/1033|872/1033|890/1033|931/1033|1.818×|

D−C：global帮助/伤害15/6、净+9；frontier 29/11、净+18。C−B：global 0/0、净0；frontier 6/11、净−5。D−B净+9/+13，相对B的并行净损失追回比例为9/(951−901)=18%、13/(942−877)=20%。这不是A对/B错子集的救回比例：后者D为11/55、21/75，不能混用。

E−D净+25/+41，E−C净+34/+59。E只在原触发位置、固定原右位、只填左词后重算，不是K1等价实现、数学上界或已成功的验证算法。

单生成种子42；10000次bootstrap种子20261013，以164基础题为整组单位；行加权与基础题等权同时报告。重复开发/机制诊断人口不是独立封存测试。重要区间原精度见表（单位为比例，不是百分点）：

|策略|比较|行加权95%CI|基础题等权95%CI|
|---|---|---|---|
|global_confidence|D-C|[0.0, 0.01794616151545364]|[0.0020831903530074258, 0.02277200145721486]|
|global_confidence|E-D|[0.013527550307399197, 0.03499102209944751]|[-0.0038378185740589456, 0.026512390316145135]|
|left_to_right_frontier|D-C|[0.006233072719412659, 0.029412486289903585]|[-0.010347251833532321, 0.03045249423679301]|
|left_to_right_frontier|E-D|[0.026136107698607698, 0.05402650356778797]|[0.012442893798096783, 0.07613783918130436]|

global D−C行加权下界是精确0，基础题等权下界为正；frontier D−C行加权为正，基础题等权跨零。global E−D基础题等权跨零，frontier E−D两种权重均为正。不能把所有改善都称为对权重选择稳定。

## 评测超时与异常

总计9110条通过、1220条评测失败，其中13条`timed out`、1207条`failed:`。运行失败journal不存在与评测失败是不同层次；“未恢复错误0”不等于“超时0”。没有未知评测结果字符串，也没有passed与failure_reason不一致。

|策略|A超时|B超时|C超时|D超时|E超时|
|---|---:|---:|---:|---:|---:|
|global_confidence|2|0|0|1|1|
|left_to_right_frontier|3|0|1|2|3|

13次全部集中在HumanEval/32的L0/L1/L2/L4，涉及A/C/D/E，B没有超时；本轮不是均匀随机分散的超时模式。各超时评测墙钟约3.09–3.12秒，没有发现相同task_id且相同输出文本在另一记录通过的反例。参见[逐例超时](timeout_cases.csv)、[分组](evaluation_groups.csv)与[原错误文本计数](evaluation_reasons.csv)。

已固定评测器版本`88062ff9859c875d04db115b698ed4b0f0395170`，本地checkout HEAD只读核对一致；沿用已记录的source/enablement overlay两个digest，没有改评测器、重新评测或打开新测试。源码调用是check_correctness(...,3.0,0)，子进程join等待timeout+1，内部time_limit使用实际墙钟SIGALRM。result为空的子进程异常和真实内部计时超时都可能被归为`timed out`；`failed:`只保存异常消息，不能据空消息制造AssertionError类型。

共享资源可能通过CPU调度/墙钟计时影响评测标签，现有数据不能排除。GPU快照只有总利用率/显存/任务数量，不包含CPU负载；每例没有evaluator开始/结束UTC、子进程CPU时间、退出码/信号、异常类型/traceback或超时来源。因此无法把13次超时归因为算法不终止、慢算法或资源争用。集中同一基础题是事实，不是排除资源影响的证明。

最小后续检查（本轮未执行，需另行授权）：只对这13条保存输出在CPU隔离环境用同一固定评测器和3秒参数评测，记录起止UTC、CPU/墙钟、主机负载及子进程退出码和超时来源；不重新生成、不重跑全量质量实验。

另存[超时敏感性边界](timeout_sensitivity.csv)：若唯一不确定标签是现有13次超时，D−C的净增区间为global[9,10]、frontier[17,20]，E−D为[24,26]/[39,44]。这只是将已有超时视为未知时的代数上下界，不是新通过状态，不用于重算原通过率或显著性。

## GPU背景与速度证据界限

匿名化原GPU时序共有1052条，时刻2026-10-13T01:38:09Z至2026-10-13T02:32:52Z；包含前次阻塞、预加载、smoke和full。每个快照均有6个其他任务，显存总占用91761–109505MiB，GPU总利用率0–96%，中位0%。其中1049个快照可识别本作业存在（01:44:58Z至02:32:52Z）。

其他任务常驻并不证明它们始终在计算；总利用率也无法归属到某个任务。快照是在行边界/部分smoke时采样，不是连续轨迹，不能精确关联某条评测。导出时删除其他用户姓名、PID、路径、命令和GPU UUID，只保留匿名数值；原件留本地。已有两份run_manifest的GPU进程路径也作隐私删节，原件留本地ignored目录，不重写Git历史。

1.657×/1.818×是共享H200上的纯生成速度比观测。排除了模型加载/预热、评测、写盘，采用CUDA同步和按行轮换，但轮换不能替代独占计时，不能称最终公平加速。旧报告的“公平计时后到底快多少”标题过宽，见[带日期勘误](errata.20261013.zh.md)，旧报告保留。file_write_sec还仅覆盖第一次SQLite INSERT/commit，不能解释为全量写盘时间。

## 可复制命令与保存

Python>=3.10，NumPy验证版本2.2.6（evidence/requirements.txt）；命令在仓库根目录执行，不加载模型：

```bash
python analysis/analyze_dreamon_markov_k2_diagnosis.py \
  --result-dir analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1 \
  --output-dir /tmp/k2_recomputed

python analysis/review_dreamon_markov_k2_diagnosis.py \
  --result-dir analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1 \
  --output-dir /tmp/k2_independent_review
```

直接复制evidence文件夹也可用`--evidence-dir`运行；不需要SQLite、原作者绝对路径、模型缓存、分词器或原始数据。原summary.csv、paired_comparisons.csv、smoke_checks.json、background_exit.json和旧report内容保留；复算产物另存。

最终归档脚本已增加只读导出、GPU隐私删节和evidence文件入git；脚本本轮仅做语法/源码检查，没有执行后台归档入口或重启实验。独占计时仅给出[后续资源条件与命令](exclusive_timing_future.zh.md)，未执行。

验证：7项新证据/统计测试及1项既有完整十组分析测试通过；独立隔离复算再次通过，1588单元格差异0；py_compile、git diff --check通过。生成器与runner文件无修改，旧报告和表格无修改。reviewer_gate_disabled，使用本地diff review与fresh CPU verification。

复核源码/证据的最终版本由包含本报告的Git提交确定；提交/推送后的完整HEAD在本轮最终回复给出，避免在提交内容中伪造自引用提交号。
