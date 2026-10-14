# 2026-10-14 批次B：dreamcoder_base

官方single-line数据，当前诊断生成设置。全人口1033填空行/164基础题；共享164基础题开发评测，不是独立未见测试。seed42，主模型与v2 KL/full/best rank256头冻结，lambda1。

状态：complete；逻辑行6198/6198，实际新生成6198，旧证据复用0，逻辑别名0；未完成0。

|策略|组|N|通过|通过率|实际提交/前向|平均NFE|新生成秒/行|平均头调用|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|global_confidence|A|1033|307|0.2972|4.0000|16.0000|0.3380|0.0000|
|global_confidence|C|1033|309|0.2991|4.0000|16.0000|0.3714|36.4560|
|global_confidence|D|1033|346|0.3349|4.0000|16.0000|0.3717|36.2623|
|left_to_right_frontier|A|1033|193|0.1868|4.0000|16.0000|0.3356|0.0000|
|left_to_right_frontier|C|1033|184|0.1781|4.0000|16.0000|0.3818|47.9129|
|left_to_right_frontier|D|1033|278|0.2691|4.0000|16.0000|0.3807|47.9167|

组定义：批次A的A/B/C/D分别为原生K4/链规则无头K4/零头K4/训练头K4；批次B的A/C/D分别为原生无头/零头/来源头K4；批次C的A/B/C/D分别为K1无头/K2无头/K2零头/K2训练头。C的K2无头与零头还包含既有提交规则及局部重采样差别，主要学习收益看D-C。

配对救回/改错与ID见paired_comparisons.csv；旧K2 DreamOn锚点只纳入可证明不受共享扩展边界修复影响的行，表内N明确披露。普通断言、语法、运行异常及官方3秒超时均保留为有效评分，没有反复重采样。

新生成耗时在CUDA同步边界测量，排除加载、官方执行和写出，包含离散轨迹缓冲及事件记录；trace_record_cpu_sec只覆盖Python记录组装，不隔离整个日志传输成本。head_projection_cuda_sec只测小头前向，head_serial_cuda_sec包含小头及局部采样。共享H200负载为匿名时序观测，不是公平加速测量；不能仅据K或NFE宣称加速。旧A4耗时含官方评测且无同步边界，未混进新计时均值。

实际每前向提交数仅计选中位置直接提交，广播EOS额外删除单列delete_count；名义K不等于实际提交数。新行全部轮记录离散选位、草稿/最终提交ID、实际前驱链边/局部seed及动作/停止原因；不保存全词表logits/隐状态。旧复用轨迹缺口逐行注明，不能据此声称完整机制归因。

Base保持固定64画布、EOS/pad原位和原生清理，完整画布top-k同分规则；64步时间表/eps0.001/EOS罚项3.0不变。K4约16次和K2约32次填满时的时钟与EOS罚项进度不同，这是跨K混杂。DreamOn64/64是参数；仍沿用扩展预算来自max_gen_len的语义，未改生成协议为另一套多行论文设置。

观察到的配置耗时差异（只比较本轮有新计时的行；各组可用N不同）：
global_confidence：A=0.3380s/行(新N=1033)；C=0.3714s/行(新N=1033)；D=0.3717s/行(新N=1033)
left_to_right_frontier：A=0.3356s/行(新N=1033)；C=0.3818s/行(新N=1033)；D=0.3807s/行(新N=1033)

证据：per_case.jsonl.gz、../single-line_contexts.jsonl.gz或../multi-line_contexts.jsonl.gz、run_manifest.json、summary.csv、paired_comparisons.csv、integrity_audit.json、smoke_checks.json及匿名gpu_background.jsonl。
