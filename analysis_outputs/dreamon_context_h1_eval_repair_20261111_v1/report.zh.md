# H1正式best完整评测补跑
状态：complete；1033行/164题完整：True。
复用H1 epoch5正式best，不训练、不修改权重。修复BF16→FP32接口，原1027条技术失败重跑，6条成功路径复用；原始失败记录和输出完整保留。
SingleLine/global/K4，seed42，T=.2，top_p=.9，batch1；A公共基线3099条直接复用，模型、随机输入、FP32采样、选位、结构屏障和官方评分不变。
|组|通过/N|通过率|
|---|---:|---:|
|H1|843/1033|81.61%|
|H2|851/1033|82.38%|
|Z4|820/1033|79.38%|
|H4-old|842/1033|81.51%|
|K1|949/1033|91.87%|

H1−Z4：救回47、改错24、净差23；百分点区间[0.4844610156162229, 4.011741682974559]。
H1−H4-old：救回26、改错25、净差1；百分点区间[-1.3850735531308085, 1.5384985563041382]。
H1−K1：救回8、改错114、净差-106；百分点区间[-12.707320580248757, -8.04385117947763]。
H1−H2：救回29、改错37、净差-8；百分点区间[-2.4317412463103376, 0.8983388644397932]。

纯学习恢复比例：{'matched': 1033, 'numerator': 23, 'denominator': 129, 'value': 0.17829457364341086, 'reason': 'defined', 'ci95_positive_denominator': [0.04132231404958678, 0.3035747354497354], 'bootstrap_valid': 10000, 'bootstrap_total': 10000}。
配对区间以164基础题分组计算；共享GPU耗时仅作观测。
H2最佳验证KL从0.29015降至0.11201（下降61.4%），部署采样与teacher不一致率从7.63%降至4.585%；但1033行程序通过率仅82.38%。KL是预测分布目标，不保证程序通过。H2相对零头救回56/改错25，净增31；相对旧静态头救回38/改错29，净增9，区间跨0；相对K1救回8/改错106，净少98，恢复比例24.0%。现有证据只能说明有限学习收益，不能确定缺口成因，不能宣称多词信息因果有效。
证据与恢复：`cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-context-h1-eval-repair-20261111-v1 && bash experiments/context_h1_eval_repair_v1/launch.sh`；日志 `/home/shx/.cache/dllm_infilling/context_h1_eval_repair_20261111_v1/pipeline.log`；逐行结果 `/home/shx/.cache/dllm_infilling/context_h1_eval_repair_20261111_v1/per_case.jsonl.gz`。
