# H3 上下文＋短记忆注意力头正式训练

状态：ready_for_background；完整1033评测：False。

公共库：H1 formal 20261107（9aff455），原地只读，不采集、不复制。实际规模、源组及32桶见data_summary.json。

实现：B fca5775 的 heads.py 原样；一层短记忆注意力，总宽256、4头，learned-attention；读取当前链全部此前已提交普通词及其H0/位置关系。K2最多1条、K4最多3条，无跨案例/链/轮缓存。参数量见technical_checks.json。

初始化seed20261107，只有W_out置零，内部随机；不继承任何旧权重/优化器。

共享H1 train.py：FP32完整词表T1 KL(fresh||old+delta)，rank256、强度1，AdamW lr3e-4/betas(.9,.999)/eps1e-8/wd0/clip1，固定有效批128，最多5轮。

顺序default_rng(20261107+epoch)，每轮尾批64仍按实际样本数归一化；精度、损失、teacher、源划分和H1一致。

修复：KeywordHeadAdapter将DreamOn的BF16 old_hidden转FP32，匹配缓存训练；H1工作区未改。

固定best：SingleLine/global/K4/1033行/seed42/T=.2/top_p=.9/batch1，完全复用A采样与结构动作。

|组|通过/N|通过率|缺失|

|---|---:|---:|---:|

|H3|0/0|None|1033|

|Z4|820/1033|0.7938044530493708|0|

|H4-old|842/1033|0.8151016456921588|0|

|K1|949/1033|0.9186834462729913|0|

H3−Z4：配对0；救回0、改错0，净差0；百分点区间None。

H3−H4-old：配对0；救回0、改错0，净差0；百分点区间None。

H3−K1：配对0；救回0、改错0，净差0；百分点区间None。

纯学习恢复比例：{'matched': 0, 'value': None, 'reason': 'no_matched_triples'}（分母非正不报告）。

H1配对：{'status': 'pending', 'reason': 'H1 has 6/1033 scored lines; separate repair needed'}。H2配对：{'status': 'pending', 'reason': 'H3 training not started'}。

资源计数在generation_comparison.json；完整主模型forward、实际提交词、头调用、注意力调用/目标行/有效历史记录及结构动作分别报告，不宣称公平共享GPU加速。

缺少同数据Hctx及必要消融，不能宣称多词信息有效或选择性注意力优于简单平均。单历史权重为1；注意力权重不能直接作为因果解释。共享GPU时间仅为观测。

代码：a8c3088318b3b8ee118f2de06075d410f6810e3d；服务器证据：/home/shx/.cache/dllm_infilling/context_h3_training_20261111_v1；精确路径见evidence_index.json。

恢复（保留checkpoint及数据游标，不重训已完成阶段）：`cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-context-h3-training-20261111-v1 && bash experiments/context_h3_training_v1/launch.sh`。
