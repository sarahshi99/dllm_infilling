# H1正式训练（2026-11-07）

状态：ready_for_background。数据实际4/220000；源程序组train/validation={'train': 1, 'validation': 0}。

缺口总计219996，详见data_summary.json；保留真实K2/K4及32桶，原分组隔离，external_test未打开。

FP32全词表KL，rank256，强度1；seed20261107，W_out=0、gamma=1；AdamW lr3e-4，有效batch128，最多5轮。



|组|通过/N|通过率|遗漏|

|---|---:|---:|---:|

|H1|0/0|N/A|1033|

|Z4|820/1033|0.7938044530493708|0|

|H4-old|842/1033|0.8151016456921588|0|

|K1|949/1033|0.9186834462729913|0|

H1−Z4：配对0，救回0、改错0；差值百分点评估区间None。

H1−H4-old：配对0，救回0、改错0；差值百分点评估区间None。

H1−K1：配对0，救回0、改错0；差值百分点评估区间None。

纯学习恢复比例：{'matched': 0, 'numerator': 0, 'denominator': 0, 'value': None, 'reason': 'nonpositive denominator'}。



大缓存、生成、评分、失败日志及优化器只在服务器 /home/shx/.cache/dllm_infilling/context_h1_training_20261107_v1；精确位置见evidence_index.json。

共享唯一cuda:0，单GPU进程，阶段释放DreamOn，不等待独占；时间仅观测值，不称公平加速。

其余结构可直接用bank.Bank读取同一queue.sqlite，不重采teacher；训练日程/初始化规则见train.py及README.zh.md。

本轮没有训练H0/Hctx/Hgated/H2/H3/H4，没有自身轨迹补训、超参搜索或多seed。

代码提交：bac2b3d9add68a8ac9dca9922441199da1be0886；恢复：`bash experiments/context_h1_training_v1/launch.sh`。
