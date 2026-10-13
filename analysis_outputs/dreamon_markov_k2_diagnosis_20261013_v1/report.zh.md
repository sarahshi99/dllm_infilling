# DreamOn + Markov K2 归因与机制诊断

完整 10330/10330 行；1033 行/164 基础题是反复使用的开发/机制诊断人口，不是新的独立测试。单生成种子 42，95% 区间仅表达这一生成种子条件下的基础题重采样不确定性。

A=K1 无头，B=K2 无头，C=同路径零输出头 λ=1，D=v2 KL λ=1，E=固定右位只提交左词后重算。

|策略|组|通过/1033|通过率|生成总秒|中位/P95秒|输出词元总计|基础/额外/总前向|普通提交/总前向|相对A速度|
|---|---|---|---|---|---|---|---|---|---|
|global_confidence|A|951/1033|92.06%|215.98|0.1706/0.4355|9135|10192/0/10192|0.896|1.000×|
|global_confidence|B|901/1033|87.22%|126.17|0.0968/0.2306|9268|5921/0/5921|1.549|1.712×|
|global_confidence|C|901/1033|87.22%|129.75|0.0993/0.2403|9268|5921/0/5921|1.549|1.665×|
|global_confidence|D|910/1033|88.09%|130.31|0.0997/0.2367|9287|5947/0/5947|1.548|1.657×|
|global_confidence|E|935/1033|90.51%|200.37|0.1607/0.3815|9161|5834/3902/9736|0.933|1.078×|
|left_to_right_frontier|A|942/1033|91.19%|221.08|0.1703/0.4531|9517|10549/0/10549|0.902|1.000×|
|left_to_right_frontier|B|877/1033|84.90%|116.43|0.0964/0.2192|9265|5428/0/5428|1.707|1.899×|
|left_to_right_frontier|C|872/1033|84.41%|120.94|0.1003/0.2289|9298|5449/0/5449|1.706|1.828×|
|left_to_right_frontier|D|890/1033|86.16%|121.58|0.1002/0.2300|9316|5477/0/5477|1.701|1.818×|
|left_to_right_frontier|E|931/1033|90.13%|201.89|0.1615/0.4114|9209|5446/4413/9859|0.934|1.095×|

E 的速度只表示诊断成本；E 不是验证器、数学上界或 K1 的等价实现，global 没有重选右位置。

|策略|配对|帮助/伤害|净行数|行加权差及95%CI(pp)|基础题等权差及95%CI(pp)|
|---|---|---|---|---|---|
|global_confidence|D-C|15/6|+9|+0.87 [+0.00,+1.79]|+1.16 [+0.21,+2.28]|
|global_confidence|C-B|0/0|+0|+0.00 [+0.00,+0.00]|+0.00 [+0.00,+0.00]|
|global_confidence|E-C|35/1|+34|+3.29 [+2.20,+4.41]|+2.46 [+0.61,+4.07]|
|global_confidence|E-D|28/3|+25|+2.42 [+1.35,+3.50]|+1.30 [-0.38,+2.65]|
|global_confidence|B-A|5/55|-50|-4.84 [-6.37,-3.42]|-4.57 [-7.12,-2.13]|
|global_confidence|C-A|5/55|-50|-4.84 [-6.37,-3.42]|-4.57 [-7.12,-2.13]|
|global_confidence|D-A|9/50|-41|-3.97 [-5.42,-2.54]|-3.42 [-5.81,-1.06]|
|global_confidence|E-A|7/23|-16|-1.55 [-2.55,-0.59]|-2.11 [-4.72,+0.23]|
|left_to_right_frontier|D-C|29/11|+18|+1.74 [+0.62,+2.94]|+1.04 [-1.03,+3.05]|
|left_to_right_frontier|C-B|6/11|-5|-0.48 [-1.31,+0.36]|+0.11 [-1.12,+1.70]|
|left_to_right_frontier|E-C|64/5|+59|+5.71 [+4.00,+7.60]|+5.35 [+2.20,+8.69]|
|left_to_right_frontier|E-D|48/7|+41|+3.97 [+2.61,+5.40]|+4.31 [+1.24,+7.61]|
|left_to_right_frontier|B-A|10/75|-65|-6.29 [-8.24,-4.44]|-6.36 [-9.91,-2.98]|
|left_to_right_frontier|C-A|11/81|-70|-6.78 [-8.80,-4.88]|-6.25 [-10.07,-2.57]|
|left_to_right_frontier|D-A|10/62|-52|-5.03 [-6.67,-3.48]|-5.21 [-8.64,-2.11]|
|left_to_right_frontier|E-A|7/18|-11|-1.06 [-2.05,-0.10]|-0.90 [-2.76,+0.96]|

训练出的头究竟有没有帮忙？
global_confidence：D−C 净 +9，C−B 净 +0；本轮配对点估计支持训练修正有额外贡献；行加权95%区间包含零，稳定性仍不确定。本轮 D−B 净变化为这两项之和，不能只凭旧 D−B 分配归因，也不能据此说过去所有收益都是假的。
left_to_right_frontier：D−C 净 +18，C−B 净 -5；本轮配对点估计支持训练修正有额外贡献；基础题等权95%区间包含零，稳定性仍不确定。本轮 D−B 净变化为这两项之和，不能只凭旧 D−B 分配归因，也不能据此说过去所有收益都是假的。

公平计时后到底快多少？
global_confidence/B：相对 A 1.712×，纯生成时间减少 41.58%。
global_confidence/C：相对 A 1.665×，纯生成时间减少 39.92%。
global_confidence/D：相对 A 1.657×，纯生成时间减少 39.66%。
left_to_right_frontier/B：相对 A 1.899×，纯生成时间减少 47.34%。
left_to_right_frontier/C：相对 A 1.828×，纯生成时间减少 45.29%。
left_to_right_frontier/D：相对 A 1.818×，纯生成时间减少 45.01%。

让模型重新算，能不能明显救回来？
global_confidence：E−C 净 +34，E−D 净 +25；这两项行加权区间均在零以上，支持本轮覆盖右位仍有恢复空间。
按基础题等权时至少一项95%区间包含零，因此不能把行加权证据称为对所有题目权重都稳定的改善。
left_to_right_frontier：E−C 净 +59，E−D 净 +41；这两项行加权区间均在零以上，支持本轮覆盖右位仍有恢复空间。
两项基础题等权95%区间也均在零以上，基础题等权结果同样支持恢复空间。

|策略|方法|A正确/B错误人口救回|其他人口新增伤害（相对B）|
|---|---|---|---|
|global_confidence|C|0/55|0/978|
|global_confidence|D|11/55|6/978|
|global_confidence|E|33/55|1/978|
|left_to_right_frontier|C|3/75|11/958|
|left_to_right_frontier|D|21/75|11/958|
|left_to_right_frontier|E|59/75|6/958|

下一轮最值得改哪一件事，为什么？
优先改进头对提交左词后右位变化的拟合：真实重算在本轮覆盖位置给出额外恢复证据。下一轮需另行授权。

本次按用户后续授权在共享 H200 上运行，其他进程可同时存在。配置按行轮换且记录 GPU 背景负载，纯生成计时仍可能受到外部负载变化影响，不能当成空卡专用测量。

所有相邻机会、头调用、重算调用、结构停止、扩展/删除、非相邻回退分母分列于 summary.csv。普通提交计数与输出长度分开；总前向包含 E 的额外前向。bootstrap 按基础题整组抽样 10000 次，种子 20261013；不追加生成种子或调参。

实际设备/精度：NVIDIA H200 NVL / bfloat16 model, float32 rank-256 heads (same as previous)。加载和预热在计时外，decode 前后 CUDA 同步；评测与写盘分开计时。GPU 背景记录见 gpu_background.jsonl。

代码 `codex/dreamon-markov-k2-diagnosis-v1@a55b1fd78795ae5f83845302a76f31780dda8cd2`，完整性 `passed`；旧报告和被替代判断保留。

运行命令：

```bash
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python experiments/run_dreamon_markov_k2_diagnosis.py --original-config /home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-markov-k2-eval-v1/analysis_outputs/dreamon_markov_k2_full_20261011_v1/run_config.json --output-dir analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1 --budget-gpu-hours 12 --allow-shared-gpu --min-free-gib 32
```
