# B 首次七头统一 KL 训练结果（2026-10-28）

**完成**：七头各1 epoch、144步，总1008步；train2304/validation768，源题组690/233，无组交集。
训练代码提交 `a6541f6d0907cd10a8e98b826927fdaf1bc4d47f`；单worker共用GPU0 H200，FP32。worker墙时89.13秒，包括验证和写盘，不是推理测速。
没有加载/重跑DreamOn、采新teacher或生成HumanEval答案；其他GPU进程未中断。

共同旧 logits：validation T1 KL=0.373785，TV=0.222129。以下全部为768状态上的相同离线教师目标，非代码通过率。

|模型|初始KL|训练后KL|训练后TV|源题组等权KL|best step|peak allocated MiB|
|---|---:|---:|---:|---:|---:|---:|
|H0-static|0.373661|0.370023|0.219775|0.310825|144|1578|
|Hctx-old-context-only|0.378788|0.350822|0.202859|0.291966|144|856|
|Hgated-author-family|0.373739|0.370229|0.219918|0.311025|144|1597|
|H1-context-prev|0.373661|0.367397|0.218122|0.308125|144|1600|
|H2-GRU|0.373351|0.346641|0.202042|0.288471|144|1610|
|H3-short-attention|0.373428|0.349393|0.203250|0.291013|144|1606|
|H4-causal-mixer|0.383733|0.421928|0.219822|0.353513|0|1624|

|模型|历史1词KL|历史2词KL|历史3词KL|
|---|---:|---:|---:|
|H0-static|0.168524|0.311641|0.629904|
|Hctx-old-context-only|0.169081|0.285435|0.597951|
|Hgated-author-family|0.168553|0.312005|0.630129|
|H1-context-prev|0.167805|0.308355|0.626032|
|H2-GRU|0.164317|0.282004|0.593603|
|H3-short-attention|0.165586|0.284965|0.597628|
|H4-causal-mixer|0.209399|0.367447|0.688939|

H2 的本次末点KL最低；Hctx也已有明显改善，不能把上下文候选的全部收益归因于新token/多词记忆。
H4 末点KL比初始更差，虽TV下降，仍按预先规定的KL选best_step=0；best.pt是未训练初始权重，真正训练后的权重在last.pt。负结果完整保留，没有加轮次、改学习率或重抽种子救结果。
单seed、单epoch、单超参数pilot，未作显著性结论；H4退化不证明架构无效，H2名次也不证明更长记忆必要。历史1词相当K2，不能验证多词记忆。
其他order/span/来源分桶、初始/最终逐桶KL/TV和源题组等权指标完整保存在artifacts/training_summary.json。
T=.2/top-p=.9实际生成质量、完整结构候选调度及公平端到端速度尚未评测；没有在本批自动启动它们。

14个best/last checkpoint已核验模型/计划/步数、FP32形状与有限参数/Adam状态，清单与hash见artifacts/checkpoint_inventory.json。
大权重、optimizer、memmap和train.log只保留在外部cache。checkpoint路径相对此次run根目录：/home/shx/.cache/dllm_infilling/context_heads_training_20261028_v1。
训练/恢复命令见README.zh.md；重新导出小型报告：`CUDA_VISIBLE_DEVICES='' /home/shx/miniconda3/envs/dllm_env/bin/python -m experiments.context_heads_v1.training_v1.report`。
旧B与接入补验payload/ready保留；本次新训练结果单独manifest和B_training_done发布。完成后停止，不自动训练第二轮。
