# 2026-10-13 三起点来源核查与现有头零训练迁移

当前用户已明确授权主控协调A/B/C三个智能体、共享GPU按资源可用串行执行、无需逐批确认；覆盖仓库默认不使用智能体和默认GPU2/3。实际仅GPU0 H200，不抢占六个外部服务。新主控分支从已核查340170ce创建；原工作区未提交文件和旧报告保持原状。

审稿动机：区分确切DreamOn权重的跨模型迁移、同模型校准、固定长度Base辅助实验，隔离训练贡献和二次采样路径。假设为现有v2 KL rank256/lambda1头可直迁，允许负结果；词表/接口不相容或原生规则不能对齐即报告阻塞，不靠重训/改映射解决。

方案选择：优先复用DreamOn自身10×1033诊断；查找另外两份官方DreamOn，缺失保持主表缺失；兼容的Base单列固定64辅助。拒绝把量化版/第三方Java或HF历史提交当三个起点。无新测试集、无927人口拼接、无训练/K4/隐藏状态头。

人口：原1033稳定ID、164基础题开发人口；seed42及原derive_row_seed；固定16个字典序等距稳定ID技术检查，不按正确率筛选。每目标两顺序、四组A=K1无头/B=K2无头/C=K2同类同秩零输出头lambda1/D=K2来源头lambda1；温度0.2/top_p0.9/bf16/初始64。DreamOn max_gen_len64不改128；Base遵循官方固定长度停止和截取。

输出：docs/paper_agent/transfer_20261013/，analysis_outputs/zero_training_transfer_20261013/{controller,a_dreamcoder,b_dream,c_diffucoder}/。公共代码仅主控合入；各智能体独立worktree/adapter/输出。来源检索和CPU审计并行；所有GPU都由主控持有/tmp/dllm_infilling_h200_diagnosis.lock，全轮累计12 GPU作业小时（含加载、技术检查、评测写盘），默认一次1作业。下载不计；共享计时仅成本，不报告正式速度比。

成功条件：来源精确revision可验证、完整普通token映射及规则一致、来源定义strict加载全部参数、一次shift；16行原生K1与无头adapter对齐、C/D同路径、无非法token/未恢复错误；估时可容纳则收益正负均完成1033×8，配对帮助/伤害与164题cluster CI。停止条件：技术错误、接口不兼容、预算不足、明确资源阻塞。质量低本身不停止，不更参。

运行前在对话显示exact command，运行清单保存命令、环境、版本、PID、日志、输出、预算和恢复入口。最终本地diff review+fresh verification；reviewer_gate_disabled只指不另开reviewer，用户明确要求的A/B/C照常执行。提交只含本轮独占文件并push新分支。

## 2026-10-13 DreamCoder GPU执行动作

CPU实测5项通过：官方原生局部补丁参考两顺序K1/K2一致、C/D相同二次抽样路径、全局预算不重复累计心跳且跨session相加、目标/种子主键去重。头严格加载all keys，来源output_mask不变。max_length为prefix+64+suffix，无右端额外pad；原生日程被显式min(K,remaining)替代，两顺序仅改变选位，不创造新长度控制。EOS罚项3沿用该Base原生默认，两组头λ恒1。

Exact command（主控工作树）：

```bash
nohup bash scripts/run_zero_training_transfer_dreamcoder_20261013.sh > analysis_outputs/zero_training_transfer_20261013/controller/dreamcoder_launch.log 2>&1 &
```

内部命令：`CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python experiments/run_zero_training_transfer.py --target dreamcoder_base --target-registry analysis_outputs/zero_training_transfer_20261013/controller/target_registry.json --output-dir analysis_outputs/zero_training_transfer_20261013/dreamcoder_base`。

共享卡实测仍有>32GiB可用才加载；互斥锁不等待其他作业，资源不足退出并保存阻塞。16条固定ID先校验A/B原生参考一致、C/D路径/非法token；全量估时以16条成本×1033/16，1.5余量与全局剩余预算比较。若技术错误立即停该目标，修复后重做16条。共享耗时只计成本，不生成正式加速比。输出包含原子progress、SQLite成功事务、failure journal、compact gzip、配对/cluster CI、exit及账本。

启动修正：首次nohup未存活（0 GPU成本、未生成manifest），改用 `tmux new-session -d -s zero_transfer_dreamcoder_20261013 'bash scripts/run_zero_training_transfer_dreamcoder_20261013.sh'`；方法、数据、参数不变。


## 2026-10-13 另外两目标的原生差异与下载阻塞

B/C精确generation_utils均没有DreamCoder EOS罚项，只有一处selection marker（DreamCoder两处），native硬编码eps1e-12；固定K补丁后eps只影响已移除的比例调度，EOS罚项0因此不改变分布。专用模块分别experiments.dream_base_zero_training_adapter、experiments.diffucoder_zero_training_adapter，各自绑定penalty0/独立原生参考；entropy为项目既有Base设置、两原生版本均支持，不冒充官方默认maskgit_plus。两目标各两顺序×K1/K2完整CPU参考等价检查通过；C另用高概率EOS检测误罚项。

通用runner后续版本改为按target registry选择独立模块，并记录本目标代码blob及native EOS参数；这是准备未运行目标。当前DreamCoder Python已加载475ea4c版本，全部行仍用该已存版本/manifest参数，无热重载或中途改变。若该运行需恢复，先将runner恢复为475ea4c中的同一blob并focused commit，再用原命令和同一主控目录的全局预算账本；旧源码在Git保留，不新建空预算或覆盖用户修改。

Dream Base保留138539863 bytes .part、0完整shard；DiffuCoder保留300623137 bytes partial、0完整shard。两路官方/镜像resolver、Xet和备用Git-LFS CDN均遭持续TLS/低速/超时，有限重试已停止，仅停止本轮自己的CPU下载，没有触碰GPU服务。两路GPU16/full均未运行，迁移收益未知。恢复命令/精确revision/每shard LFS摘要/片段清单在b_dream和c_diffucoder专属输出目录，不以Base未测成绩填主表。
