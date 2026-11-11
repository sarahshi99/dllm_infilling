# 本地审阅与技术证据

reviewer_gate_disabled；依用户要求只在本会话串行完成。

- H3直接调用已发布H1正式train/accumulated_update/evaluate：唯一共享文件改动是plan的H3初始化描述，损失、数据、精度、顺序、有效批量、优化器和best规则均未改。
- heads.py与B fca5775逐字节一致；H3只选learned-attention。BF16→FP32沿用H2已发布适配器，无重复shift。
- 只读公共Bank检查完整32桶、220000唯一状态、源组隔离、单源单桶≤4、所有分片非空且最后修改早于已发布初始验证。公共库不复制，不读external_test。
- 53项CPU测试全部通过、无skip；包含与A固定版本实际correction/sampling互操作、H3未来/跨链/案例隔离、padding、单历史权重1、归一化、批量/逐词、非零q_proj学习信号、有效批量累积及优化器/RNG恢复。
- 测试命令：CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 CONTEXT_HEADS_A_ADAPTER=/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-context-foundation-20261015-v1/experiments/context_foundation/adapter.py /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m unittest experiments.context_h3_training_v1.test_cpu experiments.context_h1_training_v1.test_cpu tests.context_foundation.test_adapter experiments.context_heads_v1.test_cpu experiments.context_heads_v1.integration_v1.test_cpu -v
- launch.sh通过bash -n；Python编译通过；git diff --check通过。真实公共状态GPU技术检查结果自动写入technical_checks.json，临时权重不进入正式训练。
- 后台只启动一个GPU worker，训练时无DreamOn；训练最多5轮、有限OOM重试，数据未就绪时72h CPU等待；评测OOM保存已训练状态。
- 固定评测复用A的三个公共基线，H2完整结果可追加配对，H1不完整须待补；所有配对使用基础题分组bootstrap。每次生成尝试单独记录，非OOM中断行不会重新生成取优。
- 残余风险：共享GPU负载会改变运行时间；训练和1033行生成的质量尚待后台实际执行。没有Hctx或结构消融，不能作多词或选择性注意力因果结论。未运行全量任务作为前置技术门。
