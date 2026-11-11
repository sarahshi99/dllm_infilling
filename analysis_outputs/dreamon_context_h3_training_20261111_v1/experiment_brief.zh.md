# H3 正式训练执行说明（2026-11-11）

用户已授权本轮必要实现、正式训练、固定评测、提交推送和后台自动接续；仅H3，不使用子智能体。reviewer_gate_disabled，本地diff审阅与fresh verification。

候选实现：复用H1训练器并沿用H2通用适配（采用）；重写独立训练器（不采用，会使比较口径分叉）；读取H1/H2未提交工作（不采用）。
基础：A 43a52fd5fb43392e2ede7b6be659d3261dd6818a；B fca5775e616508c61f19c29689e61e4d59ec1912；H1 9aff4553c3ec955b96704e8f404a0cd785a5a272；H2 9302a1e0027076836de35068ddd2cc1f829cc118。本worktree从已推送H2提交建立。

目的：在同数据、同损失、同初始化输出、同日程下度量上下文＋短记忆注意力头。保持B heads.py不变；256宽/4头/一层，learned-attention，每目标全部当前链已提交历史（K2≤1、K4≤3），不启用Jacobi。缺少同数据Hctx和消融，不能据此证明多词信息或选择性注意力的因果优势；单历史权重1不能证明选择性。

公共数据：H1正式库原地只读，200000/20000状态；10384/1054实际源程序组；32组合各6250/625；280GB不复制。复用coverage/source audit/technical/full validation/5轮训练完成证据。H1评测仅6/1033有效，H2评测1033/1033完整，二者训练均完成。公共采集已关闭，无生产worker在运行。本轮不接触external_test，不重采数据，不改H1/H2工作区或思维导图。

训练：seed20261107，W_out=0且内部正常随机，不继承权重；FP32 head和完整词表T1 KL(fresh||old+delta)，AdamW lr3e-4/betas(.9,.999)/eps1e-8/wd0/clip1；有效批128；固定数据顺序、尾批和最多5轮与H1/H2相同。固定完整验证平均KL选best，同分选早；last保存优化器、数据游标及RNG。技术检查通过即训练，无质量改善门。

评测：best，DreamOn SingleLine/global/K4/1033行，seed42/T=.2/top_p=.9/batch1；复用A共同随机数、FP32采样及官方评分。公共Z4/H4-old/K1直接复用3099条；H1/H2结果完整且协议匹配才配对。164基础题分组bootstrap，失败不重复抽样挑优。

命令：`bash experiments/context_h3_training_v1/launch.sh`。环境 `/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python`，唯一共享cuda:0，单GPU worker。训练不加载DreamOn，预留4GiB；评测只加载一份模型，预留8GiB。OOM保存durable进度并减micro至1，仍失败退出；评测OOM单独标训练完成评测受阻，恢复不重训。CPU公共依赖等待最多72h/60s，明确失败或到期退出。

成功标准：必要技术检查通过，5轮完成，初始及每轮完整验证，best固定评测和评分完整，归档报告及证据索引并退出。阻塞标准：公共数据损坏/实质配置差异/等待到期、micro1仍OOM、存储或模型加载失败。所有GPU时间只作共享环境观测，无公平加速主张。最终产物为独立run日志、progress/exit、checkpoint、验证曲线及分桶、原始生成/逐行评分、配对表、中文报告、精确恢复入口。
