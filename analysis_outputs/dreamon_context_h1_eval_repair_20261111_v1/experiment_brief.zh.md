# H1评测补跑与H2解释

用户已授权补H1评测。只读取H1正式best（epoch5），不训练、不改损失/权重/采样/选位；保留原始失败归档。独立worktree从H2已发布9302a1e建立。采用H2已验证的old_hidden.float()适配；可复用原6条成功生成；原1027条BF16/FP32技术失败重新执行一次，绝不重复普通失败挑优。

命令：`bash experiments/context_h1_eval_repair_v1/launch.sh`。同一共享cuda:0，单worker、单DreamOn BF16，FP32头与A采样。现有H3及他人进程不动。模型加载沿用公共load_engine，唯一接口扩展允许本次显存保留4GiB（默认8GiB不变）；不量化不改模型。OOM记录blocked并退出，不无限重试。

独立输出：/home/shx/.cache/dllm_infilling/context_h1_eval_repair_20261111_v1。tmux dreamon_h1_eval_repair_20261111_v1。目标SingleLine/global/K4/1033行/164基础题/seed42/T=.2/top_p=.9/batch1，A官方评分和共同随机输入不变。公共3099条基线直接协议审计复用，配对H2已完成结果。原H1输出和失败记录均保留，评测补充报告单独发布。

成功：1033唯一生成及评分、完整性检查、164题分组配对bootstrap、原始结果和紧凑报告、提交推送。技术失败必须保留准确阶段与恢复命令。无subagent，reviewer_gate_disabled，本地diff review及CPU检查。
