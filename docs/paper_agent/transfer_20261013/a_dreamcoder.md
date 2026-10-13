# 2026-10-13 A：DreamCoder 来源与冻结迁移审计（追加节点）

任务：保全已完成 DreamOn 诊断、核对 v2 KL 头与普通词表、准备原始 DreamCoder 固定64辅助迁移。仅 CPU 与来源联网检查；GPU 由主控统一调度，A 不启动。研究人口是1033行/164基础题开发诊断集，seed42；无训练、无新封存测试访问。已有 DreamOn 自身结果只作为校准，原始Base单列，不能填补另两份DreamOn权重。

维护动机：隔离来源训练贡献与二次抽样路径贡献；完整词表和原生长度语义是可解释迁移的必要条件。成功标准：既有逐行证据10330条唯一主键、源头严格加载全部参数、普通词元一致、原生参考与适配器在固定16条上相同。技术错误暂停该目标；不按正收益决定是否全量。输出限定本文件、A独占适配/测试文件和 analysis_outputs/zero_training_transfer_20261013/a_dreamcoder/。遵循 karpathy-guidelines；公共接口由主控合入。

## 既有证据保全

只读原诊断 SQLite 一次，导出无损紧凑 gzip JSONL：`dreamon_reused_per_case.jsonl.gz`，包括输出、token ID、评测、seed、前向、相邻/回退/结构/头触发分母。完整10330行，十组各1033、164基础题，缺失/重复/未恢复错误均0；既有16条检查passed。旧报告和原始SQLite不改写、不重评测。新增 `dreamon_reuse_paired.json` 补D−B及其余目标配对；沿用旧clustered_difference，164题整组bootstrap10000次，seed20261013，行加权与题等权同时给出。

原始运行是共享H200，2886.023884秒 GPU作业成本=0.801673小时。该历史计时不能作为正式独占速度比；本次A审计GPU成本0。生成代码a55b1fd78795ae5f83845302a76f31780dda8cd2，用户核定诊断HEAD340170ce6ae44477027dd7e94ec7a9c0f0910384包含收尾修复。

## 来源登记与证据边界

|行|精确模型ID / revision|性质|来源判断|
|---|---|---|---|
|DreamCoder→DreamOn校准|Dream-org/DreamOn-v0-7B / 8ccc74750e43177327f29dab9e91882ba759e194|官方长度微调权重|复用既有完整诊断的A–D，E保留作为历史机制证据|
|DreamCoder Base辅助|Dream-org/Dream-Coder-v0-Base-7B / 2346ccd3be517d0d314152b988a3b9bafa7d6d63|官方原始Base、固定长度|准备独立迁移；不能称DreamOn三起点齐全|

路径均见 `source_registry.json`；本地HF snapshots下4个safetensors分片均存在；分片仅检查存在与size，不重新扫描模型大文件。generation_utils.py、modeling_dream.py、tokenization_dream.py、分词器配置均固定同一行的HF revision。联网读取HF官方模型卡，mirror API返回精确sha与登记一致；mirror只是官方库元数据传输路径。直连HF在本机SSL EOF，未通过关闭证书校验绕过。

初始化证据：官方固定提交 [run_dreamon.sh](https://github.com/DreamLM/DreamOn/blob/8a0a54918412eda9402a327646f7f067f7160ec8/run_dreamon.sh) 明确 `model.partial_pretrain=Dream-org/Dream-Coder-v0-Base-7B`；[官方博客Training](https://hkunlp.github.io/blog/2025/dreamon/) 说明从DreamCoder-7B在OpenCoder educational instruct微调；[官方issue6](https://github.com/DreamLM/DreamOn/issues/6)讨论当前DreamCoder DreamOn权重与另外两起点发布计划。这些支持起点身份，但脚本未固定初始模型revision，博客/模型卡也未披露完整训练运行清单，不能认证发布DreamOn实际初始化为当前缓存2346ccd；该revision只用于明确的原始Base辅助行。目录名或config不作为完整来源证据。

## 头与分词兼容

源头保留registry `kl/full/best`，rank256、lambda1，参数实际为 **77,856,768**。登记checkpoint934907317 bytes，既有sha256为58fd352ef12ff57fa0eec47a94047559f9dfd84b5ba4478dca3395089c87265a；复用摘要，不重hash大文件。CPU mmap读取一次、按原MarkovHead定义构造，`load_state_dict(strict=True)`实测missing/unexpected均空。三个持久键全部加载：embedding.weight、output.weight、output_mask；head非隐藏状态输入，仅前驱token ID查表。零头需deepcopy源头并仅清零output.weight，lambda仍1并保留二次抽样。Base缺expand_token_id，不能以Base config调用make_head重定义头。

两模型config输出152064列，而实际注册token为151668/151667。逐项比较完整token→ID字典，唯一差异是DreamOn新增special `<|expand|>`=151667；排除该来源特有special后所有普通词元完全一致。merges.txt与tokenization_dream.py逐字节相同，代码、缩进、换行、Unicode代表文本编码一致；显式包含expand sentinel的文本编码不同，已如实保留审计。bos为tokenizer151665（config却151643），两模型相同；构画布用tokenizer BOS，EOS/pad151643，mask151666。

源output_mask屏蔽151643、151666、151667，原样保留。151643在源解码是delete、在Base是EOS/pad；151667在Base未注册，不能赋expand动作。未注册输出列不填零、截断、重编号或重映射；Base原生logits保留全部152064列，一旦实际采到未注册列或mask残留，技术错误暂停目标。该检查也覆盖普通词元相同但输出padding列不同含义的风险。

`DreamModel.forward`在modeling_dream.py:809直接返回lm_head logits；原生generation_utils.py:422应用一次`cat([logits[:,:1],logits[:,:-1]])`。新适配器复用已有shift_logits_to_targets恰好一次，不叠加shift。

## 固定64适配与原生参考的明确差异

独占文件 experiments/dreamcoder_fixed_transfer_adapter.py；不改公共解码文件。

- 输入固定 BOS+prefix+64 masks+suffix+EOS，max_length等于这段完整画布，不追加可生成右padding；不读答案长度，不用CAL或oracle。
- 64个最大去噪轮、温度0.2、top_p0.9、top_k=None、bf16模型/float32源头。保留原生EOS/pad penalty=3与eps=1e-3的64步时钟。EOS原位保留；没有expand/delete和结构屏障。所有4组均取中间64 token，用旧原生adapter的skip_special_tokens=True后clean_middle_text，不因头组改变截取。
- 原始native entropy使用int(active*(1−s/t))，不能称每轮严格K1/K2。参考函数在内存编译官方`_sample`局部副本，明确替换为min(K,active)；frontier再仅替换选位。原生模型方法/缓存源码不覆盖，sampler、EOS penalty、logits shift均沿用官方。故16条比较名称应为“显式K调度补丁的原生参考”，不是未修改原生API完全等价。
- global沿用native full-canvas `topk`（上下文为−inf），不能改成压缩mask向量topk；CPU测试已发现并列置信会导致不同位置。frontier用公共 `_left_frontier_ranks`；配对邻接用公共k2_pair_decision。
- C/D共用相邻普通token资格、源头lambda1、fork_rng二次抽样和原derive_row_seed。局部右词seed沿用(row_seed+1000003*step+10007*left_abs+right_abs) mod(2^63−1)。非相邻global保留并行fallback。头触发/机会/特殊输出/回退分母分别记录，固定长度无结构停止所以结构停止分母0，应报告不适用而非隐藏。

相对现有DreamOn必要差异：动态长度换为固定64，无扩删语义，64步原生EOS penalty，原生full-canvas并列选位；现有DreamOn max_gen_len64不改128。该辅助任务的低质量也可能受固定长度误差主导。

CPU exact command：

```bash
cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/transfer-dreamcoder-20261013
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m unittest tests.test_dreamcoder_fixed_transfer_adapter -v
```

2 tests通过，实际导入缓存官方generation_utils，合成前向对两顺序K1 tokenIDs/文本与原生参考逐项一致；零头与非零头执行相同头路径、固定长度、无扩删。该CPU检查不冒充真实16条技术检查。主控runner负责真实检查、计时估算与全量1033×2×4，资源和12小时预算统一累计；A无GPU执行命令、无GPU成本。

## 复用校准配对补充

四组global通过951/901/901/910，frontier942/877/872/890。D−C净+9/+18；C−B净0/−5；D−B净+9/+13。新增D−B的global帮助/伤害15/6，行加权+0.871pp [0.000,1.795]，题等权+1.155pp [0.208,2.277]；frontier24/11，行加权+1.258pp [0.200,2.377]，题等权+1.150pp [−1.096,3.465]。原始D−B不能全部归因训练；以D−C隔离学习贡献。单种子、重复开发人口，不是跨模型迁移成功。完整四组相对K1配对和两种加权区间见新增JSON。

来源与CPU准备完成；真实Base GPU结论由主控追加。任何兼容或技术失败仅能说明当前直迁条件/实现有问题；兼容而无增益说明本源头迁移未成功，不自动否定今后逐模型重训。不启动任何主模型微调或Markov训练。
