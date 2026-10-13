# 2026-10-13 后续重建准备（只准备，不执行）

优先索取每个起点Dream→DreamOn、DiffuCoder→DreamOn的模型ID、完整commit SHA、权重/分词器/自定义生成代码地址、原始Base ID及revision、训练配置、原始数据revision、过滤后人口与分组划分、数据构造种子、训练步数与最终epoch、选checkpoint规则、依赖版本与日志。只列信息，不主动联系作者/发issue。当前缺失项必须写“未找到可验证公开权重”。

两层训练分别是：(1) 各原始Base的DreamOn长度微调；(2) 各微调模型对应Markov头训练。当前迁移结果不能自动启动任何一层，也不能自动否定以后逐模型训练头。

来源：[正式论文§4.1](https://proceedings.iclr.cc/paper_files/paper/2026/file/7c020dd308d7413b5078686723b2f857-Paper-Conference.pdf)，[官方固定提交](https://github.com/DreamLM/DreamOn/tree/8a0a54918412eda9402a327646f7f067f7160ec8)。最贴正式报告的自行重建选择10epochs，产物标“按论文设置重建”，不冒充官方权重。

|项目|正式论文|官方8a0a549代码|重建决策/待验证|
|---|---|---|---|
|数据|OpenCoder educational instruction约110K Python对|prepare_data.py加载OpenCoder-LLM/opc-sft-stage2 educational_instruct，不pin revision|固定数据revision（项目已登记7d28f40d579edd7c24402d17d0c7639f991e6f8d需复核授权使用），分组后划分与基准排除另记，若偏离原官方划分明确说明|
|数据预处理|随机prefix/middle/suffix，instruction与上下文固定|prepare_data用DreamCoder tokenizer过滤prompt+response≤1024，shuffle42，前1000 eval，余下train|复用官方准备逻辑；为各Base固定tokenizer revision；不以测试通过率选epoch|
|切分|正文描述三段随机切分|sft_expand_dataset默认line，提取python code fence，最多重试；合法middle含词数/def过滤；非line可token切分|逐项登记实际保留/过滤人口、分组隔离和种子，不能把Markov prepared_data当同一SFT数据|
|epoch|10|run_dreamon.sh=5（yaml默认3被覆盖）|明确10；不偷偷沿用5并声称论文复现|
|有效batch/context/lr|128/1024/1e-5|run脚本相同，microbatch/GPU8，8卡，trainer拆microbatch并归一loss|单卡保持有效128；microbatch1/2/4/8对应累积128/64/32/16，不能直接把batch降8|
|调度|cosine/10%warmup|trainer get_cosine_schedule_with_warmup，yaml ratio0.1|固定相同；训练步数按实际人口N：10×ceil(N/128)，N≈110K时约8600步，按dataloader drop_last行为复核|
|合并|0.5，static/dynamic inverse各半|merge_prob0.5、use_uniform_merge_prob0.5；dataset构造器裸默认0.4被yaml覆盖|使用配置传入0.5；动态p=0.5×(1-num_masked/input_length)|
|删除个数|0到64|torch.randint(0,min(64,剩余context))上界排他，最多63，可能更少|保持代码构造并说明差异；若改为包含64另标实现变更|
|delete损失|每段删除总损失降权|run脚本weight_eos=true，trainer EOS项平均后与非EOS总和按nonEOS数+1归一|复用代码、验证shift/attention/位置处理|
|推理长度上限|Lmax128且扩展次数上限128|generator max_gen_len控制，项目既有诊断64|本轮64仅开发诊断口径，不能混称论文128复现|

单卡资源预估（理论准备，未实测）：7B全参bf16权重约14GB、梯度约14GB、FP32 Adam一二阶状态约56GB，另有潜在FP32主权重约28GB；基础总量约84–112GB（十进制），尚未计激活、全词表logits、FSDP/通信缓冲及碎片。143771MiB H200约140.4GiB，当前共享约89.6GiB已用，因此当前不能据理论启动训练。gradient checkpointing官方已开；CPU/offload、低精度optimizer或其它节省改法必须另列差异，LoRA不能称完全相同训练。

时间必须在以后明确授权后以该Base、context1024、有效128做有限pilot，测optimizer step耗时与峰值显存，按S=10×ceil(N/128)，T≈S×step_sec+数据/验证/保存开销报告范围。8×H800约5h是论文集群观测，不能按8倍线性宣称单卡40h。当前不运行pilot，不给未经测量的单卡时间承诺。
