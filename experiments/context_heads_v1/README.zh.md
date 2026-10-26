# B：四个上下文候选与三个强对照，CPU 实现交付

任务批次 20261015，执行日期 2026-10-26。起点 `d39a51d494efc2a1d1661eb28622b18b3b741fa6`
（已包含对 `560e98aa84667fbaad7ee0caae99e76fdca01275` 的补验），分支
`codex/dreamon-context-heads-20261015-v1`。
契约 `dreamon-context-v1-20261015`，schema 1。仅实现与 CPU 验证；没有训练、
GPU 使用、主模型加载、teacher 采集、HumanEval 答案生成或质量/速度实验。

## 范围与路线

已检查本地/远端同项分支、历史报告、用户修改；未发现更晚的 B 实现。
A 的 `context_foundation/` 正在独立 worktree 编写，本分支不修改、不合并其文件。
代码、测试、报告与小型证据全部在本目录；项目日志仅追加一项。
单会话、无子智能体；`reviewer_gate_disabled`，完成本地 diff review 与新的 CPU 验证。

采用统一显式 selected-set 接口与七种结构，执行器独立于头。推荐先比较相同覆盖、
相同数据/初始化/目标的结构收益，再考虑缓存或 GPU 优化。风险是 K2 无法识别多词记忆收益，
以及全词表投影可能支配成本；CPU 张量检查不证明生成质量或加速。
完成标准为因果性、接口、公式、数值边界及普通词 Jacobi 等价通过；任何需要加载主模型、
训练、生成数据或启动 GPU 的动作均超出本批范围。

## Tensor 接口与共同语义

机器可读定义见 [api_contract.json](api_contract.json)，Python 类型见 [contract.py](contract.py)。
`B` 是 batch（可为 1），首批 `K<=4`，`Q` 可为 0，`r=256`。小模块宽度固定 256、
注意力 4 head、GRU 单层、残差系数 lambda=1。`max_slots=8` 仅用于长链边界检查，
不扩大首批真实覆盖。固定 V、H、特殊词集合写入 `HeadConfig`。

| 名称 | 形状 | 含义 |
|---|---|---|
| old_hidden | B,K,H | 整个选中集合的目标对齐旧 H0 |
| position_ids | B,K | X0 原画布坐标 |
| chain_ids | B,K | 普通词链编号；padding 为 -1 |
| selected_mask | B,K | 显式有效槽位 |
| query_indices | B,Q | 要修正目标在 selected set 中的槽位 |
| old_logits | B,Q,V | 对应目标旧分数，仅残差相加/采样/loss 时必需 |
| token_ids | B,K | 已发生记录或 provisional 猜测 |
| token_state | B,K | 0 未知、1 committed、2 provisional |
| history_mask | B,Q,K | 每个 query 可读的有效同链更早槽位 |
| latent | B,Q,256 | 输出 |
| delta_logits | B,Q,V | 可选输出，`project=False` 时不分配 |

mask 为 bool，坐标/槽位/状态为 int64，验证使用 float32。模块不会按 token 值或 padding 值
猜测可见性。读取未来、跨链、未知、结构候选或默认读取 provisional 都报错。不可见
token 可有任意占位值；padding h（甚至 NaN）不参与注意力/摘要。old_hidden 与 old_logits
均视为冻结信息；没有任何完整 logits 的拼接输入。

**位置必须由 A 先对齐。** 仓库 `selected_target_logits` 按目标位置的前一位置读取，
边界处理由 A 的真实 DreamOn 桥核查。B 收到的是已经对齐的 H0，绝不再次 shift，
也不假定 hidden 的同下标是目标预测位置。h_prev 是原来那次完整前向的旧表示，
不是把新词写入后重跑得到的表示。

`forward(batch, project=True, zero_override=False, allow_provisional=False)` 可 teacher force；
`step` 要求 Q=1，吃完整 selected set 与当前实际前缀；`reset(batch, chain_id=None)` 返回
清空后的显式前缀，清全部或一条链，不改变 X0 几何。各实例不保存案例/轮/链状态。
H2 每个 query 从显式前缀重建 GRU 状态；这是 <=3 记录的 CPU 参考实现，不声称已有 GPU cache。

初始覆盖：按原坐标连续性拆链，每链首词由主模型固定；锚点/空 history 输出严格为零。
`zero_override=True` 在所有 bias 后严格清零 latent 与 delta。每轮/每链 GRU 从零开始。
`split_chains` 支持 barrier mask，结构槽位必须从普通 selected set 排除。
`BarrierEvent(chain_id, reason)` 清掉普通记忆；expand/EOS/删除后的新坐标与实际提交由 A 提供。
临时结构候选不作为普通 token 保存。`allow_gaps=False`；跨间隔接口显式预留，首轮关闭。

## 七种公式、输入与归属

设 `J_i={j: history_mask[i,j]}`，`p_i=P h_i`，`d_i` 为同链空间次序；次序不依赖 token。
`f_i=[x_i/1024,d_i/Kmax,phase]`，phase 是固定 round metadata；
`rho_ij=[(x_i-x_j)/1024,(d_i-d_j)/Kmax]`，所以原坐标距离与生成次序有区别。
`MLP` 都是两层线性、中间 SiLU、宽度 256（含 bias）。每种输出最终均为 `W_out z_i`，
`W_out` 为 V×256、无 bias。配置中的特殊词行 delta 被清零；该操作不保护 softmax 概率。
除 Hctx 外，每种都有独立 E 为 V×256。所有以下 z 都受锚点/zero mask 约束。

| 模型 | latent 公式 | 读取输入 |
|---|---|---|
| H0-static | `z_i=E[token_prev]` | 最近可见普通前词；几何/mask |
| Hctx-old-context-only | `z_i=MLP([p_i;f_i])` | 目标旧 h、固定位置/阶段；无 token embedding 模块 |
| Hgated-author-family | `z_i=Eprev * sigmoid(W_g[h_i;Eprev]+b_g)` | 目标旧 h、最近前词；几何/mask |
| H1-context-prev | `z_i=Eprev * (1+tanh(MLP([p_i;p_prev;rho_i,prev])))` | 双位置旧 h、前词、距离/次序 |
| H2-GRU | `a_ij=Linear([E_j;p_j;rho_ij]); s=GRUCell(a_ij,s); z_i=MLP([s;p_i;f_i])` | query 可见前缀按空间次序编码、目标旧 h/位置 |
| H3-short-attention | `a_ij=Linear([E_j;p_j;rho_ij]); z_i=MLP([Attn_i;p_i;f_i])` | query 旧 h，至多三条普通记录及距离 |
| H4-causal-mixer | 下式 | 目标旧 h、固定全选中集合旧 h 摘要、位置、可见此前 token/旧 h |

H0 是仓库旧静态公式的同接口版本。`load_legacy_static` 映射 `embedding.weight`、
`output.weight`、`output_mask` 原键，也接受作者的 `markov_w1.weight`/`markov_w2.weight`。
本地旧 KL full/best 权重实际通过一致性检查，见 artifacts/legacy_compatibility.json；
只验证兼容，不把该 checkpoint 当作后续唯一静态对照，也不用于候选初始化。

Hctx 是“旧上下文/容量/蒸馏校准”强对照，没有性能承诺。它只用 mask 判断是否为锚点，
不会因为可见新词内容或前缀长度从 1 变成 3 而改变同一个非锚点目标的输出。

Hgated 对齐公开 DSpark/DeepSpec 门控家族；作者代码固定在
[005e03b81cec38b7da6399833d609ee89a2587f2](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/markov_head.py)，
MIT 授权完整保留于 references/DeepSpec-LICENSE.txt，源码/授权 hash 见 references/provenance.json。
核验到作者是 V×r embedding、无 bias 的 r→V output，gate 为 **带 bias 的 (H+r)→r**，
concat 顺序是 `[hidden;prev_embedding]`，sigmoid 后逐元素乘。我们的核心公式/尺寸一致，
但包裹了显式 mask、零锚点、特殊词 delta mask、共同初始化和 A 接口；没有使用作者采样器，
因此称 **adapter 版家族对照**，不称整个作者实现原样复现。

H1 的 gamma 最后一层初始化为 0，故 gamma=1 时精确回到同初始化 H0；其余 gamma 参数可学习。
没有叠加 GRU、注意力或主模型重跑；仅 target h 的消融留待有收益后。

H2 的记录编码含目标相对位置，所以参考 step 重新按 query 构建 s；它不是跨 query 的隐式缓存。
`memory_mode=full-memory|reset-every-step|last-only`。后两种此实现结果相同，但保留执行含义；
后续公平比较须分别重训，不能把推理时清空一个 full-memory 训练头当成公平 ablation。
K2 只有一个历史词，不能据此宣称多词记忆有效。

H3 query 为 `Wq p_i`，key/value 从 a_ij 线性映射成 4×64。注意力分数为
`dot(q,k)/sqrt(64) - softplus(b_head)*abs(x_i-x_j)/1024`，显式 causal/padding mask。
`attention_mode=learned-attention|uniform-mean|last-only`；后两种仅供将来各自重训。
空前缀权重全零，单历史各 head 权重严格为 1。因此 K2 不能验证选择性读取，权重图也不是因果解释。

H4 令 `g=masked_mean(old_hidden)`，仅包含固定有效旧 h，不含任何 token 猜测：

```
b_i = F_query([P h_i; P_global g; P_pos f_i])
a_j = F_record([P h_j; P_global g; E[token_j]])
c_i = b_i + sum_{j in J_i} L[d_i,d_j,:] * a_j
z_i = c_i + MLP_256(c_i)
```

L 按通道存储 Kmax×Kmax×256，仅严格更早槽位通过 history_mask 生效；上三角存储参数仍计数，
没有偷读未来 token。全局旧 h 摘要允许包含未来位置的**固定旧信息**。
这是参考 [xPress v1](https://arxiv.org/html/2608.02438v1) 公式 5–9 的轻量因果混合适配，
不是新贡献或论文原样复现：原文先融合每目标旧 h、global mean、移位前词，再用每通道下三角混合，
transform 为 r→2r→r；这里拆 query base/未移位可见记录，加入固定坐标/阶段，使用 r→r→r，
以 query mask 和链次序适配不连续选中集合，没有原文推理算子融合。
原文 co-training/CE/TV/consistency loss 也没有移入本模块。

## Sequential / Jacobi 的准确范围

[execution.py](execution.py) 独立于结构，同一 H4 权重支持两种执行方式；H3 也预留同一接口，
本批只做普通词 CPU 等价检查，不增加训练方案。传入 canonical 全 K query/旧 logits、初始猜测、
每目标固定 `u[B,K]`，及确定性 `sampler(scores,u)->int64[B,Q]`。每条链的首锚点来自初始值且固定。
sequential 读取刚确定的此前词；Jacobi 一次 pass 全组仅读取上一轮的此前猜测，provisional 状态为 2。
Jacobi 的最终结果仍标 provisional，B 没有发布普通生成提交器。

当旧信息/集合/锚点/普通支持/u 固定，采样函数确定且严格因果，长度 L 的链最多 L−1 轮到达同头
sequential。K4 测 1/2/3 轮；非零 H4 翻转前词反例：

| 执行 | 固定锚点+三个目标 |
|---|---|
| initial | 0,0,0,0 |
| sequential | 0,1,0,1 |
| Jacobi 1 | 0,1,1,1 |
| Jacobi 2 | 0,1,0,0 |
| Jacobi 3 | 0,1,0,1 |

还有 H3/H4 稠密随机参数、固定 u 的随机采样检查；重新抽 u 的反例会不相等。
长链 8 的传播反例需要 7 轮，3 轮不够。链较短按实际长度/屏障处理；没有双向猜测。
该结果不是 DreamOn fresh 等价，也不保证答案正确。

**第二批完整生成前**，A 的共享 adapter 必须单独验证 provisional expand/EOS 等候选的调度，
以及最终结构屏障之前的前缀和 sequential 相等。B 的普通词执行器拒绝结构输入/输出，
不能替代这项检查；未过时仅能发布本批普通词 CPU 等价。A 本批真实生成不启用新 Jacobi。

## 初始化与后续训练接口

默认 seed=20261015；E 与 W_out 都 normal(0,.02)，相同形状的 token 头从相同随机值开始，
Hctx 无 E 但 W_out 初始张量相同。额外参数是各结构按同一 seed 独立构造的默认线性/GRU 初始化，
H1 gamma 末层清零，H4 L normal(.1,.02)，H3 distance_raw=0。构造不改变调用方 CPU RNG。
这与旧静态头 E normal(0,1)、W_out=0 的历史初始化不同；没有候选继承训练过的静态头。
后续若采用共同旧表初始化，必须所有适用结构一起采用并记录。当前非零 W_out 用于排除平凡等价测试。

[loss.py](loss.py) 只提供 `teacher_kl(old_logits,delta_logits,teacher_logits,query_mask)`：
T=1，full-vocab `KL(teacher||student)`，student=softmax(old_logits+delta)，teacher/旧 logits detach。
teacher 定义为冻结 DreamOn 在 **X0+当前目标所在链可见已发生前缀** 的分布，其他链本轮新词不写入。
teacher 状态库仅 committed 合法前缀；provisional 不得冒充已提交。跨间隔整轮前缀另定义协议。
所有七种头使用同一数据、目标与训练预算；没有结构独享 CE、teacher、rollout 或默认 DCD。
T=.2/top-p=.9 的实际指标另算，本批没有生成/评测。

特殊词 delta=0 后 `exp(old_special)/(sum exp(old+delta))` 的分母仍会变化，所以特殊词概率会变化。
普通支持限制只用于本批等价 mock，不能外推成实际结构概率保护或结构调度规则。

## 真实词表、参数、MAC 与激活

本地 tokenizer/config 实测固定快照 `8ccc74750e43177327f29dab9e91882ba759e194`：
Qwen2TokenizerFast，base vocab=151643，含新增词 `len=151668`，最大 token id=151667；
**模型 logit V=152064（含 396 个补齐行）**，H=3584。
两种 V 的完整 meta 表分别保存于 artifacts/accounting.json 与 accounting_tokenizer_support.json。
默认残差维数匹配 A 保留的完整旧 logits；若未来 A 显式统一截取 tokenizer 支持，所有头、
teacher、student 需一起使用 V=151668，不能在残差相加时悄悄截取。

新共同配置特殊词为 tokenizer 核验的 `[151643,151665,151666,151667]`。
旧 checkpoint 只屏蔽 `[151643,151666,151667]`，因此兼容性检查显式用旧 mask，
不把 BOS 151665 的 mask 差异伪装成权重不一致。普通 mock 也排除 tokenizer 范围外补齐行。

下表为模型全支持 V=152064、H=3584、r=256。MAC 为单次 B=1,K=4,Q=3 dense forward，
1 MAC=一次乘累加；不计 bias add、非线性、mask/index、逐元素 gate/distance、反向与分配器成本。
H2 参考实现即使不可见记录也计算 K 次 GRU 后用 mask 丢弃，已按实际路径计入。

| 模型 | E+W_out 参数 | 额外参数 | 总参数 | 核心 MAC | 全词表投影 MAC |
|---|---:|---:|---:|---:|---:|
| H0 | 77,856,768 | 0 | 77,856,768 | 0 | 116,785,152 |
| Hctx | 38,928,384 | 1,050,112 | 39,978,496 | 3,148,032 | 116,785,152 |
| Hgated | 77,856,768 | 983,296 | 78,840,064 | 2,949,120 | 116,785,152 |
| H1 | 77,856,768 | 1,115,392 | 78,972,160 | 6,096,384 | 116,785,152 |
| H2 | 77,856,768 | 1,642,240 | 79,499,008 | 13,312,256 | 116,785,152 |
| H3 | 77,856,768 | 1,444,868 | 79,301,636 | 10,369,280 | 116,785,152 |
| H4 | 77,856,768 | 2,365,952 | 80,222,720 | 9,114,880 | 116,785,152 |

按 tokenizer 实际 V=151668，两张表合计 **77,654,016** 参数；Hctx 一张 38,827,008。
额外参数不变。模型全支持两张表 fp32 共 311,427,072 bytes=297 MiB；训练还要加梯度/
优化器、冻结主模型及其激活。output_mask buffer 另计 608,256 bytes。不是“小头”即可忽略大表。

| K4 执行（B=1，排除固定锚点） | 投影调用 | 每调用目标数 | 投影总位置 | 投影 MAC | 采样词表元素数 |
|---|---:|---|---:|---:|---:|
| sequential | 3 | 1,1,1 | 3 | 116,785,152 | 456,192 |
| Jacobi 1 | 1 | 3 | 3 | 116,785,152 | 456,192 |
| Jacobi 2 | 2 | 3,3 | 6 | 233,570,304 | 912,384 |
| Jacobi 3 | 3 | 3,3,3 | 9 | 350,355,456 | 1,368,576 |

这是可关闭跨链/间隔覆盖之外同一组三目标的开销；H3/H4 同样计数。
没有重新投影锚点；直接对含锚点 Q=4 调用 forward 则四行都会投影，不能按三行记账。
当前 CPU 执行器按 batch 行逐个执行，每条链有自己的锚点。
mock 每位置 softmax+CDF 为 O(V)，A 的 top-p 若排序另有 O(V log V) 工作；没有 GPU ms 数据。
旧 h 投影/摘要在参考 forward 中会重新计算，未声称已有 cache。JSON 另列实际 sequential/
Jacobi3 全部 core MAC，不能把一次全组 forward 的 core 当成三轮总成本。

激活按逻辑张量分别记账，**不是测量 peak，也不是完整 autograd 内存上界**：

| 张量 | 元素数（B=1,K=4,Q=3） | fp32 bytes |
|---|---:|---:|
| old_hidden 输入 | 14,336 | 57,344 |
| old_logits / delta / student / mock probs / mock CDF 各一张 | 456,192 | 1,824,768 |
| rank latent | 768 | 3,072 |
| H2/H3 record concat | 6,168 | 24,672 |
| H2/H3 records；H4 mixer coefficients 各一张 | 3,072 | 12,288 |
| H2 逐 unroll states / 三 gates | 3,072 / 9,216 | 12,288 / 36,864 |
| H3 keys / values 各一张 | 3,072 | 12,288 |
| H3 scores / weights 各一张 | 48 | 192 |
| H4 records | 1,024 | 4,096 |

完整 shapes/bytes 清单及逐参数组见 accounting.json；线性/SiLU/GRU autograd 会保存额外中间量，
FP32 KL 的 teacher probabilities/logprobs 等全 V 张量也另占内存。不能把 rank 激活表当总内存。
Jacobi3 的全词表投影/采样工作量是 sequential 的三倍；低维并行不意味着端到端更快。

## CPU 证据、复现与 A 适配

使用 `/home/shx/miniconda3/envs/dllm_env/bin/python`，Python 3.10.20、torch 2.5.1+cu121、
transformers 4.38.2；虽然 torch 包含 CUDA 构建，所有张量/操作仅 CPU（真实尺寸仅 meta）。
根目录系统 Python 的 torch 缺少 libcudnn，故明确使用该环境；没有为此启动 GPU。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/shx/miniconda3/envs/dllm_env/bin/python -m unittest experiments.context_heads_v1.test_cpu -v
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/shx/miniconda3/envs/dllm_env/bin/python -m experiments.context_heads_v1.verify_cpu
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/shx/miniconda3/envs/dllm_env/bin/python -m experiments.context_heads_v1.adapter_example
```

verify_cpu 仅读取本地 tokenizer 元数据和本地旧头（mmap），不会读取主模型 tensor 或数据集；
无本地文件时记录缺项，不下载。此次两者均存在。单元测试覆盖：各结构 shapes/有限梯度、
冻结输入、zero/bias/锚点、旧公式与键映射、gamma=1、作者 adapter 门控公式、未来/不可见词扰动、
padding NaN/无效 token、chain/round reset、Hctx 无新词输入、H3 单记录权重、GRU/attention 开关、
稠密非零参数的同头 Jacobi3 等价、一二轮反例、重抽 u 反例、长/短/空链、空间乱序、屏障、
provisional、tokenizer 补齐行支持和 mock adapter。

23 项 CPU 单测通过（含各模型/模式的 subtests），实际 checkpoint 兼容及 adapter mock 通过；
完整结果为 artifacts/tests.log、verification.json。CPU float32 同形状公式逐位一致、零 mask/锚点
精确为零，采样 token 精确相等；step-vs-batch 使用 atol=1e-6,rtol=1e-5，attention 和为 1 的容差
atol=rtol=1e-6。实际旧 checkpoint 同形状差为 0；不同 GEMM 行数最大差 4.768e-7，在上述容差内。
不承诺近 tie、不同 kernel/硬件的 token 仍相同；理论等价前提是同一个确定函数。

[adapter_example.py](adapter_example.py) 是可运行的小尺寸范例：

```python
from experiments.context_heads_v1 import ContextHead, HeadConfig
from experiments.context_heads_v1.adapter_example import corrected_scores

# 下面三个值由 A 的已核验 tokenizer / old logits / hidden 冻结配置提供。
head = ContextHead("H1-context-prev", HeadConfig(V, H, tuple(special_ids), tokenizer_size=tokenizer_size))
# aligned_tensors 必须包含整个 selected set 的旧 h/原坐标/链/有效 mask。
# 仅 query 部分的旧 logits 为 [B,Q,V]，history_mask 为 [B,Q,K]。
student_scores = corrected_scores(head, aligned_tensors)
# A 继续调用自己的共享采样器、选位器和结构屏障/提交器。
```

旧 logits 只在 adapter 加法与 loss 中使用。B 不修改 A 的 runner、旧 decoder 或训练脚本。
A 应保留全部 selected-set 元数据，不仅保存目标和已发生词。mock sampler 的 T=1 CDF 不是 A 的
T=.2/top-p=.9 复现；接口相同不代表已验证实际 GPU sampler。

## 发布与未执行事项

ready manifest 为本目录 manifest.json。用户明确要求跨 worktree 交付 contract/hash：它防止 A
把不同 worktree 的同名 manifest 当成这一提交；普通分支名不能标识可变的本地文件内容。
除此之外没有新增研究冻结/gate。`publish_ready.py` 只在分支内容已提交、push 并核验远端 HEAD 后，
检查 manifest 所列内容与当前 commit 一致，向真实 git common dir 的
`research_runs/dreamon_context_20261015_v1/B_done.json` 独占创建该文件，不删除/覆盖其他 ready 或锁。
ready 含 contract/schema/commit/branch/manifest path+hash/created_at。

后续待执行（本批均未启动）：同共同数据新训 H0/Hctx/Hgated/H1–H4 的 T1 teacher KL；
相同覆盖/特殊支持/预算与初始化；独立验证实际 sampler/结构调度；按真实 T=.2/top-p=.9
生成评测；有必要时分别重训 GRU last-only、attention mean/last-only 等消融；收益后再做
H1 target-only、多种子和 GPU 延迟/全词表投影采样核算。没有自动启动 200k 采集、5 epoch、
ablation grid、训练、新 backbone、A 合并或思维导图修改。交付发布后退出。
