# B 接入补验：A 的实际关键字接口与 CPU 采样函数

执行日期 2026-10-27；原任务约束继续有效：单会话、无子智能体、仅 CPU，不训练、不跑大模型、
不使用 GPU、不生成 HumanEval 答案、不修改思维导图或 A/C 文件、不自动合并 A。
本补验只追加 `experiments/context_heads_v1/integration_v1/` 和项目日期日志。
原 B 提交 `ec8b159ce276c7a919a924c1b6991ffb64a5b8bf` 的全部 manifest payload、
manifest.json 及公共 B_done.json 均保留；本次独立 manifest/ready 补充原交付。

## 为什么继续这一步

B 首批已经完成。恢复时只读核查发现 A 已发布
`f521bd3f91719c40a1fcfda7c39f977fdd48b5be`（A_done 状态 partial），C 也已发布完成标记。
本轮不补跑 A 的 GPU 剩余项，不借其质量结果扩大研究范围。
具体接入差异是：

1. A 的 `Engine.correction` 用九个公共字段作为 keyword 参数调用头，期望返回包含
   `latent`、`delta_logits` 的字典；B 核心接收 HeadBatch，返回 HeadOutput。
2. A 初始化零对照时访问 `head.head.embedding/output/output_mask`；Hctx 本来就没有 token embedding。
   不能直接拿 Hctx 去冒充旧原始头，也不能因此给它偷偷加 embedding。
3. A schema 特殊集合是 tokenizer 的 4 个特殊词加 396 个词表补齐行，共 400 个。
   B 原型配置只列 tokenizer 特殊词，另用 tokenizer_size 排除 mock 采样补齐行。
   接入应采用 A 的完整固定配置，不用 token 值推断特殊集合。

因此增加最小 keyword/dict 包装器与 schema→HeadConfig 读取函数，并对 A 已提交代码做
CPU 接入测试。成功标准是公式与调用结果一致、无额外可见信息、原 payload 不变；
不是质量、加速或实际结构生成验证。CPU 稠密模型只有小词表，真实尺寸只对缓存做类型/形状检查。

## 接入方式

```python
from experiments.context_heads_v1 import ContextHead
from experiments.context_heads_v1.integration_v1 import KeywordHeadAdapter, config_from_a_schema

# schema 是 A 的 protocol_schema.json；len_tokenizer 是独立核验的 tokenizer 大小。
cfg = config_from_a_schema(schema, tokenizer_size=len_tokenizer)
new_head = ContextHead("Hctx-old-context-only", cfg)
wrapped = KeywordHeadAdapter(new_head, legacy_zero_head=already_loaded_frozen_old_static_head)

# A 已有的实际 correction 调用方式，无需 B 改写它的文件：
result = wrapped(**aligned_common_fields)
delta = result["delta_logits"]
```

`aligned_common_fields` 恰为原契约九个字段：整个 selected set 的目标对齐旧 h、原坐标、链、
有效 mask、query 槽位、query 旧 logits、token ids/state、逐 query 显式 history mask。
包装器不 shift hidden、不补历史、不改 dtype、不从 token 猜 mask，不接受 labels/答案/测试字段。
默认 committed-only；H3/H4 也不会通过该接入包装器启用 provisional/Jacobi。
旧 logits 由 A 在采样边界相加；头不读取完整 logits 作为结构输入。

`legacy_zero_head` 只引用 A 已加载并冻结的**原始旧静态头**，供现有零对照路径使用；
不复制表、不把旧训练权重灌给候选。未提供该参数时，直接 keyword correction 仍可用，
访问 `.head` 会给出明确错误。Hctx 核心始终没有 embedding。
包装器通过 nn.Module 注册该引用，故遍历 wrapper.parameters()/state_dict() 时会同时看到
旧零对照头和新头；这是推理接入器。未来训练仍直接使用 ContextHead/其 parameters 和共同 KL，
不能把 wrapper 的旧零头参数算成候选额外容量或交给候选训练优化器。

`calls` 只累计成功调用的 query 行数（B×Q），与 A 的计数约定相同；是仪表计数，
不是跨案例模型记忆。A 仍负责选位、采样、特殊候选调度、结构提交与锚点跳过。
新包装器 `zero_override=True` 沿用真实 B 计算路径后严格归零，不冒充无头路径。

`config_from_a_schema` 使用原契约/schema、H、V、rank、max_k 和特殊集合，
不改冻结协议。此次 H=3584、V=152064、tokenizer len=151668；特殊集合包含 151668..152063。
不同于扩大普通词支持，补齐行通过配置保持不可进入普通记忆；没有新训练方案。

## 本轮实际检查

- 原有 23 项 + 新增 9 项 CPU 单测，合计 **32 项通过、0 skip、0 failure/error**。
- 七头 keyword→HeadBatch→dict 的形状、公式、梯度、zero 与调用计数一致；provisional 被拒绝。
- B H0 与 A 实际 `FrozenHeadAdapter` 使用同一小词表非零 E/W，输出逐位一致。
- 用 A 实际 `Engine.correction` 方法检查七头接入及其旧零对照路径；仅构造不含模型的空测试对象
  来调用此方法，没有调用 Engine.__init__、forward、fresh 或 decode。
- 直接 CPU 调用 A 的 `sample_key`、`uniform`、`distribution`、`inverse_cdf`，
  T=.2/top-p=.9、stable sort、FP64 CDF，固定每位置 u。H3/H4 三个确定种子及单链/多链通过
  同头 sequential/Jacobi3 **token 精确相等**；非零传播反例仍需第三轮，一二轮不保证相等。
- 同分数 tie 与显式补齐支持 mask 的采样确定性通过。没有把这些调用称为 GPU 验证。
- 按 A 已有 tensor manifest 的固定顺序仅读前 12 个 `states/` 包的九个公共字段，
  核对其已有 hash，检查 shapes/dtypes/同链更早 committed 普通前缀。未读取 x0、teacher arrays、
  答案、测试或 passed 字段；没有采集新状态、调参或建立训练集。
- A adapter、schema 来自上述固定提交，动态 import 的旧 decoder 依赖与 B 基线文件哈希一致。
  源文件没有复制/合并到 B。代码与 schema hash、状态包技术检查结果见 artifacts/。
- 验证前后原 B manifest 的全部 payload hash 及公共 B_done.json 字节保持不变。
  只引用先前完成的真实旧 checkpoint 核验，不重新加载主模型或七份真实 V 的头。

这 12 个状态是**有界技术抽查**，不证明整个 A 状态库无问题；原 h 的真实 DreamOn 对齐依赖
A 已发布桥验证，本次只确认接口不再移动它。A 本身 partial，不因本补验变为 complete。
各结构公式、初始化、rank、成本及既有 ablation 限制均沿用原 B 报告。
CPU 同形状公式、zero、token 比较用精确检查；继承的 float32 step-vs-batch 容差仍为
atol=1e-6、rtol=1e-5。未承诺不同硬件/kernel 或临界 tie 下的 token 等价。

## 复现与证据

在 B worktree 根目录运行，使用原已核验的 CPU 环境，不安装新依赖：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/shx/miniconda3/envs/dllm_env/bin/python -m experiments.context_heads_v1.integration_v1.verify_cpu
```

默认 A 源路径为同级 `dreamon-context-foundation-20261015-v1` worktree，可通过 `--a-root` 指定。
本地已存在且版本相符的 A 源、ready、schema、状态索引和缓存必须可读；不会自动拉取模型或状态。
最多读取 12 个现有状态，可用 `--state-limit 1..12`，输出默认写入本新目录的 artifacts/，
不会改写首批 artifacts。依赖实际版本与 OS、A 固定 commit/hash、抽查统计见 verification.json。

单独复现 9 项新单测：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 CONTEXT_HEADS_A_ADAPTER=/absolute/path/to/A/experiments/context_foundation/adapter.py /home/shx/miniconda3/envs/dllm_env/bin/python -m unittest experiments.context_heads_v1.integration_v1.test_cpu -v
```

原 23 项证据保留，本次完整日志为 artifacts/tests.txt；状态抽查为 state_contract_checks.json。
本地 diff review，无子智能体；已有测试通过后不扩大 CPU 搜索或启动实验 grid。

## 发布与剩余边界

只新增本目录与日期日志的 focused commit，在原 B 分支推送并核验远端 HEAD。
新 manifest.json 只列本补验目录；`B_integration_done.json` 在真实 git common dir 的原公共目录
独占创建，标明 supplements=B_done.json、contract/schema、commit/branch、manifest path/hash 与时间。
原 B_done、A/C ready、锁和原 B payload 不改。该补充 ready 不授权 A 自动合并或开启新执行器。

仍待后续明确任务：GPU 采样/性能、实际生成、临时 expand/EOS 及最终屏障之前的前缀调度，
统一 teacher-KL 训练及质量评测。本轮没有完成这些项目；普通词 CPU 等价不能代替它们。
