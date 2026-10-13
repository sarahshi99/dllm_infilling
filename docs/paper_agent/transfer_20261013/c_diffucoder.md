# 2026-10-13 C：DiffuCoder 来源和零训练迁移准备

本日期节点只追加新证据。没有训练主模型、Markov 头或映射，没有执行 GPU 作业；不替换旧诊断人口或旧结论。公共动态解码文件未修改。使用 karpathy-guidelines 做最小独占适配和本地 diff review。

## 来源结论

DiffuCoder-7B 经 DreamOn 长度微调的确切目标：**未找到可验证公开权重**。核查 [官方固定代码](https://github.com/DreamLM/DreamOn/tree/8a0a54918412eda9402a327646f7f067f7160ec8)、[正式 ICLR 2026 论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/7c020dd308d7413b5078686723b2f857-Paper-Conference.pdf)、[issue 6](https://github.com/DreamLM/DreamOn/issues/6) 和 Dream-org/全站 DreamOn 模型搜索。issue 中 2026-01-22 的发布承诺没有提供下载 ID/revision；当前列举仅识别 Dream-org/DreamOn-v0-7B，不能把其 revision、GGUF 或第三方 Java 衍生版当作 DiffuCoder 起点。检索范围不证明权重绝对不存在。对应主表行应保持缺失。

辅助候选：[Apple 官方 apple/DiffuCoder-7B-Base](https://huggingface.co/apple/DiffuCoder-7B-Base/tree/a2c33054cbe99b9ab7af074f09b1d13943157ff1)，精确 revision `a2c33054cbe99b9ab7af074f09b1d13943157ff1`，同时是此次 API 当前 HEAD。官方模型卡将其定义为 foundational masked diffusion code model，提供原生 Dream-style generation；这些证据证明原始 Base 来源，不证明 DreamOn 微调。

下载/分词器/生成代码统一登记位置：`/home/shx/.cache/dllm_infilling/model_snapshots/apple--DiffuCoder-7B-Base/a2c33054cbe99b9ab7af074f09b1d13943157ff1`。完整 registry 与在线 API 摘要在 `analysis_outputs/zero_training_transfer_20261013/c_diffucoder/`。不能仅凭本目录名认证来源；registry 关联精确官方仓库 revision 和 HF 文件大小/LFS登记。

补充候选审计：第三方 `o-1-o/dreamon-re@191717910d0eb26efe0b09fdfb66f93d48fc2a44` 公开文件只有 README/.gitattributes，没有可下载权重和初始化证据；不计可用目标。Apple Base 的 HF finetune 列表只列出 Instruct 类型，不能据此填 DreamOn 行。原始 API 摘要单独保存。

## 缓存现状与技术审计

历史 `/tmp/diffucoder_base_repo_probe_20260609` 已不存在，HF hub 中也没有 apple 模型权重缓存。旧文档的四 shard 下载成功/1033 行 CAL 成绩仍保留，不能当作本轮固定长度迁移结果。旧 runner `clean_scripts/run_dreamcoder_official_infilling.py` 和持久 CAL 源码 `~/.cache/dllm_infilling/calibrated_adaptive_length-741e8418/diffucoder_cal` 尚在；CAL 的长度搜索、oracle 分支、旧 seed/截取不能混进当前矩阵。

CPU 完整比较（不是仅比较 vocab size）：

| 审计项 | 结果 |
|---|---|
| 普通 token→ID | 151664 项完全相同，包括非 special 的工具/FIM词元 |
| merges / 分词代码 | `merges.txt` / `tokenization_dream.py` 字节完全相同 |
| 代表文本 | Python/空白CRLF/Unicode组合字符/数值运算/工具FIM token/尾空白六组分词完全一致 |
| model 输出维 | 两者均 152064；不截断、不填零、不重编号 |
| tokenizer vocab | DreamOn 151668，Base 151667；唯一新增映射 `<\|expand\|>`→151667 |
| source head 屏蔽列 | 保留 151643 EOS、151666 mask、151667 source expand；Base 151667 未命名，不赋扩展动作 |
| BOS 差异 | 两者 tokenizer BOS=151665，而 config BOS=151643；固定画布使用 tokenizer BOS |
| logits 对齐 | 模型 `forward` 输出原生 logits；官方 `_sample` 第 411 行进行一位 shift，适配器只 shift 一次 |

实际 transformers 4.52.3 加载类为 source `Qwen2TokenizerFast` / target remote `DreamTokenizer`，并非只凭相同 remote 文件认证运行规则。已核查实际 fast backend 正则与 target `pat.pattern` 完全相同、两者 NFC normalization、byte-level BPE、所有共享 added-token 元数据完全相同；source 的 chat template 不参与 infilling。literal `<|expand|>` 的分词差异单独登记为已知 special 差异，不冒充普通 token 相同。

详细比较见 `tokenizer_compatibility.json`。`direct_transfer_status=passed` 的范围明确是 CPU tokenizer/维度/代码兼容；来源头的严格加载由 A/主控统一核对，GPU 16 条技术检查尚未通过。原头必须从**来源 DreamOn 配置**构造并 strict load；用 Base 配置调用 `make_head` 缺少 expand 字段，不能靠新造随机参数回避。

## 固定画布适配差异和最小补丁

独占 `experiments/diffucoder_zero_training_adapter.py` 提供精确 profile、BOS+prefix+64 masks+suffix+EOS 画布以及**未修改官方 `_sample`**的 native reference。不加载 GPU，不选择长度，不实现扩删。参考直接令 `max_length=input_length`，避免旧 runner 追加一个右端 EOS workaround；公共解码接口由主控统一合入。

CPU 原生参考发现不可忽略的不一致：官方浮点时间日程 `int(num_mask*(1-s/t))` 令 canvas64/steps64 的实际每轮转移出现 `1,0,...,2`，canvas64/steps32 出现 `2,2,2,2,1,...,3`，所以不能标注严格 K1/K2。`native_schedule_k1.json`、`native_schedule_k2.json` 为确定性 fake-model 机制证据，不是质量评测。

最小可比补丁建议：将原生每轮转移数改为 `min(requested_k, remaining_masks)`，冻结 global confidence 或 frontier 顺序，其余 sampler/shift/温度/top_p 完全沿用原生；明确登记这项相对原生日程差异。此改动不改变固定长度64或引入长度控制。在主控确认公共实现与独立参考相同前，本文件不声称适配已通过技术检查。

结束、截取与特殊词处理需在主控矩阵统一：保留64画布、不把 EOS 当删除；不得每组使用不同清理规则。独立 native reference 同时返回原始 token IDs、raw 和去 special 的文本，避免隐藏清理行为。不得根据答案长度或小样本收益选择方案。

## 下载、执行清单与阻塞

四 shard 官方登记总大小 15,231,271,528 bytes。模型元数据已恢复；权重正在恢复，未完成前不能载入 GPU。默认环境代理访问 HF 触发 TLS EOF；直连 mirror metadata 可用，直连 Xet CDN 很慢。已发现 mirror resolver 直连、CDN 使用配置代理可返回合法 Range 206，提供最多8并发、每范围最多3次的有限可恢复下载器。初次C自有直接下载已停止并保留 partial，没有终止其他作业。

```bash
python scripts/download_diffucoder_transfer_20261013.py \
  --output-dir analysis_outputs/zero_training_transfer_20261013/c_diffucoder \
  --model-path /home/shx/.cache/dllm_infilling/model_snapshots/apple--DiffuCoder-7B-Base/a2c33054cbe99b9ab7af074f09b1d13943157ff1 \
  --workers 8 --max-wall-sec 1200
```

命令只下载，GPU成本0。初次日志 `analysis_outputs/zero_training_transfer_20261013/c_diffucoder/download.log`，补充范围内恢复的当前日志 `download_resume.log`，最终状态 `model_download.json`；分片持久保存在模型目录 `.download_chunks`，重复命令复用已完整范围和范围内partial。每次墙钟上限1200秒（正在阻塞的网络请求还受30/90秒timeout约束），不无限等待。最终拼接同时流式算sha256并与精确HF LFS登记比较，不额外重扫大文件。签名 CDN URL 不写入证据。下载完成后由主控统一锁调度预固定均匀16条、两顺序/四组技术检查，才能估计 1033×8 全批时间；现阶段不存在可信的全批 GPU 时间预测或质量成绩。当前阻塞是确切 DreamOn 权重缺失、Base 权重恢复与公共适配/GPU检查待完成。

本地验证：两脚本 `py_compile` 成功；tokenizer 全映射 CPU审计完成；官方 `_sample` fake-model日程验证完成。环境：`dllm_env` 加 `PYTHONPATH=/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages` 可导入较新版 transformers；不加时旧 transformers 缺少 `is_torchdynamo_compiling`，需采用已登记兼容环境。共享卡时间不能作正式加速比。

网络补充验证：mirror Git-LFS batch 返回 HTTP200 和 `us.aws.cdn.hf-mirror.org` 下载域名，但该备用 CDN 的小范围 direct 下载 ConnectionError、proxy 下载 SSLError，仍未解除权重恢复阻塞。只登记响应域名和错误类型，不保存签名 URL。

### 2026-10-13 10:43 UTC 有限恢复收尾

权重恢复仍为 `blocked_incomplete_weights`：未得到任何完整 shard，所有已落盘片段保留。正常HF代理TLS EOF、mirror/Xet直连timeout、两段式代理部分字节恢复后持续不增长、Git-LFS备用CDN两种路径失败。第二次恢复会话约六分钟、连续数分钟字节数不变后停止C自己新增的CPU下载 PID843727；没有触碰其他GPU/进程。`model_download.json` 保留每个片段大小、官方LFS/size登记、完整shard计数和恢复命令。GPU成本0；预固定16条技术检查、1033×8矩阵均未运行，收益未知，不能解释为迁移失败。解除条件是完整恢复精确revision权重，再由主控统一调度既定技术检查。

### 2026-10-13 独立固定K参考补齐

主控发现不能把 DreamCoder 原生参考的源码形状用于所有 Base。重新核查本目标精确 HF revision：DiffuCoder `_sample` **没有 EOS penalty**、仅一个 `x_` 选择标记，且硬编码 `eps=1e-12`。DreamCoder 原生实现要求两个标记且默认 EOS penalty3，不能直接硬套。

独占C模块现有两个明确入口：`decode_fixed_transfer` 仅复用 A 的普通固定长度解码计算，并强制本版本原生 `eos_penalty=0`；`native_fixed_reference` 从实际 DiffuCoder `_sample` 独立构造，只替换每轮 `min(K,remaining)` 和 frontier 的位置选择。保留全部原生 sampling/shift/temperature/top_p、原生硬编码eps；没有扩删、CAL或答案长度。旧 `released_native_reference` 保留用于原始浮点日程审计，不能混作矩阵固定K参考。

使用精确缓存 `generation_utils.py` 与包含明显 EOS 概率、依赖位置/内容的 CPU fake model 校验：global/frontier × K1/K2 四组，完整64 token IDs及清理后文本与独立原生参考完全相等，forward分别64/32；零/非零头共用同一适配入口，调用计数与头触发计数一致；添加 penalty3 被显式拒绝。3项 unittest 通过（0.777秒），详细四组 token证据和命令见 `fixed_adapter_cpu_checks.json`。这是接口机制证据，**不是**1033开发人口的GPU质量结果；GPU16仍未运行、下载阻塞保持，CPU/GPU成本没有混算。没有恢复网络下载或GPU。

## 仅准备的后续重建信息

向作者索取：DiffuCoder→DreamOn 精确 HF ID/revision/checksums、初始化 Base revision、tokenizer/生成代码 revision、训练与数据处理 commit、OpenCoder 数据 revision/筛选划分、epochs与有效batch/优化器日程、合并/删除构造参数及官方评测配置。此处没有主动联系作者。

若需自行重建，先做独立研究决定：固定 Base/data/group split，采用官方数据准备和构造；正式论文110K educational_instruct、10epochs、有效batch128、context1024、lr1e-5、cosine/10%warmup、merge0.5、静态动态各半，选择10epochs并登记官方 `run_dreamon.sh` 5epochs等差异。先测显存/时间，单卡以梯度累积维持batch；长度微调与随后各自Markov头训练分别登记，不使用测试通过率选择轮次。不在本轮自动启动任一训练。
