# 2026-10-13 B：Dream 起点来源与零训练迁移准备

Dream-7B 经 DreamOn 长度微调的公开权重：**未找到可验证公开权重**。此判断只覆盖本次公开检索，不声称权重不存在。正式 DreamOn 行保持缺失；下列 Dream Base 仅是固定长度辅助迁移，不能补成论文的三份 DreamOn 权重矩阵。

## 来源与范围

- 正式论文：[ICLR 2026 DreamOn](https://proceedings.iclr.cc/paper_files/paper/2026/file/7c020dd308d7413b5078686723b2f857-Paper-Conference.pdf)。官方实现固定 [8a0a54918412eda9402a327646f7f067f7160ec8](https://github.com/DreamLM/DreamOn/tree/8a0a54918412eda9402a327646f7f067f7160ec8)。
- [官方 issue 6](https://github.com/DreamLM/DreamOn/issues/6) 当前仍开放；2026-01-22 回复曾承诺下一周发布另外两份，本次可见回复没有下载地址。未联系作者、未发 issue。
- [Dream-org 模型清单](https://huggingface.co/Dream-org/models) 与 HF API 全站 `search=DreamOn` 本次返回 14 项。可见官方 DreamOn 只有 `Dream-org/DreamOn-v0-7B`；量化与第三方 Java 衍生不算 Dream 起点。原 API 快照保存于本路输出目录。
- 辅助模型精确 ID=`Dream-org/Dream-v0-Base-7B`，revision=`6572adb5535263e4d1a337b56942ba48b6dee2a9`；[官方 model card](https://huggingface.co/Dream-org/Dream-v0-Base-7B) 明确称其为 Dream 7B 的 Base。该 ID 与修订来自官方 API/提交页，未由目录名或 `_name_or_path` 推断训练来源。
- 本地路径：`/home/shx/.cache/dllm_infilling/zero_training_transfer_20261013/dream_base/6572adb5535263e4d1a337b56942ba48b6dee2a9`。这是官方 Base，Apache-2.0。分词器与模型生成代码来自同一精确 HF revision。原 `/tmp/dream_base_repo_probe_20260609` 和标准 HF Base 缓存已不存在，旧 LCAL/长度选择实验不复用为本轮成绩。

## CPU 兼容审计

独立脚本 `experiments/audit_dream_base_transfer.py` 不构造主模型，不加载/改写头，不使用 GPU。`compatibility_audit.json` 是实际运行证据。

- `vocab.json` 的 151643 项普通 token→ID 逐项相等；文件字节因格式不同而不等，因此不能用文件名或字节外观代替语义比较。加入已注册 token 后两者共有 151667 项，ID 差异为 0。
- `merges.txt`、`tokenization_dream.py` 字节完全相等。实际 source tokenizer 是 `Qwen2TokenizerFast`，target 是 `DreamTokenizer`，所以还核对 source slow 规则、实际 fast backend 的正则/NFC、BPE ranks、byte encoder/decoder、errors；全部相等。7 条预定代表文本涵盖代码、缩进、CRLF、中文、重音字符及分解组合、空格差异和特殊词，编码全部相等。
- 两者 tokenizer BOS 都为 151665，config BOS 都为 151643，已有不一致必须保留记录；画布使用 tokenizer BOS。EOS/pad=151643，mask=151666，输出宽度均为 152064。
- source 唯一多注册 `<|expand|>`=151667。Base 没有 expand 动作，此列只是未注册输出列；保留原头完整 `output_mask` 与所有参数，不重新构造屏蔽语义，不填零补映射，不截断/重编号。source 头 strict load 由主控/A 的既有定义审计统一执行。
- Base 中未注册输出列 151667–152063 若实际被采到，立即作为技术错误暂停；不为通过检查重新归一化、改采样或删除这些列。EOS 原位保留，绝不当作删除动作。

此处 `direct_transfer_status=passed` **仅指 CPU tokenizer/输出列兼容条件**，不代表 GPU 16 条检查或质量迁移成功。

## 原生适配与最小差异

[官方 Base `DreamModel.forward`](https://huggingface.co/Dream-org/Dream-v0-Base-7B/blob/main/modeling_dream.py) 只投影 logits；[官方 `_sample`](https://huggingface.co/Dream-org/Dream-v0-Base-7B/blob/main/generation_utils.py) 在调用 logits hook 之前 shift 一次。直接 forward 的适配需 shift 一次；调用 `diffusion_generate` 或其已 shift 的 hook 不再 shift。实际精确版本源文件已保存本地，不依赖上述 main 页面继续变化。

原生策略不是严格每次 K：线性 timesteps 以 `int(num_mask * (1-s/t))` 决定转移个数；也没有 frontier 策略。不能把原生 schedule 与要求的固定 K1/K2 默称完全相同。A 已准备独立 `dreamcoder_fixed_transfer_adapter.py`，主控统一合入；B 复用其官方采样、最小 fixed-K/frontier reference 补丁，不覆盖公共解码文件。固定 canvas64，prefix/suffix 按现有 BOS/EOS 模式构建，temperature0.2/top_p0.9/bf16/batch1。EOS penalty3 是既有 Base 评测请求参数，但当前官方 Base sampler 并未读它，不能宣称其实际生效。

最终截取只取原始64位置中段，用相同 `skip_special_tokens=True` + 既有 `clean_middle_text`。不读答案长度，不用 CAL/LCAL。相对 DreamOn：没有长度扩删；64 是初始及固定画布，而 DreamOn 的64是最大生成长度；论文128上限另属论文口径。

## 冻结人口、技术检查与资源

主控复用诊断 `run_manifest.json.smoke_task_ids` 的16个按稳定ID均匀预选行，和1033行/164基础题人口；seed42、原 `derive_row_seed`。每顺序四组：K1无头、K2无头、K2同类同秩零输出头λ1、K2原v2 KL best头λ1。rank256、前驱token查表；不新增隐藏状态。无头与最小原生reference一致、零头和训练头路径一致、无非法输出列/残mask后，才由主控排队全量。是否全量不依据小样本正收益。

本智能体未运行任何GPU或训练，累计 GPU 作业成本=0；全局12 GPU作业小时由主控统一累计。模型下载不计GPU。当前不提供通过率、帮助/伤害、区间或正式加速比，全部待GPU检查/全量真实证据。

## 命令与恢复

轻量文件已完整下载。网络环境的默认代理访问HF与mirror出现TLS EOF，取消代理直连HF超时；`requests.Session(); session.trust_env=False` 直连 `https://hf-mirror.com` 成功。精确revision的官方API快照含每个shard的LFS大小和摘要。

```bash
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python experiments/audit_dream_base_transfer.py \
  --source /home/shx/.cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B/snapshots/8ccc74750e43177327f29dab9e91882ba759e194 \
  --target /home/shx/.cache/dllm_infilling/zero_training_transfer_20261013/dream_base/6572adb5535263e4d1a337b56942ba48b6dee2a9 \
  --output analysis_outputs/zero_training_transfer_20261013/b_dream/compatibility_audit.json
python scripts/download_dream_base_transfer.py > analysis_outputs/zero_training_transfer_20261013/b_dream/weight_download.log 2>&1
```

首次requests流式下载在Xet重定向后停滞并出现read timeout；本智能体停止自己的下载进程，保留 `.part`，未触及其他进程。下载器已改为curl两分片并行、最多3次重试，每次180秒总时限和连续30秒低于128KiB/s的低速停止，`.part`保留且Range恢复；四个shard总计15231271864字节。完成下载后核对官方LFS大小与摘要，成功才重命名为最终文件。运行状态见 `weight_download.log`、最终 `weight_download_manifest.json`；不在Git提交权重。主控最终registry/运行清单登记GPU检查和实际质量状态。本轮未另行训练；缺失DreamOn权重的自行重建与分别训练头保持未来方案。

## 2026-10-13 最终节点：下载阻塞，B 路停止 GPU 前准备

四个 shard 均已耗尽3次有界尝试，下载器退出1，无后台B下载作业；没有完整可加载的Base权重。另测Git-LFS batch返回备用 `us.aws.cdn.hf-mirror.org` 链接成功，但该CDN的1MiB Range经代理出现TLS EOF，未解除阻塞。`lfs_route_probe.json`保存紧凑错误，省略临时签名下载URL。

`execution_status.json` 与 `weight_download_manifest.json`保存最终状态、每个 `.part` 字节数和恢复命令；部分权重未认证，不能加载评测。初次来源登记中的 `weight_download_status=running...` 是当时快照，以本节点和最终执行状态为准。CPU兼容结论仍为通过，GPU 16行技术检查与1033×8全矩阵均**未运行**；该结果不能解释为已训练头迁移失败，也不触发重训。B累计GPU作业小时=0。解除下载阻塞后，先完整核对LFS摘要，再由主控用A统一适配安排16条技术检查与预算估时。
