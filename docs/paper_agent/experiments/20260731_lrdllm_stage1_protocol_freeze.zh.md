# LR-DLLM Stage I-only 本地适配协议冻结

日期：2026-07-31 UTC

准确标签：**local LR-DLLM Stage-I-only adaptation from arXiv:2602.07546v1 Algorithm 1**。

运行标签：**Stage-I-only selector + fixed-canvas decode**。

本文件在任何 GPU technical smoke、evaluator outcome 和 12-case selected-length 分布出现前冻结。它不覆盖 `docs/paper_agent/current_action.md` 顶部尚未完成的 official CAL SingleLine 动作。

## 1. 基线与开始状态

- base branch：`origin/codex/ccfa-execution-sprint-v1`
- base HEAD：`48661dcc52ee7b2c840f5a2dde9eb9b35f8af078`
- implementation branch：`codex/lrdllm-stage1-adapter-v1`
- clean worktree：`git_workspace/.worktrees/lrdllm-stage1-adapter-v1`
- 初始 `git status --short`：空
- `current_action.md` 顶部动作：`CAL-PHASE1B-SINGLELINE-838`，状态 `adapter_verified_ready_for_commit_then_smoke`；本轮不覆盖该动作。

## 2. 来源审计与实现身份

本轮核对了 arXiv `2602.07546v1` 的论文页、v1 TeX source、Algorithm 1、论文作者信息和定向 GitHub/web 检索。论文 v1 source 未给出作者代码仓库链接；截至 2026-07-31 UTC，没有找到可验证为论文作者正式发布、同时具有明确 repository provenance、commit 和 license 的 LR-DLLM 实现。因此不能使用第三方未核实代码，也不能把本实现称为 official LR-DLLM、LR-DLLM reproduction、full LR-DLLM 或 protocol-matched LR-DLLM。

允许的唯一解释是：从论文公开 Algorithm 1 做出的 **paper-guided Stage-I-only adaptation**。official full LR-DLLM 仍因无 audited author code 而 blocked；Stage II 不实现。

## 3. Algorithm 1 的冻结数学定义

1. 本轮只实现论文 Algorithm 1，即 Stage I 的 instance-specific length regularization 与初始长度选择。
2. 候选长度严格为满足 `2^m <= MAX_LENGTH` 的 2 的幂。`MAX_LENGTH=128` 时为：

   ```text
   1, 2, 4, 8, 16, 32, 64, 128
   ```

   非 2 次幂上限不额外追加，例如 `MAX_LENGTH=100` 只到 `64`。
3. 对候选 `L` 构造 `prefix + L masks + suffix`，每个候选恰好执行一次 model forward，只统计 mask span。
4. 论文置信度是 token distribution 的负熵，不是平均 top-1 probability，也不是 top-2 gap：

   ```text
   conf_i = sum_v p_i(v) log p_i(v)
   A(L) = mean_i conf_i
   ```

   实现使用 float32 的 `log_softmax`，再计算 `sum(p * log p)`；分数通常不大于 0，越接近 0 表示越有信心。
5. 自变量使用自然对数 `log L`。
6. 使用带截距的一元 ordinary least squares：

   ```text
   x_j = log(L_j)
   y_j = A(L_j)
   k_hat = sum((x_j-x_mean)(y_j-y_mean)) / sum((x_j-x_mean)^2)
   intercept = y_mean - k_hat*x_mean
   ```

7. 校正分数与选择规则为：

   ```text
   CL(L) = A(L) - k_hat * log(L)
   selected_length = argmax_L CL(L)
   ```

8. Algorithm 1 伪代码在收集全部指数点后直接执行一次拟合；正文的 “long-span regime” 没有给出阈值或点集。因此 primary 严格使用全部指数 probe：`fit_scope=all_exponential_probes_algorithm1`，不擅自取 upper half。
9. `k_hat` 不 clamp、不取绝对值，不做 clipping、ridge、robust regression、低长度点删除或跨样本全局拟合。
10. 浮点最大值平分时，为工程确定性选择更短长度；这不是论文声明。实现不引入 epsilon tie window，只对实际数值相等的最大分数按更短长度排序。
11. 所有 raw entropy、`log L`、OLS 参数、R² 和 `CL` 必须 finite；任何 NaN/Inf fail-stop。OLS 至少需要两个不同指数长度，因此 selector 要求 `MAX_LENGTH >= 2`。
12. Stage I 选择后长度固定，后续不进行 Stage II、逐 token 提交、局部长度搜索、动态扩缩、rescue、候选 action controller、结果驱动重搜或 evaluator 调用。

## 4. Prefix/suffix 与 probe forward 协议

- model：`GSAI-ML/LLaDA-8B-Base`
- tokenizer：与 official CAL adapter 相同，prefix/suffix 分别使用 `add_special_tokens=False, padding=True, return_tensors="pt"`。
- probe attention mask：`prefix_attention + ones(L) + suffix_attention`。
- mask token：从 tokenizer 的 `mask_token_id` 解析；LLaDA 当前应对应 upstream 默认 mask id，但不硬编码替代 tokenizer 身份检查。
- selector 只读取任务的 `prompt` 和 `suffix`。不得读取 `canonical_solution`、reference completion、oracle length、passed label、task ID 模式、tests 或历史 evaluator outcome。
- `torch.no_grad()`、`model.eval()`；熵至少 float32。
- `MAX_LENGTH=128` 必须精确记录 `probe_forward_count=8`，否则整行 fail-stop。
- token-forward 定义为每次实际 model call 的 `batch_size * input_sequence_length`。当前 batch size 固定为 1，因此 probe token-forwards 是八个 probe input token counts 之和。

## 5. Stage I 后的正式 fixed-canvas decode

正式解码只复用 pinned official CAL 的 LLaDA fixed-length decoder，不把新 arm 加入 `experiments/p1_official_cal_adapter.py::ARMS`，也不修改 pinned upstream source。

证据链：

- pinned CAL repository commit：`741e8418a88a732b4c92812424d4f03cab1f7b1f`
- existing `official_fixed32` adapter config：`initial_gen_length=32, steps=None, block_length=None, span=1, max_gen_length=128, dstep=-1, use_bias=False, temperature=0.0, cfg_scale=0.0`
- pinned `llada_cal.generate` 在 `dstep < 0` 时不进入 CAL length discovery，并返回 `search_steps=0`。
- pinned implementation 随后执行：

  ```text
  curr_gen_length = gen_length
  curr_steps = steps if steps is not None else curr_gen_length
  curr_block_length = block_length if block_length is not None else curr_gen_length
  ```

因此本适配对任意 Stage-I-selected `L` 传入：

```text
steps=None
gen_length=L
block_length=None
temperature=0.0
cfg_scale=0.0
span=1
max_gen_length=128
dstep=-1
use_bias=False
```

这不是凭感觉选择 `steps=L`：adapter 保持与 existing `official_fixed32` 相同的显式调用参数，由 pinned upstream 对 `None` 做 `steps=L, block_length=L` 的解析。`max_gen_length=128` 在 `dstep=-1` 路径中不触发搜索，只为保持 fixed32 参数模板等价。测试必须用 mock argument capture 证明 `L=32` 调用参数逐项等价。

正式解码仍执行 fixed canvas 内 LLaDA 标准 low-confidence denoising schedule；它不改变 canvas 长度，不是 LR-DLLM Stage II，也不是 selector-level remasking/rescue。formal decode 返回的 middle token count必须等于 Stage I 的 `selected_length`，upstream `search_steps` 必须为 0。

与论文完整 LR-DLLM 的差异：论文 Stage II 会动态局部调整 remaining length 并逐 token commit；本轮完全省略 Stage II。论文中用于完整方法采样的 `temperature=0.2, top_p=0.9` 不移植到这个本地 fixed-canvas control；本轮按共同协议要求保持 existing official fixed decoder 的 `temperature=0.0, cfg_scale=0.0`。因此本实验只回答：

> LR-DLLM 的 instance-specific length regularization，单独作为生成前长度选择器时是否可执行。

## 6. 数据、隔离与 outcome-blind 约束

- benchmark：`HumanEval-SingleLineInfilling CAL-Rest non-frozen subset`
- seed：`42`
- evaluator commit：`88062ff9859c875d04db115b698ed4b0f0395170`
- smoke manifest：`analysis_outputs/baseline_manifests_20260731_v1/cal_singleline_rest_nonfrozen_smoke12_manifest.jsonl`
- smoke SHA256：`56559f3f83ba1ce84c9e03622c5ced6e2ed8d2fa08145a1a64dc0cae0885caa1`
- smoke rows：`12`
- full manifest：`analysis_outputs/baseline_manifests_20260731_v1/cal_singleline_rest_nonfrozen_manifest.jsonl`
- full SHA256：`52ef81385984a362fee8729c52cbfa7480265582cabb8250d3ffc27b0aa59af0`
- full rows/clusters：`838/143`；本轮只 preflight，不执行。

只读取既有 non-frozen manifest 和其 outcome-blind summary。不得重新构造 population，不得打开 frozen controller test 样本或 sealed test 文件。运行 dataset loader 只解析 manifest 指定的 source rows，不把完整 1,033-row SingleLine 文件解析为 Python 对象。frozen controller test 保持 `sealed`、`test_evaluation_count=0`。

12-case smoke 只作技术 smoke。完成后可以报告 accuracy，但禁止据此修改 probe grid、entropy、fit scope、tie-break、decoder 或任何参数。

## 7. 成本与 fail-stop gate

每行分别记录：

```text
probe_forwards
formal_decode_forwards
total_forwards
probe_token_forwards
formal_decode_token_forwards
total_token_forwards
wall_sec
peak_memory_bytes
```

必须满足：

```text
total_forwards = probe_forwards + formal_decode_forwards
total_token_forwards = probe_token_forwards + formal_decode_token_forwards
```

`MAX_LENGTH=128` 时 `probe_forwards=8`。任何 ledger 不守恒、selected canvas 长度不一致、fixed decoder `search_steps != 0`、manifest/hash/source revision 不一致或 non-finite 诊断均 fail-stop，并只写 failure journal，不污染 success-only canonical raw。

## 8. 仍不完全由论文明确的事项

1. 正文称 slope 来自 “long-span regime”，但没有阈值；本轮按 Algorithm 1 使用全部指数点。
2. 论文未规定 argmax 平分规则；本地确定性规则为更短长度。
3. Algorithm 1 没有定义 `MAX_LENGTH < 2` 时的退化 OLS；本地实现 fail-stop，而不是发明 `k=0`。
4. 论文完整方法的 Stage II decoder 与本轮 fixed-canvas decoder不是同一算法。本轮 decoder 由项目现有 official CAL fixed32 路径决定，不能用于声称完整 LR-DLLM 性能。
5. 论文没有作者官方代码可供确认实际 long-span 点集、tie 行为、数值精度或 Stage I 与具体 LLaDA fixed decoder 的粘合方式。

以上选择在 smoke outcome 前冻结；任何 sensitivity 必须另建文档、arm 和输出，不能改写本 primary。
