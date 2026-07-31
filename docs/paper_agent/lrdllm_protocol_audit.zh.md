# LR-DLLM Protocol Audit

更新时间：2026-07-31 UTC

- paper：*Improving Variable-Length Generation in Diffusion Language Models via Length Regularization*
- arXiv：`2602.07546v1`
- authors：Jia Li et al.

## Current verdict

截至 2026-07-31，对 arXiv v1、论文 v1 source、作者公开页面和定向 GitHub/web 搜索的复核仍未找到作者发布且可审计 repository provenance、commit 与 license 的 LR-DLLM 实现。仓库内也没有作者代码镜像。

当前状态必须拆开报告：

- **official full LR-DLLM**：仍 blocked；无 audited author code，不能声称 official reproduction。
- **local Stage-I-only Algorithm-1 adaptation**：LLaDA-8B-Base 的 CPU 实现、测试、fixed decoder 等价审计与 immutable manifest preflight 已完成；准确标签为 **local LR-DLLM Stage-I-only adaptation from arXiv:2602.07546v1 Algorithm 1**。
- **Stage II**：本地这个新 arm 未实现。历史 DreamCoder full-method preregistration/adapter 记录不等于本轮 Stage-I-only arm，也没有被恢复或执行。

新实现不得称为 official LR-DLLM、LR-DLLM reproduction、full LR-DLLM 或 protocol-matched LR-DLLM。若未来发现作者代码，必须新建 source audit，验证作者身份、repository、commit 与 license，再比较语义；不得静默替换历史 raw。

## Paper-confirmed algorithm boundary

- confidence 是当前 masked positions 上 token distribution 的平均负熵，不是 mean top-1 confidence。
- Stage I probe grid 为 `{1,2,4,8,16,32,64,128}`。
- 对全部指数 probe confidence 使用带截距 OLS 拟合 `A(L) ≈ alpha + k log L`，并计算 `CL(L)=A(L)-k log L`。
- 从指数 probe 中选择初始剩余长度；exact tie 的本地确定性规则是更短长度。
- Stage II 在 `{L-1,L,L+1}` 上反复局部搜索；稳定后提交最左侧 token，更新 prefix，并把剩余长度减一。
- `MAX_LENGTH/MAX_GEN=128`。
- token sampling 使用 `top_p=0.9`、`temperature=0.2`。

论文正文使用 “long-span regime”，而 Algorithm 1 对全部 probe 拟合且没有提供 long-span 阈值。本轮 Stage-I-only primary 按 Algorithm 1 使用全部指数点，不擅自取 upper half。

## Local Stage-I-only fixed decode boundary

本轮独立模块为 `expvision_dllm_clean/lrdllm_stage1.py`，独立 adapter 为 `experiments/p1_lrdllm_stage1_adapter.py`；没有修改 CAL-lite 历史行为，也没有把新 arm 加入 official CAL `ARMS`。

Stage I 选择长度后只运行一次 pinned official CAL LLaDA fixed-canvas decoder：`dstep=-1`、`use_bias=False`、`steps=None`、`block_length=None`、`temperature=0.0`、`cfg_scale=0.0`。pinned upstream 在固定路径内把 `steps` 和 `block_length` 解析为选定 `L`；`L=32` mock argument-capture 与 existing `official_fixed32` 逐参数等价。该 decoder 不再改变 canvas 长度，因此不是 Stage II，也不能外推完整 LR-DLLM 最终性能。

完整冻结见 `docs/paper_agent/experiments/20260731_lrdllm_stage1_protocol_freeze.zh.md`。

## Current execution status

- CPU tests：27 个指定 core/adapter/official-CAL regression tests 全部 `OK`。
- `py_compile`、CLI help、launcher `bash -n`、`git diff --check` 和 CPU preflight 通过。
- smoke/full manifest SHA256 分别为 `56559f3f83ba1ce84c9e03622c5ced6e2ed8d2fa08145a1a64dc0cae0885caa1` 与 `52ef81385984a362fee8729c52cbfa7480265582cabb8250d3ffc27b0aa59af0`；full 只 preflight。
- 2026-07-31 07:25 UTC GPU 0 上已有三个外部计算进程，占用约 105 GiB；ECC 为 0。因此 technical smoke 状态是 `blocked_by_existing_gpu_process`，未抢占、未结束进程，也未启动 838 full。

## Historical sanity boundary

`analysis_outputs/lrdllm_same_protocol_sanity_20260703_phase2_lrdllm_sanity/` 和 `analysis_outputs/lrdllm_final_attempt_20260708_phase4_v4/` 只记录当时缺少足够算法细节与实现的 blocker；`generation_executed=false`，没有 LR-DLLM outcome。它们不会被改写。

本轮新增的 ambiguity ledger 和 preregistration 位于 `docs/paper_agent/lrdllm_author_unverified_ambiguity_ledger.zh.md`。在该文档 commit/push 之前，不允许执行技术 smoke、机制 smoke或正式结果。

本轮 Stage-I-only arm 的独立冻结与实现报告不改写上述历史 full-method preregistration：

- `docs/paper_agent/experiments/20260731_lrdllm_stage1_protocol_freeze.zh.md`
- `docs/paper_agent/experiments/20260731_lrdllm_stage1_implementation_and_smoke.zh.md`
