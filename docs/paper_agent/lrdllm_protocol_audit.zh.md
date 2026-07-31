# LR-DLLM Protocol Audit

更新时间：2026-07-31 UTC

- paper：*Improving Variable-Length Generation in Diffusion Language Models via Length Regularization*
- arXiv：`2602.07546v1`
- authors：Jia Li et al.

## Current verdict

截至 2026-07-31，对 arXiv v1、Jia Li 公开 publication page、定向 GitHub 搜索、OpenReview、PMLR 和论文主页的复核均未找到作者发布的 LR-DLLM 实现、仓库 commit 或可核验 license。仓库内也没有作者代码镜像。

因此本项目允许的唯一实现标签为：

**paper-guided, author-unverified reimplementation of LR-DLLM**

不得称为 official、paper-faithful、exact reproduction 或 protocol-matched author implementation。若未来发现作者代码，必须新建 source audit，比较语义后再决定是否废止本地 preregistration；不得静默替换历史 raw。

## Paper-confirmed algorithm boundary

- confidence 是当前 masked positions 上 token distribution 的平均负熵，不是 mean top-1 confidence。
- Stage I probe grid 为 `{1,2,4,8,16,32,64,128}`。
- 对 probe confidence 拟合 `A(L) ≈ alpha + k log L`，并计算 `CL(L)=A(L)-k log L`。
- 从指数 probe 中选择初始剩余长度。
- Stage II 在 `{L-1,L,L+1}` 上反复局部搜索；稳定后提交最左侧 token，更新 prefix，并把剩余长度减一。
- `MAX_LENGTH/MAX_GEN=128`。
- token sampling 使用 `top_p=0.9`、`temperature=0.2`。

论文标题使用 “long-span regime”，而 Algorithm 1 的文字对全部 probe 拟合。主实现按 Algorithm 1 的字面版本执行，除非作者代码或作者说明明确推翻。

## Historical sanity boundary

`analysis_outputs/lrdllm_same_protocol_sanity_20260703_phase2_lrdllm_sanity/` 和 `analysis_outputs/lrdllm_final_attempt_20260708_phase4_v4/` 只记录当时缺少足够算法细节与实现的 blocker；`generation_executed=false`，没有 LR-DLLM outcome。它们不会被改写。

本轮新增的 ambiguity ledger 和 preregistration 位于 `docs/paper_agent/lrdllm_author_unverified_ambiguity_ledger.zh.md`。在该文档 commit/push 之前，不允许执行技术 smoke、机制 smoke或正式结果。
