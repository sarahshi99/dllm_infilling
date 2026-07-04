# LR-DLLM Protocol Audit

- paper: Improving Variable-Length Generation in Diffusion Language Models via Length Regularization
- arXiv: `https://arxiv.org/abs/2602.07546`
- checked locally: repository search for LR-DLLM / Stage I / Stage II implementation and existing literature notes.
- result: no executable official LR-DLLM implementation is present in this repository.

## Audit Verdict

`protocol_mismatch_blocked`

原因：当前仓库有 LR-DLLM 的 literature anchor 和跨 backbone 结果记录；arXiv HTML 也能核实其 Stage I/Stage II 方法描述。但本仓库没有 official code、Stage I length-regularization 实现、Stage II dynamic span adapter，也没有可确认与本地 prompt/evaluator/checkpoint 对齐的执行入口。因此本轮不能运行 full same-protocol LR-DLLM，也不能把任何本地 heuristic 称为 LR-DLLM。

## Phase 2 Sanity Result

- output：`analysis_outputs/lrdllm_same_protocol_sanity_20260703_phase2_lrdllm_sanity/`
- verdict：`protocol_mismatch_blocked`
- case_count：`10`
- generation_executed：`false`
- full_run_status：`not_run_protocol_mismatch_blocked`

该 sanity 覆盖 short/medium/true-long 与 control pass/fail strata，但只做协议可执行性审计，不运行伪 LR-DLLM baseline。

## Source Boundary

- arXiv：`https://arxiv.org/abs/2602.07546`。页面确认论文题名、作者、2026-02-07 提交日期，并描述 LR-DLLM 是 training-free variable-length framework，包含 length-regularized criterion、probing、bidirectional span adjustment 和 forward-call complexity。
- CatalyzeX：`https://www.catalyzex.com/paper/improving-variable-length-generation-in`。页面确认 paper entry 和 abstract，但未提供可直接在本仓库使用的 official implementation。
- Awesome-DLMs：`https://github.com/VILA-Lab/Awesome-DLMs`。列表中包含 2026-02-07 的 LR-DLLM paper entry，但该条目只显示 arXiv resource；没有本地可执行 code/adapter。

结论：这是 `paper-audited, code/adapter-blocked`，不是 literature anchor 缺失。

## Future Minimal Adapter

1. 定位官方代码或作者发布的伪代码/配置。
2. 将 Stage I 的 length-regularized score 接到本地 `HumanEval-SingleLineInfilling` prompt 和 LLaDA-8B checkpoint。
3. 在同一 verifier、seed、dataset subset 下跑本 sanity manifest 的 10 cases。
4. 只有 sanity 判定为 `protocol_matched_lrdllm` 或明确的 `local_stage1_adaptation` 后，才考虑 1033-case full run。
