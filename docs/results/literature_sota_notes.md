# Literature And SOTA Notes

Updated: 2026-05-29

This note records source-checked comparison anchors for future SOTA claims. It is not a SOTA claim for this project yet.

## Primary Infilling Anchor

DreamOn is currently the most relevant direct comparison found for diffusion-model code infilling:

- Paper: "DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas"
- arXiv: https://arxiv.org/abs/2602.01326
- OpenReview PDF: https://openreview.net/pdf?id=EQTPmqukiU
- Venue/status shown by sources: ICLR 2026.
- Benchmarks: HumanEval-Infilling single-line and multi-line pass@1, plus SantaCoder-FIM exact match.
- Evaluation details reported in the paper: official HumanEval-Infilling evaluation scripts; fixed mask length 64 for diffusion baselines; DreamOn dynamically adjusts length.

The paper's Table 1 reports:

| Model | HumanEval-Infilling Single-Line Pass@1 | HumanEval-Infilling Multi-Line Pass@1 | SantaCoder-FIM EM |
|---|---:|---:|---:|
| Deepseek-Coder-6.7B | 73.0 | 45.7 | 76.3 |
| Seed-Coder-8B | 89.7 | 59.3 | 77.2 |
| Qwen2.5-Coder-7B | 92.6 | 58.7 | 79.8 |
| LLaDA-8B | 48.3 | 21.1 | 35.1 |
| Dream-7B | 48.2 | 21.9 | 60.3 |
| Dream-7B + DreamOn | 88.6 | 53.3 | 73.8 |
| DiffuCoder-7B | 53.7 | 45.0 | 58.0 |
| DiffuCoder-7B + DreamOn | 92.2 | 63.1 | 77.4 |
| DreamCoder-7B | 55.5 | 43.2 | 59.3 |
| DreamCoder-7B + DreamOn | 92.1 | 63.8 | 79.0 |

## How To Compare Our Runs

Our current LCAL/LCAS experiments should not be called SOTA until we confirm the exact benchmark split, prompt format, decoding budget, model base, and metric match the literature setting.

Current local runs are best treated as:

- Direct internal comparisons when model, dataset, hardware, runner, and environment match.
- Suggestive literature comparisons when only the benchmark family matches.
- Not comparable when the model family, prompt/canvas format, oracle length availability, or evaluation subset differs.

Required fields for any future claim:

- Source paper or official repository URL.
- Dataset and split.
- Metric and pass@k.
- Model name, parameter count, and checkpoint type.
- Infilling format and whether oracle length, fixed mask length, learned length, or dynamic length control is used.
- Decode steps, temperature/top-p, and max mask/canvas length.
- Whether our result is single-line, multi-line, or mixed.

## Current Local Cross-Model Evidence

The local registry at `docs/results/model_generalization_registry.md` shows prior full Dream-Coder and LLaDA runs. These are useful for transfer planning but not enough for a paper-level SOTA claim because the prompt/canvas format differs across families.

Notable local records:

- `Dream-org/Dream-Coder-v0-Instruct-7B`, official-canvas `alpha010_cap24`: `848/1033 = 82.09%`.
- `Dream-org/Dream-Coder-v0-Base-7B`, official-canvas `alpha010_cap24`: `825/1033 = 79.86%`.
- `GSAI-ML/LLaDA-8B-Instruct`, LCAS v3 resume: `817/1033 = 79.09%`.

These should be re-run under a unified A6000 protocol before being compared to new A6000 LCAL candidates.

## Additional 2026 Anchor: LR-DLLM

LR-DLLM is another relevant variable-length inference paper:

- Paper: "Improving Variable-Length Generation in Diffusion Language Models via Length Regularization"
- arXiv: https://arxiv.org/abs/2602.07546
- Reported setting in the abstract: fully unknown-length HumanEvalInfilling and four-language McEval.
- Reported headline result in the abstract: `51.3%` Pass@1 on HumanEvalInfilling under fully unknown lengths, `+13.4%` over DreamOn in that setting.

This is relevant because it frames length selection as an inference-time confidence-bias problem, close to our project. It is not directly comparable to our current `HumanEval-SingleLineInfilling` local protocol until we match dataset split, prompt/canvas format, and unknown-length assumptions.
