# Cross-Model Literature Backbone Rerun Plan

Timestamp: 2026-06-09 11:04 CST

Update 2026-06-11 14:42 CST: DreamCoder Base/Instruct, Dream-7B, DiffuCoder-Base, LLaDA-1.5, and LLaDA-MoE local same-backbone pairs have now completed. LLaDA-MoE candidate is `801/1033 = 77.54%` versus its local `cal_lite` LCAS-v3b baseline `777/1033 = 75.22%`, pairwise `31` wins and `7` losses. This is the strongest current local transfer result, but it is still a local protocol-matched claim rather than external SOTA.

## Purpose

Design a protocol-conscious rerun matrix for all diffusion base models that appear in the relevant code-infilling literature, while keeping claims separated by backbone. This document is meant to be readable from GitHub and sufficient for a future Codex session to continue experiments without rediscovering the state.

## Immediate Local Result To Anchor The Plan

The completed `GSAI-ML/LLaDA-8B-Instruct + midcons` run is not an improvement over its own historical baseline:

| Model | Method / run | Pass | Rate | Avg sec/sample including probe | Interpretation |
|---|---|---:|---:|---:|---|
| `GSAI-ML/LLaDA-8B-Instruct` | historical LCAS-v3 | `817/1033` | `79.09%` | `6.8661` | same-backbone local baseline |
| `GSAI-ML/LLaDA-8B-Instruct` | current `midcons` bounded repair | `815/1033` | `78.90%` | `4.1766` | faster, but `-2` net tasks |

Pairwise: `17` wins, `19` losses, `798` tie-pass, `199` tie-fail. Losses are mainly short: `17` in oracle `<=8` and `2` in `9-12`; wins include `4` long-bucket wins. This is negative transfer evidence.

## Literature Sources And Reported Backbones

Primary sources checked:

- CAL: "Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly", arXiv `2602.00476`, https://arxiv.org/pdf/2602.00476.
- LR-DLLM: "Improving Variable-Length Generation in Diffusion Language Models via Length Regularization", arXiv `2602.07546`, https://arxiv.org/pdf/2602.07546.
- DreamOn: "DreamOn: Diffusion Language Models for Code Infilling Beyond Fixed-size Canvas", OpenReview PDF https://openreview.net/pdf?id=EQTPmqukiU and project link https://github.com/DreamLM/DreamOn.

HF API metadata probes on 2026-06-09 confirmed the following candidate model IDs are public and non-gated: `apple/DiffuCoder-7B-Base`, `Dream-org/Dream-v0-Base-7B`, `GSAI-ML/LLaDA-1.5`, and `inclusionAI/LLaDA-MoE-7B-A1B-Base`.

### CAL Table 2: HumanEval-Infilling Single-Line

| Backbone in paper | Baseline avg fixed-length Pass@1 | CAL avg Pass@1 | Best CAL initial length shown | Avg search steps |
|---|---:|---:|---:|---:|
| LLaDA-Base | `44.8` | `65.5` | `73.6` at `L=8` | `13.6` |
| LLaDA-Instruct | `49.7` | `69.9` | `76.9` at `L=8` | `13.7` |
| DiffuCoder-Base | `46.0` | `68.0` | `74.8` at `L=8` | `14.1` |
| DreamCoder-Base | `49.3` | `70.2` | `76.2` at `L=8` | `14.1` |

### LR-DLLM Table 3: HumanEval-Infilling

| Backbone in paper | Baseline single-line | LR-DLLM single-line | Baseline mean | LR-DLLM mean |
|---|---:|---:|---:|---:|
| LLaDA-8B | `48.3` | `69.4` | `23.4` | `36.7` |
| LLaDA-1.5 | `48.8` | `68.9` | `31.2` | `36.1` |
| LLaDA-MoE | `48.8` | `71.3` | `35.1` | `40.8` |
| Dream-7B | `48.2` | `76.7` | `22.5` | `44.8` |
| DreamCoder-7B | `55.5` | `81.6` | `39.1` | `51.3` |

LR-DLLM Table 9 reports forward-calls/generated-token overhead: Dream `3.6574`, DreamCoder `4.8910`, LLaDA-8B `5.1211`.

### DreamOn Table 1: HumanEval-Infilling Single-Line

| Model in paper | Baseline single-line | With DreamOn single-line | Notes |
|---|---:|---:|---|
| LLaDA-8B | `48.3` | not reported | diffusion baseline only |
| Dream-7B | `48.2` | `88.6` | training-based dynamic canvas |
| DiffuCoder-7B | `53.7` | `92.2` | training-based dynamic canvas |
| DreamCoder-7B | `55.5` | `92.1` | training-based dynamic canvas |
| Deepseek-Coder-6.7B | `73.0` | n/a | AR anchor, not same method family |
| Seed-Coder-8B | `89.7` | n/a | AR anchor |
| Qwen2.5-Coder-7B | `92.6` | n/a | AR anchor |

DreamOn is a training method. It is relevant as an external ceiling and as evidence that dynamic canvas helps, but it is not a clean inference-only baseline unless the comparison explicitly says "training-based method versus inference-time method".

## Local Model Status

| Literature backbone | Candidate HF/local checkpoint | Cached? | Existing local evidence | Correct runner direction |
|---|---|---:|---|---|
| LLaDA-Base / LLaDA-8B | `GSAI-ML/LLaDA-8B-Base` | yes | current best `795/1033 = 76.96%`; A6000 control `787/1033 = 76.19%` | `clean_scripts/run_lcal_official_bounded_repair.py` |
| LLaDA-Instruct | `GSAI-ML/LLaDA-8B-Instruct` | yes | current `midcons` `815/1033 = 78.90%`; old LCAS-v3 `817/1033 = 79.09%` | same LLaDA runner, but thresholds need retuning or disabling mid rescue |
| DreamCoder-Base | `Dream-org/Dream-Coder-v0-Base-7B` | yes | official-canvas cal_lite `825/1033 = 79.86%`; generic LCAS-v3 `0%` invalid/mismatched | `clean_scripts/run_dreamcoder_official_infilling.py`; adapt current policy into this runner |
| DreamCoder-Instruct | `Dream-org/Dream-Coder-v0-Instruct-7B` | yes | official-canvas cal_lite `848/1033 = 82.09%`; generic LCAS-v3 `0.10%` invalid/mismatched | DreamCoder official-canvas runner |
| Dream-7B | `Dream-org/Dream-v0-Base-7B`; optional instruct probe `Dream-org/Dream-v0-Instruct-7B` | yes, local Git/LFS checkout under `/tmp` during run | local baseline `802/1033 = 77.64%`; candidate `803/1033 = 77.73%`; pairwise `28/27/775/203` | Dream/DreamCoder-style fixed-canvas runner works; result is near-tie/slight positive |
| DiffuCoder-Base / DiffuCoder-7B | `apple/DiffuCoder-7B-Base`; optional instruct probe `apple/DiffuCoder-7B-Instruct` if available | yes, local Git/LFS checkout under `/tmp` during run | local baseline `838/1033 = 81.12%`; candidate `839/1033 = 81.22%`; pairwise `25/24/814/170` | Dream-style runner works; result is near-tie/slight positive |
| LLaDA-1.5 | `GSAI-ML/LLaDA-1.5` | yes, local path `/tmp/llada15_probe_20260609` | local baseline `817/1033 = 79.09%`; candidate `818/1033 = 79.19%`; pairwise `18/17/800/198` | LLaDA runner works; result is near-tie/slight positive, not a strong claim |
| LLaDA-MoE | candidate `inclusionAI/LLaDA-MoE-7B-A1B-Base`; optional instruct `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` | yes, local path `/tmp/lladamoe_probe_20260610` | local baseline `777/1033 = 75.22%`; candidate `801/1033 = 77.54%`; pairwise `31/7/770/225` | LLaDA-style runner works with `/tmp/no_flash_attn` and `llmxy` Transformers path; strongest current local positive |

Metadata notes:

- `apple/DiffuCoder-7B-Base` reports `architectures=["DreamModel"]`, `model_type="Dream"`, and special tokens including `<|mask|>`.
- `Dream-org/Dream-v0-Base-7B` reports `architectures=["DreamModel"]`, `model_type="Dream"`, and special tokens including `<|mask|>`.
- `GSAI-ML/LLaDA-1.5` reports `architectures=["LLaDAModelLM"]`, `model_type="llada"`.
- `inclusionAI/LLaDA-MoE-7B-A1B-Base` reports `architectures=["LLaDAMoEModel"]`, `model_type="llada"`.

## Protocol-Matched Experiment Design

### Stage 0: Manifest And Sanity

Create or update a machine-readable model manifest before launching new full runs. Each row must include:

- literature source and table row;
- exact HF checkpoint;
- cached status;
- runner family;
- prompt/canvas format;
- baseline local result path;
- planned output directory;
- expected metric and comparison type.

### Stage 1: Cached Models First

Run only cached models before downloading new backbones:

1. Treat `GSAI-ML/LLaDA-8B-Base` current `midcons` as the positive local anchor.
2. Treat `GSAI-ML/LLaDA-8B-Instruct` current `midcons` as negative transfer evidence.
3. For `Dream-org/Dream-Coder-v0-Base-7B` and `Dream-org/Dream-Coder-v0-Instruct-7B`, do not use the generic LLaDA LCAS runner. First implement/adapt bounded repair inside `run_dreamcoder_official_infilling.py`, then smoke-run `20` or `100` samples before full `1033`.

### Stage 2: DreamCoder Official-Canvas Adapter

Recommended code reuse:

- Reuse DreamCoder canvas construction, shifted logits, and native diffusion generation from `clean_scripts/run_dreamcoder_official_infilling.py`.
- Reuse length-probe scoring and bounded-repair decision logic from `clean_scripts/run_lcal_official_bounded_repair.py`.
- Keep output schema compatible with existing `summary.json` fields: pass rate, selected length metrics, oracle buckets, pairwise baseline, runtime, and repair source histograms.

Minimum acceptance before full run:

- `max_samples=20` smoke exits `0`;
- all rows have verifier output;
- no near-zero prompt/canvas collapse;
- summary reports sensible selected length histograms;
- runtime is recorded including probe time.

### Stage 3: Missing Literature Backbones

After cached runs are documented, resolve and download exact checkpoints using the mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
```

Priority order:

1. `apple/DiffuCoder-7B-Base`, because it appears in both CAL and DreamOn and is a direct diffusion-code backbone.
2. `Dream-org/Dream-v0-Base-7B`, because it appears in LR-DLLM and DreamOn.
3. `GSAI-ML/LLaDA-1.5`, because it appears in LR-DLLM and should reuse LLaDA-8B-Instruct code according to the HF card.
4. `inclusionAI/LLaDA-MoE-7B-A1B-Base`, because it appears in LR-DLLM but may be memory-heavy.

Each missing backbone must pass a model API probe before any full run: tokenizer special tokens, mask token id, generation API, shifted-logit convention if applicable, max memory footprint on GPUs `2,3`, and a `5`-sample verifier sanity check.

### Stage 4: Paper Comparison Table

The final table must be grouped by backbone, not by global score. The `Our previous/local method` column is the project's earlier local method or local control version, not a reproduction of the corresponding paper method.

| Backbone | Literature reported numbers | Our previous/local method | Our current method | Delta | Cost | Claim status |
|---|---|---|---|---|---|---|
| LLaDA-Base | CAL avg `65.5`, best `73.6`; LR-DLLM LLaDA-8B `69.4` | A6000 control `787/1033 = 76.19%` | `midcons` `795/1033 = 76.96%` | `+8` tasks | recorded in A6000 reports | above listed literature anchors; internal positive |
| LLaDA-Instruct | CAL avg `69.9`, best `76.9` | historical LCAS-v3 `817/1033 = 79.09%` | `midcons` `815/1033 = 78.90%` | `-2` tasks | `4.1766s` vs `6.8661s` | above CAL best, but worse than previous local method |
| DreamCoder-Base | CAL avg `70.2`, best `76.2`; LR-DLLM DreamCoder `81.6`; DreamOn `92.1` | official-canvas cal_lite `825/1033 = 79.86%` | bounded repair `832/1033 = 80.54%` | `+7` tasks | `3.7763s` vs `3.7847s` | above CAL, below LR-DLLM and DreamOn |
| DreamCoder-Instruct | no clean exact literature row | official-canvas cal_lite `848/1033 = 82.09%` | bounded repair `834/1033 = 80.74%` | `-14` tasks | `3.8472s` vs `3.8657s` | negative transfer; no direct paper row |
| Dream-7B | LR-DLLM Dream `76.7`; DreamOn Dream `88.6` | cal_lite `802/1033 = 77.64%` | bounded repair `803/1033 = 77.73%` | `+1` task | `3.7337s` vs `3.6494s` | slightly above LR-DLLM, below DreamOn |
| DiffuCoder-Base | CAL avg `68.0`, best `74.8`; DreamOn DiffuCoder `92.2` | cal_lite `838/1033 = 81.12%` | bounded repair `839/1033 = 81.22%` | `+1` task | `3.7538s` vs `3.6562s` | above CAL, below DreamOn |
| LLaDA-1.5 | LR-DLLM LLaDA-1.5 `68.9` | cal_lite `817/1033 = 79.09%` | bounded repair `818/1033 = 79.19%` | `+1` task | `6.6453s` vs `5.4224s` | above LR-DLLM, near-tie locally |
| LLaDA-MoE | LR-DLLM LLaDA-MoE `71.3` | cal_lite `777/1033 = 75.22%` | bounded repair `801/1033 = 77.54%` | `+24` tasks | `10.6107s` vs `8.7025s` | above LR-DLLM; strongest current local gain |

## Commands To Reuse

LLaDA-like current policy:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_lcal_official_bounded_repair.py \
  --model-path GSAI-ML/LLaDA-8B-Base \
  --output-dir /home/shx/projects/dllm_infilling/outputs_clean \
  --experiment-name full_lcal_official_bounded_repair_union_midcons_<model_tag>_gpus23
```

DreamCoder existing official-canvas baseline runner:

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=2,3 TOKENIZERS_PARALLELISM=false /home/shx/miniconda3/envs/dllm_env/bin/python clean_scripts/run_dreamcoder_official_infilling.py \
  --model-path Dream-org/Dream-Coder-v0-Base-7B \
  --mask-length-source cal_lite \
  --length-alpha 0.10 \
  --output-dir /home/shx/projects/dllm_infilling/model_generalization_runs/20260609_dreamcoder_official_rerun \
  --experiment-name smoke_dreamcoder_base_official_canvas
```

## Next Codex Prompt

Use this prompt to start the next experiment session:

```text
请严格遵循 AGENTS.md 与 docs/paper_agent/research_agent_protocol.md。先阅读 docs/paper_agent/current_action.md、docs/paper_agent/experiment_results.en.md、docs/paper_agent/paper_agent_dashboard.zh.md、docs/paper_agent/experiments/20260609_cross_model_literature_backbone_plan.md，以及最新 activity_ledger。

目标：在已经完成的 literature-backbone local same-backbone matrix 基础上，设计下一阶段能支撑论文主张的实验。不要直接声称 SOTA；先把 local protocol-matched comparisons 和 literature anchors 分开。当前最强 local transfer 是 LLaDA-MoE：801/1033=77.54%，local baseline 777/1033=75.22%，pairwise 31 wins / 7 losses，但耗时更高且 true-long under-selection 仍强。DreamOn 是 training-based external ceiling，不能当作同协议本地 baseline。

下一步优先方向：设计 stronger length signal 或 ablation plan，而不是继续盲目 full run。必须明确 hypothesis、dataset、backbone、baseline、metric、runtime/cost、output/log path、kill criteria、预期能支持的 claim。如果要跑 GPU，默认只用 CUDA_VISIBLE_DEVICES=2,3；HuggingFace 访问使用已配置 proxy 或 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1；先 tiny smoke，再 full run。

每次运行实验前，终端必须打印 experiment design、baseline、model、dataset、metric、output/log path、kill criteria。每次运行后必须更新 docs/paper_agent/current_action.md、experiment_results.*.md、activity_ledger.*.md，并把 summary 结论写到 GitHub 可读文档中，同时在终端打印 local baseline / previous internal / literature anchor 三张对比表。
```

## Reviewer-Level Claim Gate

No result is paper-ready until all of the following are true:

- same dataset split and official verifier are confirmed;
- prompt/canvas format is documented per backbone;
- model checkpoint and decoding budget match the intended comparison;
- pass rate and runtime/cost are both reported;
- local control exists for the same backbone;
- external literature number is cited with table/source;
- pairwise and bucket-level regressions are inspected;
- negative results are preserved rather than hidden.
