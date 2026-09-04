# Implementation audit

- DreamOn is frozen bf16 inference; Markov softmax/loss is float32.
- Target logits use the released one-position shift by selecting hidden state `target-1` before `lm_head`.
- Markov correction is applied after action masking and before temperature/top-p.
- Structural token correction is exactly zero.
- TV and KL use the same bank, initialization, optimizer, schedule, batch, seed, GPU, and gates.
- TV and KL ran in separate sequential processes.
- No HumanEval outcome or generation was used; HumanEval was decontamination-only.
- No full-vocabulary logits were serialized.
- Reviewer/subagent gate is disabled by repository policy; local diff review and fresh verification are used.
