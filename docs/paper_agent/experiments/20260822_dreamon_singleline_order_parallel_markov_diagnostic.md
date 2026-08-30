# DreamOn SingleLine order × parallelism × Markov premise diagnostic

Date: 2026-08-22 UTC.

This is a development/mechanism experiment on the full 1,033-row HumanEval-Infilling SingleLine population returned by the released DreamOn evaluation loader. It is not held-out or frozen-test evidence.

Question: when DreamOn is permitted to commit two or four tokens from one fresh forward, does its released global-confidence policy choose a contiguous left prefix or dispersed positions; does a deterministic left-to-right frontier preserve quality while improving real commits per forward; and does committing a predecessor systematically improve the fresh distribution at its right neighbor?

Arms: `C1/C2/C4` use released global-confidence selection over all active masks. `L1/L2/L4` use the leftmost contiguous unresolved-mask prefix. L arms apply a structural-action barrier: commit normal proposals before the earliest expand/delete, execute that action alone, discard later proposals from the old state, and take a fresh forward.

The runner will reuse the released source's model, tokenizer wrapper, prompt construction, logits alignment, entropy confidence, sampling parameters, expansion budget, delete behavior, and functional evaluator. Generation decisions never receive reference code, tests, evaluator verdicts, or oracle diagnostics.

Execution sequence: selection/alignment/action-barrier tests; 12-case smoke across six arms; C1 regression against the unmodified source route; then resumable six-arm full execution on GPU 0. Full reporting will include grouped uncertainty by base HumanEval task, paired comparisons, effective parallelism, top-K spatial summaries, and offline stale-to-fresh Markov premise diagnostics.

## 2026-08-30 authoritative follow-up

The 2026-08-23 valid-v2 six-arm quality and efficiency results remain valid. Its Markov stop judgment is not retained as authoritative because reference-direction and fresh global-rank promotion were not collected; missing fields are not negative evidence. The two-field gap is superseded by the completed v3 supplemental diagnostic in `analysis_outputs/dreamon_markov_premise_rerun_20260830_v3/`. v3 did not rerun C2/C4/L2/L4 and did not train a Markov head.
