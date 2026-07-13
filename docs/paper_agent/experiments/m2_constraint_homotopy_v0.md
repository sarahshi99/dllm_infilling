# M2 Constraint-Homotopy V0 Experiment Brief

M2 is an independent candidate method, not a prerequisite, ablation, or replacement for M1; no paper primary method has been selected.

For every allowed RandomSpanLight task, M2 compares two 64-forward fixed-canvas (`64`) LLaDA-8B-Base decodes with seed `0`: `m2_gradual_constraints` increases the influence of visible constraints continuously over denoising, while `m2_abrupt_constraints` applies none in the first 32 forwards and full influence in the final 32. Both retain the standard 64-step mask schedule, same canvas, same seed, same evaluator, and same token-forward budget.

The constraint score can read only prefix, suffix, the current proposed candidate token state, syntax compatibility of `prefix + candidate + suffix`, suffix backward obligations, suffix continuation/control requirements, and inference-visible model confidence. It cannot read reference/canonical solution, test or verifier result, passed label, oracle length, task identity, task group, source row ID, split label, or frozen-test statistic. Evaluator test code is attached only after the method/schedule is fixed and denoising has completed. Frozen controller test remains sealed at `test_evaluation_count=0`.

Raw gradual and abrupt outputs are separate, resumable, and deduplicated. Technical smoke checks exactly 12 stratified cases, method-specific keys, missing/duplicate/error audit, actual 64-forward count, frozen exclusion, resume no-op, and GPU memory. On technical pass, the runner automatically executes all 148 allowed RandomSpanLight task groups without a performance gate. Analysis reports equal-weight task-macro accuracy as primary, descriptive span-micro rate, paired gradual-vs-abrupt wins/losses, forward/token budgets, and wall time.
