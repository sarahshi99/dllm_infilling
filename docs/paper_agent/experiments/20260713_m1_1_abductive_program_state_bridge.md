# M1.1 Abductive Program-State Bridge Experiment Brief

Timestamp: 2026-07-13 UTC

Starting commit: `db1af6179aeeabf3b97fc9ce2b562af541477573`.

This is an exploratory/development round. The Phase 5 paper-promotion gate does not block implementation. Phase 5 fixed combined-proxy weights remain unchanged and serve only as a baseline.

M1.1 extracts suffix backward obligations, computes prefix/candidate forward facts, forms a contradiction set, and uses dependency-cone targeted remasking with a safe fixed64 fallback. Its fixed lexicographic state score ranks parse/boundary consistency, control-flow contradictions, unsatisfied obligations, def/use conflicts, undefined uses, and restored dependencies. Deployable selection/remasking uses no reference code, verifier/unit-test outcome, passed label, oracle length, fitted score, task ID, split label, or frozen-test statistic. It is not fused with homotopy, birth–death, particle assembly, controller, or CAL-lite.

The dataset is all `5079` allowed non-frozen rows from `HumanEval-MultiLineInfilling` after excluding `736` rows belonging to the `16` frozen HumanEval groups. Stage one is the shared `16/32/64/128 × seeds 0/1` pool on LLaDA-8B-Base with `64` steps (`40632` deployable candidates), plus one offline-only oracle ceiling per span (`5079`). Stage two writes two separate, resumable result rows per span: equal-compute generic low-confidence remask and M1 dependency-cone remask. Each real refinement runs a further `64` forward passes from the selected stage-one token state. M1 returns the fixed64 stage-one candidate only when no suffix-derived dependency cone can be mapped; it never substitutes a different generation protocol.

The smoke gate is technical only: runner, evaluator/schema, frozen exclusion, resume/dedup/missing/duplicate/error accounting, and GPU memory. The completed 8-case historical technical check is retained as implementation evidence; the current route runs the required 12-case MultiLine smoke. Passing technical smoke automatically launches full. There is no performance threshold before full.

Comparisons are fixed64, ordinary-confidence best-of-grid, actual equal-compute generic remask, M1 score-only, actual M1 full, and oracle ceiling. Deployable selection sees only candidate ordinal, candidate text, canvas, and seed. Stage-two remasking sees only the selected candidate state and suffix-derived obligations; it does not read references, task IDs, groups, split labels, verifier outcomes, or pass labels. Reporting includes equal-weight base-task macro delta as the primary estimand, span-micro descriptive accuracy, wins/losses, help/harm, offline reference-length buckets, forward/token budget, wall time, GPU cost, and peak memory. Frozen test remains sealed with `test_evaluation_count=0`; results are not held-out SOTA.
