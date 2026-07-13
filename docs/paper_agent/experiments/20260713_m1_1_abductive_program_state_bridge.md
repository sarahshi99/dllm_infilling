# M1.1 Abductive Program-State Bridge Experiment Brief

Timestamp: 2026-07-13 UTC

Starting commit: `db1af6179aeeabf3b97fc9ce2b562af541477573`.

This is an exploratory/development round. The Phase 5 paper-promotion gate does not block implementation. Phase 5 fixed combined-proxy weights remain unchanged and serve only as a baseline.

M1.1 extracts suffix backward obligations, computes prefix/candidate forward facts, forms a contradiction set, and uses dependency-cone targeted remasking with a safe fixed64 fallback. Its fixed lexicographic state score ranks parse/boundary consistency, control-flow contradictions, unsatisfied obligations, def/use conflicts, undefined uses, and restored dependencies. Deployable selection/remasking uses no reference code, verifier/unit-test outcome, passed label, oracle length, fitted score, task ID, split label, or frozen-test statistic. It is not fused with homotopy, birth–death, particle assembly, controller, or CAL-lite.

The dataset is all `5079` allowed non-frozen rows from `HumanEval-MultiLineInfilling` after excluding `736` rows belonging to the `16` frozen HumanEval groups. The shared pool is `16/32/64/128 × seeds 0/1`, LLaDA-8B-Base, `64` steps, expected `40632` candidates. No existing bank matches this full protocol exactly, so a new bank is required.

The smoke gate is technical only: runner, evaluator/schema, frozen exclusion, resume/dedup/missing/duplicate/error accounting, and GPU memory. The completed 8-case historical technical check is retained as implementation evidence; the current route runs the required 12-case MultiLine smoke. Passing technical smoke automatically launches full. There is no performance threshold before full.

Comparisons are fixed64, ordinary-confidence best-of-grid, equal-compute generic remask, M1 score-only, M1 full, and oracle ceiling, all on the same candidate pool. Reporting includes task-macro delta, span-micro descriptive accuracy, wins/losses, help/harm, offline reference-length buckets, forward/token budget, wall time, GPU cost, and peak memory. Frozen test remains sealed with `test_evaluation_count=0`; results are not held-out SOTA.
