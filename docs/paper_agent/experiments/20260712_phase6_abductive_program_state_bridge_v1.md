# Phase 6 Abductive Program-State Bridge V1 Experiment Brief

Timestamp: 2026-07-12 UTC

Starting commit: `52f07457a7974fafa69d59914bbc412a2d83e305`.

This is an exploratory/development round. The Phase 5 paper-promotion gate does not block implementation. Phase 5 fixed combined-proxy weights remain unchanged and serve only as a baseline.

The independent V1 method extracts backward obligations from the suffix, computes forward state from prefix plus candidate, and uses a fixed lexicographic ranking over parse/boundary consistency, control-flow contradictions, unsatisfied obligations, def/use conflicts, undefined uses, and restored dependencies. It uses no reference, verifier outcome, pass/fail label, oracle length, fitted score, task/split identity, or frozen-test statistic for selection. It is not fused with homotopy, birth–death, particle assembly, controller, or cal-lite.

The dataset is all `5079` allowed non-frozen rows from `HumanEval-MultiLineInfilling` after excluding `736` rows belonging to the `16` frozen HumanEval groups. The shared pool is `16/32/64/128 × seeds 0/1`, LLaDA-8B-Base, `64` steps, expected `40632` candidates. No existing bank matches this full protocol exactly, so a new bank is required.

The smoke gate is technical only: runner, evaluator/schema, frozen exclusion, resumability/duplicates, and GPU memory. Passing smoke automatically launches full. There is no performance threshold before full.

Comparisons are fixed64, confidence, prefix-only, Phase 5 fixed combined proxy, and V1, all on the same candidate pool. Reporting includes Pass@1, wins/losses, offline reference-length buckets, help/harm, latency, GPU cost, and peak memory. Frozen test remains sealed with `test_evaluation_count=0`; results are not held-out SOTA.
