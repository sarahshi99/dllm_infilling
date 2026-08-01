# DreamOn Progressive Infilling V2 Slot Experiments

Date: 2026-08-01 UTC

Branch: `codex/dreamon-progressive-v2-slots`

## Scope

Implement and run exactly three methods over the frozen 642-row development/mechanism population, in this order:

1. `v2_hard`
2. `v2_opentail`
3. `joint_opentail`

No additional method family is authorized. The existing one-shot baseline and V1 progressive accuracy rows are reused after task/model/scorer audit. Their mixed CPU/GPU timing is not treated as a pure-H200 benchmark.

## Assumptions fixed before implementation

- The generation process reads a sanitized population containing only `task_id`, `base_problem_id`, `prompt`, and `suffix`. Reference fields are available only to post-generation scoring.
- The official DreamOn EOS proposal is the selected-position delete sentinel. The official expand token is `151667`.
- A proposal that remains the mask token is an official-compatible no-op: it consumes one forward, leaves the position unresolved, and is counted separately.
- `region_forward_count` means the number of forwards on which a region was eligible and unresolved. `selected_update_count` counts actual top-1 selections from that region.
- Per-forward sequence length is the number of real, attention-visible tokens. `token_forwards` is their sum.
- The model receives the entire real canvas on every forward. Only right padding is excluded from attention and from the model call.
- Smoke and pilot outputs use separate directories and config hashes; they can never be resumed into full outputs.
- A protocol/runtime error produces an explicit failed status and is never silently decoded or scored as a normal completion.

## Task 1: Freeze protocol and sanitized population

Files:

- `repro_results/dreamon_progressive_v2_protocol/protocol.json`
- `repro_results/dreamon_progressive_v2_protocol/environment.json`
- `repro_results/dreamon_progressive_v2_protocol/generation_population.jsonl`
- `repro_scripts/prepare_dreamon_progressive_v2_protocol.py`

Verification:

- 642 unique task IDs, 115 base problems, unchanged order and task set.
- Frozen manifest SHA256 `aab2ea784635e7827c5851bcd5a7ccdc3437fdb4be45f185aa012e504b92f7bc`.
- No reference/evaluation fields in the generation population.

## Task 2: Test-first public slot generator

Files:

- `tests/test_dreamon_slot_generator.py`
- `repro_scripts/dreamon_slot_generator.py`

Red gate:

- Add tests for all 24 required protocol cases before the implementation exists.

Green gate:

- Focused pytest passes.
- `py_compile` passes.
- Source scan finds no first-line truncation helper or post-hoc repair.

## Task 3: Runner, scoring, audit, and comparison

Files:

- `repro_scripts/run_dreamon_progressive_v2.py`
- `repro_scripts/run_dreamon_progressive_v2_method.sh`
- `repro_scripts/analyze_dreamon_progressive_v2.py`
- runner-focused tests in `tests/test_dreamon_progressive_v2_runner.py`

Verification:

- Durable per-row append keyed by `task_id + method + config_hash`.
- Strict config/method/task validation on resume.
- Atomic `progress.json` updates.
- Score schema, completeness audit, and protocol-violation files exist.
- Cluster bootstrap uses at least 10,000 replicates with seed 42.

## Task 4: V2-Hard staged execution

Commands:

```bash
bash repro_scripts/run_dreamon_progressive_v2_method.sh v2_hard
```

Gates:

- Unit tests pass.
- 5-case smoke protocol audit passes.
- 30-case pilot protocol audit passes.
- Full output has exactly 642 unique rows and scores.

Commit boundary: protocol/public implementation first, then V2-Hard full result commit and push.

## Task 5: V2-OpenTail staged execution

Command:

```bash
bash repro_scripts/run_dreamon_progressive_v2_method.sh v2_opentail
```

Same gates as V2-Hard. Commit and push after the 642-row completeness audit.

## Task 6: Joint-OpenTail staged execution

Command:

```bash
bash repro_scripts/run_dreamon_progressive_v2_method.sh joint_opentail
```

Same gates as V2-Hard. Commit and push after the 642-row completeness audit.

## Task 7: Joint analysis and authority-document sync

Outputs:

- `repro_results/dreamon_progressive_v2_comparison_all642/`
- compact CSV/JSON tables and `report.zh.md`
- updates to paper-agent handoff, dashboard, results, evidence, registry/queue, decision log, research design/current action where applicable

Verification:

- Six required paired comparisons.
- McNemar exact, row bootstrap, cluster-by-base-problem bootstrap, and task-macro deltas.
- OpenTail-vs-Hard and sequential-vs-joint mechanism diagnostics.
- V1 truncation error category is absent from V2 outputs.
- Local diff review plus fresh verification; `reviewer_gate_disabled` recorded.

## Stop conditions

Stop and report instead of changing the protocol if any of the following is required:

- changing initial mask count, caps, decoding rule, activation order, or attention semantics;
- using reference information during generation;
- silently accepting unresolved masks;
- mixing rows produced by different protocol/config/runner versions;
- truncating prefix/suffix or post-processing generated strings to repair structure.

Ordinary implementation bugs are fixed within protocol v1, followed by rerunning the affected method from smoke in fresh output directories.
