# DreamOn Progressive Infilling V2 Slot Experiments

> Status: protocol version 1 is superseded for decoder evaluation. V2-Hard-v1 is retained as a decoder/protocol failure diagnostic, V2-OpenTail-v1 partial357 is abandoned, and Joint-v1 was not started. The active iteration is only `v2_hard_v2_boundary` under protocol version 2.

Timestamp: 2026-08-01 UTC

## Action

Implement and execute `v2_hard`, `v2_opentail`, and `joint_opentail` over the frozen 642-row exact-three-line development/mechanism population.

## Reviewer motivation

The historical V1 wrapper discards nonempty code generated after the first physical line in 812/1926 rounds. A reviewer needs to know whether its small aggregate gain reflects progressive conditioning or an artifact-prone wrapper. Slot-aware generation removes post-hoc truncation and creates a matched sequential-versus-joint mechanism comparison.

## Hypotheses

- `v2_hard`: future masks visible in the full canvas plus strictly local updates remove V1 truncation failures, but the oracle three-line constraint may limit capacity.
- `v2_opentail`: an open final region may repair hard final-line capacity failures, especially colon-terminated code requiring a body.
- `joint_opentail`: comparison with the identical OpenTail canvas tests whether sequential freezing helps or whether joint denoising and future semantic feedback are preferable.

## Population, model, baseline, and metrics

- Population: existing frozen manifest, 642 rows / 115 base HumanEval problems; development/mechanism population, not held-out test.
- Model: local `Dream-org/DreamOn-v0-7B` snapshot `8ccc74750e43177327f29dab9e91882ba759e194`, bf16, H200, batch 1.
- Historical accuracy controls: one-shot baseline `301/642`; V1 progressive `316/642`. Accuracy/scoring are reused after audit. Historical timing is mixed CPU/GPU and is excluded from pure-H200 latency claims.
- Primary metrics: functional Pass@1, compile, exact match, task-macro Pass@1, paired wins/losses, cluster bootstrap by base problem.
- Compute: actual forwards, real sequence lengths, token-forwards, wall time, per-process CUDA peak allocation.

## Exact staged command

Each method is launched only after implementation and focused verification:

```bash
bash repro_scripts/run_dreamon_progressive_v2_method.sh METHOD
```

with `METHOD` in the fixed order `v2_hard`, `v2_opentail`, `joint_opentail`.

## Environment and outputs

- Python: `/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python`
- GPU: `CUDA_VISIBLE_DEVICES=0`; an unrelated user process currently reserves GPU memory and will not be stopped.
- Outputs: experiment-specific directories under `repro_results/`; smoke, pilot, and full are isolated.
- Logs/tmux: `dreamon-v2-hard`, `dreamon-v2-opentail`, `dreamon-joint-opentail`.

## Success criteria

- Focused unit tests pass.
- Smoke 5 and protocol pilot 30 have zero protocol violations.
- Full has exactly 642 unique rows and one score per row.
- No future/prefix/suffix/separator mutation, PAD selection, forbidden hard-slot newline, unresolved silent acceptance, or post-hoc truncation.

## Kill criteria

- Any scientific protocol change is required.
- Persistent NaN/OOM or state corruption cannot be fixed as an ordinary implementation bug.
- Generation accesses reference/test information.

Pass@1 regression alone is not a kill criterion.

## Known risks

- The population is oracle-selected for exact three-line references, so V2-Hard is only an oracle structural diagnostic.
- A fixed four-mask initialization may still underrepresent difficult spans despite expansion.
- Sequential freezing can make early commitments irreversible.
- Joint denoising may spend updates on future regions before the first hard slot is coherent.

## Expected documentation

Protocol/config/environment records, per-method summaries and diagnostics, joint mechanism analysis, completeness audit, paper-agent dashboard/results/evidence/queue/decision updates, and pushed branch commits.

## Review gate

`reviewer_gate_disabled`: project policy forbids reviewer/subagent dispatch. The fallback is local staged-diff review plus fresh focused tests, `py_compile`, shell syntax validation, protocol-hash verification, and real-tokenizer 642-row static canvas audit before commit and before each full run.
