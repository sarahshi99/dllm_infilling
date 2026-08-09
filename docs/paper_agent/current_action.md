# Current Paper-Agent Action

Timestamp: 2026-08-09 UTC

Research decision: `execute_frontier_gated_dreamon_v0`.

Action status: implement and execute an independent `Frontier-Gated DreamOn V0` route. The historical DreamOn V1/V2/V3, Hard3, OpenTail, Nonempty, compile-gate, slot, line-wise, BoundaryShift, and V3 expansion-budget routes are frozen/superseded experiments and remain immutable negative/mixed evidence.

Branch: `codex/frontier-gated-dreamon-64`

Starting HEAD: `3eac683fcfdb2d8936e3d88416848b553ed4f69a`

Method scope:

- State is one continuous `prefix + dynamic_middle + suffix` canvas.
- `dynamic_middle` starts with exactly 64 mask tokens.
- The only method change is position eligibility: unresolved masks from the leftmost frontier through width `w` may be ranked/committed.
- Compare `w in {1,4,8,16,infinity}` with otherwise unchanged verified native DreamOn decoding.
- Newline is an ordinary token. No line slots, newline boundary logic, truncation, retries, guards, compile-time generation gates, blacklists, or repairs are allowed.

Population and automatic gate:

- Seal a frozen Pilot-30 before generation and run all five widths, exactly 150 rows.
- Before observing pilot outcomes, seal a proportional fixed 1000-row development/validation population from all 5815 MultiLine rows, excluding Pilot-30.
- Each width independently advances at Pilot Pass@1 at least `16/30`.
- If any finite width advances, run every advancing finite width and the paired `w=infinity` native baseline on the same fixed-full-1000.
- Never run the complete 5815-row population in this action.

Implementation/output scope:

- New code and raw/compact outputs: `experiments/frontier_gated_dreamon/` only.
- Authoritative compact research records may be updated after evidence exists.
- Frozen historical code, manifests, and results are read-only inputs.

Environment and baseline:

- GPU: idle NVIDIA H200 NVL, `CUDA_VISIBLE_DEVICES=0`.
- Python: `/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python`.
- Model: `Dream-org/DreamOn-v0-7B`, snapshot `8ccc74750e43177327f29dab9e91882ba759e194`.
- Official DreamOn source: commit `8a0a54918412eda9402a327646f7f067f7160ec8`.
- Dataset: `/home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-MultiLineInfilling.jsonl`, 5815 rows.
- Generation baseline: initial masks 64, max new tokens 64, 256 forward cap, entropy ordering, temperature 0.2, top-p 0.9, top-k null, algorithm temperature 0, one transfer token, full-sequence attention, native expand/delete/broadcast/stop behavior.

Planned commands:

```bash
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m pytest -q tests/test_frontier_gated_dreamon.py
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m experiments.frontier_gated_dreamon.build_manifests
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m experiments.frontier_gated_dreamon.runner equivalence
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m experiments.frontier_gated_dreamon.runner pilot
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m experiments.frontier_gated_dreamon.runner fixed-full-auto
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m experiments.frontier_gated_dreamon.analyze
/home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -m experiments.frontier_gated_dreamon.audit
```

Success criteria:

- `w=infinity` matches the unmodified native DreamOn implementation on final tokens/text, every selected position/action, and stop reason for at least five real samples, with ordinary/newline/expand/delete coverage.
- Every trace begins with exactly 64 middle masks; all frontier assertions pass and `frontier_violation=0`.
- Manifest checksums, counts, dedup keys, seeds, config hashes, JSON/JSONL parse, result counts, and frozen-file audit pass.
- Pilot and every automatically authorized fixed-full run complete or, if external GPU/runtime failure prevents completion, retain an exact resumable command and evidence-backed progress record.

Kill criteria:

- Stop before pilot if native equivalence, initial-state, frontier, ordinary-newline, resume/dedup, or schema gates fail.
- Stop any stage on config/manifest hash mismatch, duplicate unique key, unexpected population size, frozen-file mutation, or invariant assertion failure.
- Do not change generation parameters or add mechanisms to recover accuracy.

Known risks:

- Initial length 64 equals max-new-token budget 64, so expansion can only occur after native deletion frees dynamic length; equivalence coverage may require scanning more than five predeclared gate candidates.
- Per-step traces are compact but can still be large if several fixed-full configurations advance.
- This is development/validation evidence, not an official 5815 full result and not a frozen test.

Review policy: `reviewer_gate_disabled`; use local diff review plus fresh verification before experiment launch and before commit/push.
