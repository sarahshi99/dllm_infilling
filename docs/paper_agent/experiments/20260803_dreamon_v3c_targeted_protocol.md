# DreamOn V3-C targeted pure-newline protocol

Date: 2026-08-03 UTC

Decision: `iterate_v3c_targeted`

This round preserves V3-A (`v3_hard_budgeted`) as the protocol-fidelity control and preserves the negative/mixed V3-B Full result. It evaluates only:

- `v3_c0_budgeted_oneshot_pure_newline_veto`
- `v3_c_budgeted_nonconsuming_blankline`

The scientific question is whether the B subgroup effect comes from vetoing one erroneous pure-newline termination, or from accepting a blank-line structural intent and reconditioning with an inserted locked physical newline. Frozen-future-slot logits are recorded only as observational mechanism diagnostics and never influence generation.

## Frozen targeted population

The targeted population is derived only from V3-A Full `newline_boundary_event_details`; B outcomes are not read. A row is selected when the normalized proposal is exactly `"\n"`, left/right text are empty, the selected position is proven to be the active region's left edge, the A boundary action immediately empties the slot, and this is the unique strict event producing the final blank slot.

- Pure30 rows: `30`
- Base problems: `21`
- Slot distribution: `HARD_SLOT_0=6`, `HARD_SLOT_1=12`, `HARD_SLOT_2=12`
- Observed pure-newline token ID under the frozen tokenizer: `198`
- V3-A outcome on Pure30: Pass `3/30`, compile `22/30`, exact `0/30`
- Rows with resolved right tokens: `23/30`
- Rows with non-whitespace resolved right tokens: `17/30`

Artifacts:

- `manifests/v3_pure_newline_blank30.jsonl`, SHA256 `5946c7d6e6642861d2e4738616f5b39dcafa5feb5081983d9a8edf23dc55ad4f`
- `manifests/v3_pure_newline_slot2_12.jsonl`, SHA256 `37535ed030a85bba0152a9a71a792a0b61453b3109ca3cf50b541fd7afa8545a`
- `manifests/v3_pure_newline_early_slot01_18.jsonl`, SHA256 `60504fc2861d8ba7c1060c4070494cc4ce4e73597ebf2dfb6d785d857758995c`
- `manifests/v3_pure_newline_blank30.meta.json`

This is a post-hoc mechanism diagnostic, not a held-out population or unbiased performance sample.

## Frozen interventions

C0 spends at most one per-slot guard opportunity to veto the exact selected pure-newline boundary action without mutating the canvas. It performs one normal next forward and masks only the same `(canvas, position, token, line_boundary)` candidate once. A later same request falls back to A.

C spends at most one per-slot guard opportunity to insert exactly one locked structural newline immediately before the active content slot. It preserves the selected mask and all right-side resolved/masked content, keeps the same slot active, and reconditions on the next normal forward. A later same request falls back to A. The inserted newline counts toward global/context length but not hard-slot length or official expand budget.

Both methods use per-slot budget `1`, global budget `3`, and include all guard/pending/insertion state in config, resume state, hashes, traces, and transition signatures. No blanket nonempty rule, newline ban, retry loop, compile gate, BoundaryShift, AST repair, OpenTail, or Joint-OpenTail is permitted.

## Stages and frozen gate

1. Smoke5: first five Pure30 rows, engineering correctness only.
2. Pure30: primary targeted mechanism estimate.
3. Historical Pilot30: regression/isolation check, no absolute Pass threshold.
4. Full642: independently authorized only when all engineering gates pass and the method reaches Pure30 Pass `>=6/30`.

If both methods pass, both run Full642. If one passes, only that method runs. If neither passes, the round stops. Thresholds cannot be lowered after observation.

## Historical evidence preserved

- V3-A Full is a later user-authorized post-hoc Full after the original Pilot gate failed: `281/642` Pass, `538/642` compile.
- V3-B Full is an oracle structural diagnostic: `280/642` Pass, `471/642` compile.
- B versus A is `15/16` wins/losses with substantial compile harm and no overall Pass benefit.

The new targeted decision does not overwrite or weaken those results.
