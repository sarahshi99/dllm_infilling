# DreamOn V2-Hard-v2 Boundary Decoder

Timestamp: 2026-08-01 UTC

Status: implementation verification; no empirical result yet.

Decision: `iterate`.

## Scope

This round evaluates only `v2_hard_v2_boundary` on the frozen 642-row / 115-base-problem exact-three-line development/mechanism population. It is an oracle structural diagnostic, not a held-out test. V2-OpenTail-v2 and Joint-OpenTail-v2 are explicitly out of scope.

## Why V2-Hard-v1 is not the efficacy test

V2-Hard-v1 remains immutable at Pass@1 `7/642`, compile `46/642`, and 421 forward-cap failures. It is now labeled `V2-Hard-v1 decoder/protocol failure diagnostic; not a clean test of progressive-slot efficacy` because three decoder defects confound the result:

1. every newline-containing hard-slot proposal was suppressed at vocabulary-logit level;
2. EOS deleted only one selected mask, creating deterministic expand/delete loops;
3. repeated exact deterministic transitions were not detected before the 256-forward cap.

The partial V2-OpenTail-v1 run is abandoned rather than resumed.

## Protocol-v2 revisions

- Newline proposals are O(1)-lookup online `line_boundary` actions after a one-time tokenizer-vocabulary metadata scan.
- The first normalized newline terminates the proposal; left text is strictly retokenized and committed, internal right text and all current-slot positions to the right are discarded diagnostically, and the fixed separator remains unchanged.
- EOS deletes the selected and right-side unresolved masks within the active region while preserving right-side resolved tokens.
- Every action records a full pre/post state hash and transition signature. A second identical transition ends the row as `exact_deterministic_cycle`.

All population/model/attention/activation/entropy/cap/scorer conditions otherwise match V2-Hard-v1.

## Preregistered gates

Smoke 5 requires 5/5 completed, zero unresolved/runtime/protocol/cycle/invariant/cross-region-delete rows, strict sequential activation, complete traces, and direct output extraction.

Pilot 30 requires 30/30 completed, compile at least 24/30, Pass@1 at least 10/30, zero failures/cycles, and complete post-generation oracle discard/reference diagnostics.

Full 642 runs only after pilot promotion. It stops if the first 50 contain at least five non-completed rows, if the cumulative non-completed rate exceeds 5% after 100 rows, or if any runtime/invariant damage occurs.

## Paths

- Protocol: `repro_results/dreamon_progressive_v2_hard_v2_protocol/protocol.json`
- Runner: `repro_scripts/run_dreamon_progressive_v2.py`
- Orchestration: `repro_scripts/run_dreamon_progressive_v2_hard_v2.sh`
- Tests: `tests/test_dreamon_hard_v2_boundary.py`
- Smoke: `repro_results/dreamon_progressive_v2_hard_v2_smoke5/`
- Pilot: `repro_results/dreamon_progressive_v2_hard_v2_pilot30/`
- Full: `repro_results/dreamon_progressive_v2_hard_v2_all642/`
