# Phase 5 Supervision Audit Correction

Timestamp: 2026-07-12 UTC

Starting HEAD: `3315ed82d42770e3ed7d8ae20e9d5f1570940ff6` after fetch and exact remote/local verification.

## Defect

The original F3 implementation fit grouped-OOF logistic models on functional `passed` labels and supplied the resulting `score_combined_bridge` to V0. Calling that selector training-free and recording `outcomes_used_for_selection=false` was incorrect because outcome supervision entered selection indirectly through fitted weights.

## Correction

F3 is split into:

1. `supervised_probe_diagnostic`: pass-trained OOF logistic probes, diagnostic evidence only; global AUROC and within-task ranking may be reported but cannot authorize or supply V0 scores.
2. `deployable_bridge_proxy`: fixed deterministic AST/def-use/boundary/confidence formula, frozen in `deployable_bridge_formula.json` and `.md`, with no fitted parameters, pass labels, reference code, reference length, reference semantic targets, errors, task IDs, or split labels.

The current mechanism is named `AST/def-use bridge proxy V0`. It is not a full Semantic Bridge Projection or Abductive Program-State Bridge. Program-state analysis, genuine backward obligations, bridge anchors, and denoising intervention remain future work.

The primary authorization metric is cross-canvas within-task passing-vs-failing pairwise ranking. The corrected gate also requires overall within-task ranking above chance, positive grouped-bootstrap deltas against four deterministic baselines, positive paired selection net versus fixed64 and confidence, and no short-bucket net regression.

## Verification

Add tests that reject supervised scores in V0, reject formula provenance containing fitted weights, reject global-AUROC-only authorization, require within-task/cross-canvas evidence, validate indirect label-use audit fields, and produce a killed report when the gate fails. Run focused tests, py_compile, JSON/CSV parse checks, frozen-lock assertions, and `git diff --check` before the H200 command.

Frozen controller test remains `sealed`; `test_evaluation_count=0`. The candidate-bank generation protocol is unchanged.
