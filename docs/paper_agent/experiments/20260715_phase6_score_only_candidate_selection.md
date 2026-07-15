# Phase6 Score-Only Candidate Selection / M1 Historical Precursor — 2026-07-15

Input is the completed immutable MultiLine bank: `40632 = 5079 × 8` candidate rows, `148` base-task clusters, zero missing/duplicate/error. This analysis is **not M1 full**: it performs no second-stage dependency-cone or generic remasking.

Selectors were fixed without outcome tuning: fixed64 seed0, ordinary confidence, Phase5 prefix-only proxy, Phase5 combined proxy, fixed score-only abductive selector, and best-of-8 offline oracle ceiling. Primary estimand is equal-weight task macro accuracy with 10,000 cluster bootstrap and task-group label-swap; span micro is descriptive.

- Ordinary confidence versus fixed64: macro delta `-0.0620`, 95% CI `[-0.0944,-0.0285]`.
- Phase5 combined proxy versus fixed64: macro delta `+0.0531`, CI `[+0.0148,+0.0957]`.
- Fixed score-only abductive selector versus fixed64: macro delta `+0.0200`, CI `[-0.0152,+0.0599]`; conclusion `weak`.
- Best-of-8 oracle ceiling is an offline upper bound only; it cannot authorize deployment, M1 promotion, threshold changes, or method fusion.

Compact outputs: `analysis_outputs/phase6_score_only_candidate_selection_20260715_v1/`.
