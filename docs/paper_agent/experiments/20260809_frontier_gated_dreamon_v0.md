# Frontier-Gated DreamOn V0 experiment brief

Date: 2026-08-09 UTC

Stage: independent method implementation and development/validation execution.

Reviewer motivation: determine whether restricting only DreamOn's per-step commit-position frontier improves multi-line infilling without line slots, newline semantics, retries, compile feedback, or other structural interventions.

Hypothesis: native global confidence ordering may commit far-ahead masks before the left context is stable. A finite frontier may improve causal coherence, while `w=infinity` must exactly recover native DreamOn.

Comparison: five paired configurations `w={1,4,8,16,infinity}` on identical samples, model, seeds, prompt/canvas, action semantics, and evaluator.

Dataset/model/metric:

- HumanEval MultiLine Infilling, 5815-row source population.
- DreamOn-v0-7B snapshot `8ccc74750e43177327f29dab9e91882ba759e194`.
- Per-row functional Pass@1, compile, completion, action/cycle diagnostics, forwards, wall/GPU compute.
- Pilot-30 development/mechanism population; fixed-full-1000 development/validation population.

Candidate routes considered:

1. Frontier-only position eligibility. Selected because it is the requested single intervention and admits a strict native-limit equivalence check.
2. Local-attention frontier. Rejected because it would change model context and violate full-forward equivalence.
3. Newline/structure-aware frontier. Rejected because newline must remain an ordinary token and structural mechanisms are out of scope.

Files:

- New implementation, tests, manifests, outputs, summaries, and report under `experiments/frontier_gated_dreamon/`.
- Compact authoritative records under `docs/paper_agent/` only after stage status is factual.
- No historical DreamOn decoder, manifest, or result file may change.

Execution, success, kill criteria, environment, and commands are recorded in `docs/paper_agent/current_action.md`.

Expected documentation outputs: `report.zh.md`, `run_manifest.json`, `review_manifest.latest.json`, a new Chinese Codex handoff, and factual updates to experiment results, evidence snapshot, run registry/queue, idea board, decision log, dashboard/checkpoint, and activity ledger.
