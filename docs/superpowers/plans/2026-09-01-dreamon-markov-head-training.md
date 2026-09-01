# DreamOn Markov-head training implementation plan

1. Add focused numerical/data/model tests first; verify they fail before implementation.
2. Implement deterministic OpenCoder normalization, grouping, HumanEval decontamination, hash split, manifest audit, and small audit samples.
3. Implement replay-state schema and official single-token DreamOn transition generation for the two authorized policies, with deterministic stratified per-problem reservoirs.
4. Implement the rank-256 head, common zero-output initialization, structural-token masking, TV/KL/CE losses, frozen-backbone replay, checkpoint/resume, training, validation, lambda calibration, and clustered bootstrap metrics.
5. Add a sequential launcher that freezes manifests, builds the shared bank, runs TV then releases GPU, runs KL from the common initialization, conditionally opens external test once, and writes the required result registry.
6. Run focused CPU tests, py_compile, tiny overfit/replay checks, then the authorized H200 stages. Verify every gate and artifact before each focused commit and final push.

Commit boundaries: implementation/tests; frozen data manifest; TV result; KL result; final comparison/audits/research records.
