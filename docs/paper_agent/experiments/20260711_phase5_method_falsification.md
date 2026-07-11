# Phase 5 Method Falsification And Semantic Bridge V0 Incubation

Timestamp: 2026-07-11 UTC

This experiment brief freezes the Phase 5 protocol before candidate outcomes are inspected. It uses a single shared full RandomSpanLight candidate bank for independent F1–F4 premise tests and permits only Semantic Bridge V0 to cross from diagnostic to method, conditional on the F3 gate in `docs/paper_agent/current_action.md`.

The bank population is all `148` non-frozen `HumanEval-RandomSpanInfillingLight` rows. Each task has eight non-oracle candidates from canvas `16,32,64,128` and seed `0,1`; `64/0` is the fixed control. A seed-0 oracle-sufficient row is a labeled offline ceiling only. Generation uses LLaDA-8B-Base and `64` denoising steps throughout.

Deployable inputs are limited to prefix, suffix, generated candidate text, tokenizer/model inference signals, canvas, seed, and derived static/inference-visible features. Reference code, oracle length, verifier/unit-test outcomes, task IDs, split labels, and test-derived statistics are forbidden. Labels may be used only for offline fitting/evaluation with grouping by HumanEval base task. The controller frozen test remains sealed and is never loaded as an evaluation split.

Long generation is append-only/resumable. The stable key is `(row_key, candidate_kind, canvas_tokens, seed)`. Every milestone audits planned versus observed keys, duplicates, missing rows, JSONL parseability, compact/raw separation, and the frozen lock.

The registered but deferred methods are Constraint-Homotopy Infilling, Birth–Death Canvas Diffusion, and Semantic Particle Assembly. Metamorphic Equivariance is F2 auxiliary evaluation only. None may be implemented, mixed, or tuned in this round.
