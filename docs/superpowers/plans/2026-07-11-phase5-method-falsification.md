# Phase 5 Independent-Method Falsification Plan

## Task 1: Synchronize The Research Record

Files: Phase 5 experiment brief, coordination docs, CCF-A gap ledger, literature notes, and `analysis_outputs/phase5_method_portfolio_20260711_v1/`.

Verification: JSON parse for the review manifest, Markdown diff hygiene, explicit frozen-test invariants, and separate registration of all five ideas.

## Task 2: Build The Shared Bank Runner

Files: `experiments/phase5_randomspanlight_candidate_bank.py`, `tests/test_phase5_candidate_bank.py`.

Behavior: build exactly `148` allowed rows; validate canvas 128; append/resume stable keys; keep raw text local; write compact hashes/metrics; run a 12-case smoke and automatically continue full on a passed gate.

Verification: unit tests, py_compile, dry-run manifest audit, resume/duplicate/missing-key tests, frozen-lock assertion.

## Task 3: Run Smoke And Full H200 Bank

Command: exact H200 command in `docs/paper_agent/current_action.md`.

Expected Stage B rows: smoke `12 * 9 = 108`; full `148 * 9 = 1332`, including `1184` deployable rows and `148` diagnostic-ceiling rows. After the full-bank audit passes, F2 alpha-renaming mirrors run as a separately counted/audited auxiliary block and do not alter Stage B row totals. Raw JSONL remains under ignored `outputs_clean/`; compact summaries go to `analysis_outputs/phase5_randomspanlight_candidate_bank_20260711_v1/`.

Kill: any canvas-128 protocol mismatch, frozen row, duplicate/missing key, schema/evaluator failure, or changed test lock.

## Task 4: Implement And Run F1–F4

Files: `analysis/phase5_premise_falsification.py`, `tests/test_phase5_premise_falsification.py`.

Outputs: F1 diversity/semantic-unit diagnostics; F2 alpha-renaming equivariance; F3 bridge ablations and AUROC/grouped intervals; F4 within-task pairwise ranking. Labels and references are offline-only and audited.

Verification: synthetic unit tests for transformations, feature extraction, AUROC, grouped bootstrap, pairwise ranking, and forbidden-feature rejection; full parse/row audits.

## Task 5: Apply The F3 Gate

Files: `analysis/phase5_semantic_bridge_v0.py`, compact V0 output directory.

If all preregistered comparisons pass, fit/evaluate the standalone combined-bridge reranker with grouped out-of-fold predictions and report requested ablations, pairwise help/harm, buckets, significance, and cost. Otherwise, write a killed gate report and do not implement another method.

## Task 6: Close The Record And Push

Update all requested coordination/evidence files with explicit pass/fail/kill outcomes and one final decision. Run unit tests, py_compile, CSV/JSON parses, row/duplicate audits, forbidden-feature audit, `git diff --check`, local diff review, commit, push the focused Phase 5 changes to the authoritative branch, and record the exact pushed HEAD in the handoff.
