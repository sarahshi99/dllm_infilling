# Current Paper-Agent Action

Timestamp: 2026-07-11 UTC

## Action Name

Phase 5 independent-method falsification and full-first Semantic Bridge incubation.

## Current Phase And Baseline

- Authoritative branch: `codex/risk-controlled-dynamic-rescue`.
- Verified remote/local starting HEAD: `b8ae031cf3b430833613aaf29083535edc0738f6`.
- Isolated implementation worktree: `.worktrees/phase5-method-falsification`.
- Dataset: all `148` allowed non-frozen `HumanEval-RandomSpanInfillingLight` rows.
- Model: `GSAI-ML/LLaDA-8B-Base`.
- Fixed control: canvas `64`, seed `0`, same `64`-step decoding protocol.
- Frozen controller test: `sealed`; `test_evaluation_count=0`.

## Reviewer Motivation

Phase 4 established a canvas-adequacy versus rescue-adequacy gap but did not produce a safe deployable controller. Phase 5 asks whether independently motivated, inference-visible semantic structure can rank already-generated candidates before another method family is promoted. The shared full bank prevents each premise diagnostic from receiving a bespoke generation distribution.

## Registered Portfolio

The methods remain separate:

1. `Semantic Bridge Projection`: current round, premise diagnostic F3 and conditional V0 reranker only.
2. `Constraint-Homotopy Infilling`: registered, not implemented this round.
3. `Birth–Death Canvas Diffusion`: registered, not implemented this round.
4. `Semantic Particle Assembly`: registered, not implemented this round.
5. `Metamorphic Equivariance`: auxiliary evaluator F2 only; never called correctness.

No method fusion is authorized.

## Shared Candidate Bank

Each allowed task receives eight non-oracle candidates:

- canvas lengths `16,32,64,128`;
- seeds `0,1`;
- `64/0` is labeled `fixed64_control`;
- all use `64` denoising steps and the same vanilla decoding protocol.

One additional seed-0 oracle-sufficient candidate is retained only as `diagnostic_ceiling`; reference/oracle fields are prohibited from deployable feature extraction and selection. Stage B therefore has exactly `9` rows per task (`108` smoke, `1332` full). F2 alpha-renaming mirrors are generated only after the full base-bank gate, under a separate auxiliary audit, and are not counted as Stage B candidates.

Raw generated code stays in ignored local output. Compact tracked artifacts contain hashes, metrics, schemas, audits, CSV/JSON/Markdown, and LaTeX only.

## Premise Diagnostics

- F1: candidate/hash diversity, parsable AST diversity, reference-only offline semantic-unit precision/coverage, complementary correct fragments among all-fail tasks, consensus-landmark precision.
- F2: deterministic semantics-preserving alpha-renaming only; test whether equivariance adds predictive value for functional pass after controlling for canvas, seed, and ordinary confidence. Stability is not correctness.
- F3: prefix-forward, suffix-backward, combined bridge, token-length, and ordinary-confidence features; required-identifier/def-use/control-structure recovery, semantic-horizon prediction, candidate-ranking AUROC, grouped confidence intervals.
- F4: within-task passing-versus-failing pairwise ranking across canvas lengths using inference-visible scores, with grouped confidence intervals.

## Semantic Bridge V0 Gate

V0 is authorized only if combined bridge:

- exceeds prefix-only, suffix-only, token-length, and ordinary-confidence candidate-ranking AUROC by at least `0.01`; and
- has a grouped-bootstrap `95%` delta interval strictly above `0` against every baseline.

If any comparison fails, V0 is killed for this round. No substitute heuristic or fusion is allowed.

## Exact Commands

CPU preparation and tests:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python -m unittest tests/test_phase5_candidate_bank.py tests/test_phase5_premise_falsification.py
/home/shx/miniconda3/envs/dllm_env/bin/python -m py_compile experiments/phase5_randomspanlight_candidate_bank.py analysis/phase5_premise_falsification.py analysis/phase5_semantic_bridge_v0.py
```

Approved H200 smoke, then automatic full continuation on a passed audit:

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface /home/shx/miniconda3/envs/dllm_env/bin/python experiments/phase5_randomspanlight_candidate_bank.py run --dataset-jsonl /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-RandomSpanInfillingLight.jsonl --output-dir outputs_clean/phase5_randomspanlight_candidate_bank_20260711_v1 --compact-dir analysis_outputs/phase5_randomspanlight_candidate_bank_20260711_v1 --smoke-cases 12 --auto-full
```

CPU diagnostics and conditional V0:

```bash
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/phase5_premise_falsification.py --bank-dir outputs_clean/phase5_randomspanlight_candidate_bank_20260711_v1 --compact-bank-dir analysis_outputs/phase5_randomspanlight_candidate_bank_20260711_v1 --output-dir analysis_outputs/phase5_premise_falsification_20260711_v1
/home/shx/miniconda3/envs/dllm_env/bin/python analysis/phase5_semantic_bridge_v0.py --bank-dir outputs_clean/phase5_randomspanlight_candidate_bank_20260711_v1 --compact-bank-dir analysis_outputs/phase5_randomspanlight_candidate_bank_20260711_v1 --premise-dir analysis_outputs/phase5_premise_falsification_20260711_v1 --output-dir analysis_outputs/phase5_semantic_bridge_v0_20260711_v1
```

## Environment And Paths

- GPU: approved host/unsandboxed H200 path, `CUDA_VISIBLE_DEVICES=0`.
- Python: `/home/shx/miniconda3/envs/dllm_env/bin/python`.
- Log: `logs/paper_agent/20260711_phase5_randomspanlight_candidate_bank.log`.
- Local raw bank: `outputs_clean/phase5_randomspanlight_candidate_bank_20260711_v1/`.
- Compact outputs:
  - `analysis_outputs/phase5_method_portfolio_20260711_v1/`
  - `analysis_outputs/phase5_randomspanlight_candidate_bank_20260711_v1/`
  - `analysis_outputs/phase5_premise_falsification_20260711_v1/`
  - `analysis_outputs/phase5_semantic_bridge_v0_20260711_v1/`

## Success And Kill Criteria

Smoke success requires schema validation, canvas-128 protocol validation, evaluator success, resumability, exact expected row counts, zero duplicate keys, zero missing keys, zero frozen rows, and unchanged `test_evaluation_count=0`. Passing smoke automatically continues to all `148` cases.

Stop immediately if canvas `128` cannot run under the same protocol, the dataset population is not exactly `148`, any frozen group appears, raw code would enter tracked compact artifacts, duplicate/missing rows survive resume audit, or the test lock changes.

Known risks: candidate seeds may be less diverse than expected; alpha-renaming may be only approximately measurable through generation; bridge features may correlate with syntax rather than functionality; grouped uncertainty may be wide over only `148` task groups. These risks prevent positive claims unless the preregistered gate is met.

## Expected Documentation Outputs

Update the handoff, review manifest, idea board, queue, decision log, CCF-A readiness memo, experiment results, evidence snapshot, current research design/plan, and both literature notes. End with exactly one Phase 5 decision: `advance`, `iterate`, `reframe`, `stop`, or `blocked`.
