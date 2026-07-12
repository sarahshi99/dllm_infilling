# Current Paper-Agent Action

## 2026-07-12 Completion Update

The manual H200 run completed from baseline `45bead22e3d21daa707be724cf2bdcbbf776592a`. Full base bank `1332/1332` and alpha auxiliary `728/728` passed all integrity gates with `test_evaluation_count=0`. F1–F4 completed. The combined deterministic proxy had positive point estimates but failed the preregistered all-baseline grouped-bootstrap lower-bound condition. Conditional V0 wrote `killed_corrected_within_task_gate_failed`; no fallback or extra generation ran.

Phase 5 decision: `iterate`.

Authoritative result report: `docs/paper_agent/experiments/20260712_phase5_h200_candidate_bank_result.md`.

Timestamp: 2026-07-11 UTC

## Action Name

Phase 5 audit correction and full-first AST/def-use bridge proxy V0 incubation.

## Current Phase And Baseline

- Authoritative branch: `codex/risk-controlled-dynamic-rescue`.
- Verified remote/local starting HEAD: `b8ae031cf3b430833613aaf29083535edc0738f6`.
- Isolated implementation worktree: `.worktrees/phase5-method-falsification`.
- Dataset: all `148` allowed non-frozen `HumanEval-RandomSpanInfillingLight` rows.
- Model: `GSAI-ML/LLaDA-8B-Base`.
- Fixed control: canvas `64`, seed `0`, same `64`-step decoding protocol.
- Frozen controller test: `sealed`; `test_evaluation_count=0`.

## Reviewer Motivation

Phase 4 established a canvas-adequacy versus rescue-adequacy gap but did not produce a safe deployable controller. Phase 5 asks whether independently motivated, inference-visible structure can rank already-generated candidates before another method family is promoted. The shared full bank prevents each premise diagnostic from receiving a bespoke generation distribution.

Audit correction, 2026-07-12: the existing OOF logistic score is pass/fail-supervised and therefore cannot be a deployable training-free selector. It is retained only as `supervised_probe_diagnostic`. The current deployable mechanism is renamed `AST/def-use bridge proxy V0`; it uses a fixed deterministic formula and no pass/reference fitting. It is not a full Abductive Program-State Bridge or Semantic Bridge Projection. Genuine program-state analysis, backward obligations, bridge anchors, and denoising intervention remain future stages.

## Registered Portfolio

The methods remain separate:

1. `AST/def-use bridge proxy V0`: current round, deterministic F3/F4 proxy and conditional reranker only. `Semantic Bridge Projection` remains a future full-method concept, not the current implementation.
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

## AST/Def-Use Bridge Proxy V0 Formula And Gate

The deployable formula is frozen before outcomes:

- `prefix_proxy = mean(prefix_candidate_use_coverage, prefix_available_use_coverage, prefix_boundary_indent_match, min(prefix_candidate_def_use_count/3, 1))`
- `suffix_proxy = mean(suffix_required_recovery, suffix_boundary_indent_match, min(candidate_suffix_def_use_count/3, 1))`
- `token_canvas_proxy = candidate_canvas_fill_ratio clipped to [0,1]`
- `ordinary_confidence_proxy = ordinary_confidence clipped to [0,1]`
- `combined_proxy = 0.30*prefix_proxy + 0.30*suffix_proxy + 0.20*full_parse_passed + 0.10*min(candidate_control_structure_count/3,1) + 0.10*ordinary_confidence_proxy`

The `supervised_probe_diagnostic` may fit pass-trained grouped-OOF logistic models, but its scores can never authorize or enter V0.

V0 is authorized only if the deterministic combined proxy:

- has within-task passing-vs-failing pairwise accuracy `>0.5`;
- has cross-canvas within-task pairwise accuracy `>0.5` (registered primary ranking metric);
- beats deterministic prefix-only, suffix-only, token/canvas, and ordinary-confidence scores on the primary metric, with every grouped-bootstrap `95%` delta lower bound `>0`;
- has positive paired selection net versus fixed64 and versus confidence selection; and
- has no short-bucket net regression versus either fixed64 or confidence.

Global AUROC is secondary diagnostic only. If any gate condition fails, V0 is killed for this round. No substitute heuristic or fusion is allowed.

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

Known risks: candidate seeds may be less diverse than expected; alpha-renaming may be only approximately measurable through generation; the AST/def-use proxy may correlate with syntax rather than functionality; grouped uncertainty may be wide over only `148` task groups. These risks prevent positive claims unless the corrected preregistered gate is met.

## Expected Documentation Outputs

Update the handoff, review manifest, idea board, queue, decision log, CCF-A readiness memo, experiment results, evidence snapshot, current research design/plan, and both literature notes. End with exactly one Phase 5 decision: `advance`, `iterate`, `reframe`, `stop`, or `blocked`.
