# Research Idea Board

Updated: 2026-07-11 UTC

Purpose: shared idea board for user, Codex, and web ChatGPT. Ideas are ordered by current CCF-A value, not by execution cost. Negative evidence is explicitly useful and must be recorded.

## IDEA-011: AST/def-use bridge proxy V0

Source: user
Status: blocked_on_shared_bank
Scientific question: Can a fixed AST/def-use/boundary proxy rank passing candidates above failing candidates within the same task without fitted correctness labels or reference information?
Minimum experiment: F3/F4 over the full shared 148-case bank. Keep pass-trained OOF logistic models as `supervised_probe_diagnostic` only. Compare the deterministic combined proxy against deterministic prefix-only, suffix-only, token/canvas, and ordinary-confidence scores.
Decision rule: Implement standalone V0 only if within-task and cross-canvas ranking exceed chance, every primary grouped-bootstrap delta lower bound is `>0`, paired selection net is positive versus fixed64 and confidence, and short-bucket net regression is absent. Global AUROC is secondary only.
Combination rule: Must remain standalone; no homotopy, birth–death, particle, controller, or cal-lite fusion.
Related files: `docs/paper_agent/experiments/20260711_phase5_method_falsification.md`, `analysis_outputs/phase5_method_portfolio_20260711_v1/`

Scope boundary: This is not full Semantic Bridge Projection. Genuine program-state analysis, backward obligations, bridge anchors, and denoising intervention remain future stages.

## IDEA-012: Constraint-Homotopy Infilling

Source: user
Status: registered_not_implemented
Scientific question: Does a gradual schedule of inference-visible semantic constraints improve rescue quality relative to abrupt constraints under equal compute?
Minimum experiment: Future standalone smoke/full protocol after the Semantic Bridge round closes.
Decision rule: Do not implement in this round; do not combine with any other method.

## IDEA-013: Birth–Death Canvas Diffusion

Source: user
Status: registered_not_implemented
Scientific question: Can a fixed-compute population over canvas hypotheses preserve useful diversity while reallocating compute via birth/death decisions?
Minimum experiment: Future standalone equal-compute population audit with explicit particle accounting.
Decision rule: Do not implement in this round; do not borrow theoretical guarantees from continuous birth–death sampling without proof.

## IDEA-014: Semantic Particle Assembly

Source: user
Status: registered_not_implemented
Scientific question: Are complementary inference-visible fragments in all-fail candidate sets composable into correct programs without execution/reference guidance?
Minimum experiment: F1 is diagnostic only; future assembly method needs a separate preregistered protocol.
Decision rule: Do not implement in this round; no fragment fusion after seeing outcomes.

## IDEA-015: Metamorphic Equivariance Auxiliary Evaluator

Source: user
Status: blocked_on_shared_bank
Scientific question: Does strict local alpha-renaming equivariance add predictive value for functional pass after controlling for canvas, seed, and confidence?
Minimum experiment: F2 paired mirrors over reference-verified alpha transformations.
Decision rule: Report predictive increment or negative evidence; never call stability correctness and never use it as an oracle.

## IDEA-001: Stress second-regime construction and diagnostic

Source: joint
Status: done
Scientific question: Does the canvas-limited vs rescue-limited split survive outside easy SingleLine-style construction when the infill regime stresses multiline context, random spans, longer gaps, and under-selection?
Why it matters for CCF-A: This is now the largest official-data claim-boundary question. The full allowed official population is not easy and shows both large oracle-canvas recoverability and substantial rescue-limited/harm-risk behavior.
Minimum experiment: Completed a source-labeled official 120-case manifest and bounded diagnostic over `HumanEval-MultiLineInfilling`, `HumanEval-RandomSpanInfilling`, and `HumanEval-RandomSpanInfillingLight`; completed the approved smaller 48-case hard-tail diagnostic derived from fixed first-pass labels; completed reviewer-requested full104 hard-tail taxonomy/stress; completed full allowed official population diagnostic over all non-frozen rows.
Expected positive outcome: Control or deployable policy fails on a meaningful fraction while oracle-sufficient canvas recovers a nonzero subset, giving cross-regime evidence for diagnostic claims.
Expected negative outcome: Extreme/random-span strata remain mostly rescue-limited or oracle harms control enough that second-regime must be written as a scope boundary for current canvas-only claims.
Cost: First-pass GPU, 48-case hard-tail GPU, full104 hard-tail GPU, and full allowed official GPU diagnostic are completed. No more second-regime GPU work is planned unless a concrete bug appears.
Risks: Official random-span/extreme cases may stress semantics more than canvas length; oracle canvas can harm control; hard-tail reuse can overfit if described as a benchmark rather than follow-up diagnostic.
Decision rule: Write official second-regime as mixed stress evidence. Full allowed strengthens the mixed diagnostic claim; the 120-case first pass remains the preregistered/unbiased estimate; fixed hard-tail runs provide taxonomy/robustness. Do not run additional second-regime GPU work unless a concrete bug appears.
Related files: `analysis_outputs/second_regime_official_manifest_20260708_v1/`, `analysis_outputs/second_regime_official_diagnostic_20260708_v1/`, `analysis_outputs/second_regime_official_hard_tail_manifest_20260708_v1/`, `analysis_outputs/second_regime_official_hard_tail_diagnostic_20260708_v1/`, `analysis_outputs/second_regime_official_hard_tail_full104_20260708_v1/`, `analysis_outputs/second_regime_official_full_allowed_diagnostic_20260709_v1/`, `analysis_outputs/second_regime_official_data_recovery_20260708_cpu_v1/`, `docs/paper_agent/experiment_queue.md`

## IDEA-002: Dream-Coder expanded diagnostic and case-level taxonomy

Source: codex
Status: done
Scientific question: Is Dream-Coder's mixed 15-case result a sampling artifact, or does it expose a real model-dependent canvas/rescue boundary that differs from LLaDA H200?
Why it matters for CCF-A: A second backbone can upgrade the paper from a single-model diagnostic to a stronger generalization audit, but only if the qualitative claim is bounded correctly.
Minimum experiment: Completed a bounded Dream-Coder expanded37 oracle-sufficient diagnostic and optional full allowed SingleLine diagnostic, then published case/taxonomy summaries by stratum, oracle length, selected length, primary/simple/oracle pass, and error type.
Expected positive outcome: Oracle canvas recoverability persists, and the mixed missed-vs-triggered pattern can be explained by model-specific length selection or task composition.
Expected negative outcome: Recoverability disappears or becomes inconsistent; the paper should keep Dream-Coder as weak/mixed evidence and avoid model-agnostic claims.
Cost: CPU manifest construction, bounded 37-case GPU diagnostic, and optional full allowed 927-case GPU diagnostic are completed.
Risks: Dream-Coder has no trace-remasking adapter, so E/F/G refinement cannot be compared; train/calibration/validation-derived cases are not a held-out benchmark.
Decision rule: Use as Dream-Coder second-backbone diagnostic evidence because oracle-canvas recovery persists, but do not claim model-agnostic confirmation because the missed/triggered split remains model-dependent.
Related files: `analysis_outputs/second_backbone_oracle_diagnostic_20260708_phase4_fullaccess_v3/`, `analysis_outputs/research_planning_20260708_cpu_claim_audit/`, `analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/`, `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/`, `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/`

## IDEA-003: Substitute strong baseline pack for blocked LR-DLLM

Source: codex
Status: proposed
Scientific question: If LR-DLLM remains blocked by missing algorithmic detail, what fair oracle-free baseline pack can replace it without pretending to be an official reproduction?
Why it matters for CCF-A: Reviewers will expect strong baselines; a transparent substitute baseline pack is more defensible than a weak or invented LR-DLLM adapter.
Minimum experiment: Assemble a baseline table from protocol-matched local CAL, V6, fixed-canvas controls, Dream-Coder existing full runs, and any official-code baselines that are actually runnable; include compute, data, and protocol comparability.
Expected positive outcome: The paper has a credible baseline section despite LR-DLLM blockage.
Expected negative outcome: No strong substitute is protocol-compatible; the paper must foreground diagnostic claims and list LR-DLLM as a blocked external baseline.
Cost: Low CPU documentation plus existing compact result extraction; no GPU unless a new official baseline becomes runnable.
Risks: Baselines may mix protocols or hardware histories if not clearly separated; reviewers may still ask for LR-DLLM.
Decision rule: Include only baselines with runnable code or existing audited outputs, explicit protocol labels, and no hidden test usage.
Related files: `analysis_outputs/lrdllm_final_attempt_20260708_phase4_v4/`, `docs/paper_agent/lrdllm_protocol_audit.zh.md`, `docs/results/run_registry.md`

## IDEA-004: Controller route closure analysis

Source: joint
Status: done
Scientific question: Can the paper close the controller route scientifically, showing why oracle action-bank headroom fails to become a safe deployable policy?
Why it matters for CCF-A: A rigorous negative result can become a contribution if it explains the gap between upper bound, harm risk, calibration size, and inference-visible features.
Minimum experiment: Consolidated V1/V2/V3 validation curves, oracle upper bound, selected top-k policies, harm confidence bounds, and frozen-test decisions into one route-closure table and narrative.
Expected positive outcome: The paper can claim a diagnostic gap: recovery space exists, but risk-controlled inference-time selection remains unsolved under sealed-test discipline.
Expected negative outcome: The closure is too thin; more controller tuning would still not be justified without a new evidence source.
Cost: Low CPU consolidation; no GPU.
Risks: Overstating closure could look like giving up; underexplaining negative results weakens contribution.
Decision rule: Close Controller V4 unless a new non-validation evidence source appears. Do not open frozen test on V3.
Related files: `analysis_outputs/controller_validation_h200_20260707_v1_replay/`, `analysis_outputs/controller_v2_h200_20260707_phase3_v2_validation/`, `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/`, `analysis_outputs/controller_route_closure_20260708_v1/`, `docs/paper_agent/controller_route_closure_table_plan.md`

## IDEA-005: Model-dependent canvas/rescue boundary analysis

Source: codex
Status: done
Scientific question: Are canvas-limited and rescue-limited regimes intrinsic to unknown-length DLLM infilling, or are they backbone-specific and mediated by each model's length-selection behavior?
Why it matters for CCF-A: It turns mixed Dream-Coder evidence into a sharper scientific question rather than a failed replication.
Minimum experiment: Completed Dream-Coder expanded37 and full allowed SingleLine diagnostics, then compared primary/simple/oracle pass, oracle gain, harm, and rescue-limited counts against LLaDA H200 qualitative claims.
Expected positive outcome: The paper can report a model-dependent boundary and explain why LLaDA triggered-long remains rescue-limited while Dream-Coder may recover some triggered proxies.
Expected negative outcome: No stable pattern; downgrade to "mixed second-backbone audit" and avoid broad claims.
Cost: Completed bounded 37-case GPU diagnostic and optional full allowed 927-case GPU diagnostic; remaining work is CPU writing.
Risks: Small strata can overfit; Dream-Coder lacks trace-remasking actions.
Decision rule: Use Dream-Coder as second-backbone diagnostic evidence for oracle-canvas recoverability, but forbid model-agnostic confirmation of the LLaDA missed-vs-triggered split.
Related files: `analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/`, `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/`, `analysis_outputs/dreamcoder_full_allowed_singleline_diagnostic_20260710_v1/`, `analysis_outputs/research_planning_20260708_cpu_claim_audit/dreamcoder_case_taxonomy.csv`

## IDEA-006: Error taxonomy across backbones and regimes

Source: codex
Status: proposed
Scientific question: Which failure modes are canvas-limited, which are rescue-limited, and which are verifier/task-specific across LLaDA, Dream-Coder, SingleLine, and second-regime stress?
Why it matters for CCF-A: A strong taxonomy can support a diagnostic paper even without a positive controller.
Minimum experiment: Build compact per-case taxonomy from existing outputs: error type, compile status, selected vs oracle length, primary/simple/oracle pass, stratum, and policy family.
Expected positive outcome: Failure modes cluster in interpretable ways and support a figure/table.
Expected negative outcome: Errors are heterogeneous; taxonomy becomes a cautionary appendix rather than a main claim.
Cost: Low CPU.
Risks: Existing compact CSV may omit enough candidate/error detail for deep taxonomy; must not copy raw generated code.
Decision rule: Promote only if the taxonomy yields stable clusters tied to the paper's mechanisms.
Related files: `analysis_outputs/research_planning_20260708_cpu_claim_audit/`, `analysis_outputs/route2_error_analysis_20260617_165806/`, `analysis_outputs/oracle_canvas_attribution_20260703_phase2_attr_v2/`

## IDEA-007: Compute-matched oracle-free policy pack

Source: web
Status: revised
Scientific question: Are apparent gains from longer canvas or rescue policies due to better decisions, or just more compute/canvas budget?
Why it matters for CCF-A: Compute-matched comparisons make the negative controller result and oracle upper bound much harder to dismiss.
Minimum experiment: Compare V6, conservative top-k, always-expand, fixed-canvas, and compute-matched simple gates using existing validation/action-bank costs and pass/loss counts.
Expected positive outcome: The paper can show that naive extra compute causes harm while targeted policies are safer but too weak.
Expected negative outcome: Compute matching erases the diagnostic gap; claims need narrowing.
Cost: Low CPU using existing action-bank and V3 cost curves.
Risks: Cost proxies may not transfer across backbones; average cost can hide tail latency.
Decision rule: Use only as validation/diagnostic evidence, not as held-out test performance.
Related files: `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/cost_risk_pareto.csv`, `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/topk_policy_curves.csv`

## IDEA-008: Confidence/probe signal interpretability audit

Source: codex
Status: proposed
Scientific question: Which inference-visible signals actually predict recoverability versus harm, and where do they fail?
Why it matters for CCF-A: It explains why risk-controlled deployment is hard even when oracle action-bank headroom exists.
Minimum experiment: Analyze score distributions and feature ablations for recoverable/harmable/noninformative validation cases, with calibration-vs-validation shift.
Expected positive outcome: The paper gets an interpretable account of weak selector signal and calibration limits.
Expected negative outcome: Signals are too noisy; support negative controller result but not a strong interpretability contribution.
Cost: Low CPU.
Risks: Post-hoc overinterpretation; small validation split.
Decision rule: Include only if feature signals align with preregistered/forbidden-feature discipline and improve explanation beyond pass counts.
Related files: `analysis_outputs/controller_feasibility_h200_20260707_phase3_v2_feasibility/`, `analysis_outputs/controller_v3_h200_20260708_v3_candidate_screen_v3/`

## IDEA-009: Under-selection stress dataset construction

Source: user
Status: rejected
Scientific question: Can we intentionally construct tasks where length under-selection is common enough to stress policies, without turning the dataset into an artificial easy oracle-canvas demo?
Why it matters for CCF-A: Official second-regime first-pass already produced nontrivial failures, so synthetic stress is lower priority unless hard-tail official analysis fails to isolate mechanisms.
Minimum experiment: Defer synthetic stress. Official first-pass plus 48-case hard-tail already separate canvas-recoverable, rescue-limited, deployable-harm, and oracle-harm cases well enough for current paper framing.
Expected positive outcome: A clearly labeled stress set exposes nontrivial canvas-limited failures and selector risk not captured by official first-pass.
Expected negative outcome: Stress set is too synthetic or too easy; use only as appendix/unblock evidence.
Cost: Medium CPU+GPU, but currently not selected.
Risks: Dataset construction bias; reviewers may reject it as benchmark evidence; it could distract from newly available official evidence.
Decision rule: Do not run synthetic stress in the current step; revive only if official hard-tail evidence is found buggy or insufficient by web review.
Related files: `analysis_outputs/second_regime_official_hard_tail_manifest_20260708_v1/`, `analysis_outputs/second_regime_official_hard_tail_diagnostic_20260708_v1/`, `analysis_outputs/second_regime_unblock_20260708_phase4_continue_v3/`

## IDEA-010: Paper table and figure evidence consolidation

Source: codex
Status: done
Scientific question: What exact tables and figures can support a diagnostic-driven mixed paper without overclaiming?
Why it matters for CCF-A: The project is now evidence-rich but claim-fragile; paper quality depends on clean tables, negative evidence, and source separation.
Minimum experiment: Completed compact evidence matrix, main results table, official second-regime write-up, failure taxonomy, and paper claim rewrite.
Expected positive outcome: Web ChatGPT and the user can review the paper's claim stack directly.
Expected negative outcome: Evidence remains too scattered; defer broad writing until the top two diagnostics are stronger.
Cost: Low CPU, completed.
Risks: Consolidation can accidentally mix A6000 historical and H200 current evidence; must preserve labels.
Decision rule: Every table row must cite one compact artifact and carry status labels: H200 evidence, official diagnostic, hard-tail stress, blocked, validation-only, or sealed.
Related files: `analysis_outputs/paper_evidence_consolidation_20260710_v1/`, `analysis_outputs/research_planning_20260708_cpu_claim_audit/`, `docs/paper_agent/codex_handoff.latest.zh.md`, `docs/results/run_registry.md`

## IDEA-011: DreamOn boundary carry hypothesis

Source: DreamOn V3 diagnostics
Status: deferred after reframe
Scientific question: Can next-line-like resolved text discarded by online boundaries be carried into the next slot without reference dependence or post-hoc truncation?
Evidence: B Full has resolved discard on 356/642 rows; 51/164 complete discarded texts exactly match the oracle next line. The post-hoc A/B Full comparison shows nonempty guard alone does not improve Pass (B 15 wins / 16 losses versus A) and therefore does not explain the remaining boundary-related residual.
Decision rule: Do not implement automatically. Require independent review and a fresh matched protocol.

## IDEA-012: Pure-newline termination versus blank-line reconditioning

Source: V3-A blank-slot trace audit
Status: completed; advance to independent review
Scientific question: Is the narrow B subgroup effect explained by vetoing one exact erroneous pure-newline termination, or does inserting a real locked blank physical line provide useful reconditioning while preserving the content canvas?
Evidence before execution: A Full contains exactly 30 strict leading pure-newline events that immediately blank the final slot; A passes 3/30. Twenty-three events discard resolved right tokens and seventeen discard non-whitespace resolved right tokens.
Decision rule: C0/C independently require Pure30 Pass at least 6/30 plus complete engineering isolation before Full642. Frozen-future logits are observational diagnostics only and cannot enter selection.
Forbidden expansion: no blanket nonempty, compile gate, BoundaryShift/carry, AST repair, OpenTail, Joint-OpenTail, or held-out run.

Result: C0 reached 5/30 on Pure30 and failed its Full gate. C reached 14/30, then completed Full642 at 292/642 versus A 281/642. C vs A was 12 wins / 1 loss; C vs C0 on Pure30 was 10 wins / 1 loss. No-trigger isolation passed on 612/612 Full rows. Physical blank-line reconditioning is supported over simple veto on this post-hoc development diagnostic, while the frozen-future confidence hypothesis is weakened by only 2/31 positive events. Next step is independent review and a separately preregistered held-out validation, not another development variant.
