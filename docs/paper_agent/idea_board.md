# Research Idea Board

Updated: 2026-07-10 UTC

Purpose: shared idea board for user, Codex, and web ChatGPT. Ideas are ordered by current CCF-A value, not by execution cost. Negative evidence is explicitly useful and must be recorded.

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
Minimum experiment: Run a bounded Dream-Coder oracle-sufficient diagnostic on a larger train/calibration/validation-derived manifest, then publish a case taxonomy by stratum, oracle length, selected length, primary/simple/oracle pass, and error type.
Expected positive outcome: Oracle canvas recoverability persists, and the mixed missed-vs-triggered pattern can be explained by model-specific length selection or task composition.
Expected negative outcome: Recoverability disappears or becomes inconsistent; the paper should keep Dream-Coder as weak/mixed evidence and avoid model-agnostic claims.
Cost: CPU manifest construction is done; GPU diagnostic is bounded to 37 cases, one backbone, oracle-sufficient action only.
Risks: Dream-Coder has no trace-remasking adapter, so E/F/G refinement cannot be compared; train/calibration/validation-derived cases are not a held-out benchmark.
Decision rule: Treat as strong support only if the expanded manifest shows nontrivial oracle-canvas recovery with no short-regression explosion and a stable, interpretable case taxonomy. Otherwise keep as mixed generalization evidence.
Related files: `analysis_outputs/second_backbone_oracle_diagnostic_20260708_phase4_fullaccess_v3/`, `analysis_outputs/research_planning_20260708_cpu_claim_audit/`, `analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/`, `analysis_outputs/second_backbone_oracle_diagnostic_20260708_dreamcoder_expanded37_v1/`

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
Status: proposed
Scientific question: Are canvas-limited and rescue-limited regimes intrinsic to unknown-length DLLM infilling, or are they backbone-specific and mediated by each model's length-selection behavior?
Why it matters for CCF-A: It turns mixed Dream-Coder evidence into a sharper scientific question rather than a failed replication.
Minimum experiment: Compare LLaDA H200 and Dream-Coder by stratum: primary selected length, oracle length, primary error type, oracle-canvas pass, and whether triggered-long proxies recover.
Expected positive outcome: The paper can report a model-dependent boundary and explain why LLaDA triggered-long remains rescue-limited while Dream-Coder may recover some triggered proxies.
Expected negative outcome: No stable pattern; downgrade to "mixed second-backbone audit" and avoid broad claims.
Cost: Low CPU after Dream-Coder expanded diagnostic; GPU only for expanded Dream-Coder oracle run.
Risks: Small strata can overfit; Dream-Coder lacks trace-remasking actions.
Decision rule: Require stratum-level consistency on an expanded manifest before using this in central claims.
Related files: `analysis_outputs/dreamcoder_expanded_manifest_20260708_cpu_v1/`, `analysis_outputs/research_planning_20260708_cpu_claim_audit/dreamcoder_case_taxonomy.csv`

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
Status: running
Scientific question: What exact tables and figures can support a diagnostic-driven mixed paper without overclaiming?
Why it matters for CCF-A: The project is now evidence-rich but claim-fragile; paper quality depends on clean tables, negative evidence, and source separation.
Minimum experiment: Build a compact evidence matrix: H200 baselines, oracle attribution, controller V1/V2/V3, Dream-Coder diagnostic, second-regime status, LR-DLLM blocker, and frozen-test integrity.
Expected positive outcome: Web ChatGPT and the user can review the paper's claim stack directly.
Expected negative outcome: Evidence remains too scattered; defer broad writing until the top two diagnostics are stronger.
Cost: Low CPU.
Risks: Consolidation can accidentally mix A6000 historical and H200 current evidence; must preserve labels.
Decision rule: Every table row must cite one compact artifact and carry status labels: H200 evidence, historical reference, synthetic, blocked, validation-only, or sealed.
Related files: `analysis_outputs/research_planning_20260708_cpu_claim_audit/`, `docs/paper_agent/codex_handoff.latest.zh.md`, `docs/results/run_registry.md`
