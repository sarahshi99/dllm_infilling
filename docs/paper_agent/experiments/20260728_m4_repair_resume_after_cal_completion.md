# M4 Repair Resume After Official CAL Completion

Timestamp: 2026-07-28 UTC
Plan: FAST-SPRINT-01 / M4 V0

## Action

Launch the pre-frozen `Semantic Particle Assembly` repair route as `12` technical smoke cases followed automatically by a deterministic `148`-group MultiLine-Core pre-screen. The fair primary comparison remains `assembly_without_repair` versus `best_single` at `512` standalone forwards; the repair arm is a separately labeled `576`-forward secondary comparison.

### Data-source reconciliation

The originally wired Phase-5 RandomSpanLight bank has `2,060` rows, while the user-authorized immutable bank is `40,632 = 5,079 × 8` and is MultiLine. A read-only visible-context audit found zero matches between the `164` RandomSpanLight rows and the 40,632-bank rows. It is therefore impossible to use the mandated bank while honestly calling the GPU repair run RandomSpanLight. The runner will instead select exactly one MultiLine source span per non-frozen base-task group by a fixed SHA-256 hash of the visible source identity, commit the compact 148-row manifest before GPU launch, and retain the old RandomSpanLight offline audit as historical structural evidence. This is a data/protocol reconciliation, not an outcome-driven method change; any cross-method ranking will mark M4 versus M1/M3 as cross-source rather than apples-to-apples.

## Reviewer Motivation and Hypothesis

A reviewer needs to know whether visible program fragments add value beyond best-single selection at equal compute, rather than whether an extra repair decode buys improvements. The hypothesis is that AST/basic-block/def-use fragment assembly changes candidates and yields a positive task-macro delta with more paired help than harm without reference, tests, task IDs, split labels, oracle length, or passed labels.

## Inputs, Environment, and Exact Command

- Dataset: outcome-blind one-span-per-group `HumanEval-MultiLineInfilling` subset, `148` non-frozen base-task groups, selected from the mandated bank before GPU launch.
- Model: `GSAI-ML/LLaDA-8B-Base` through the existing runner/cache.
- GPU: H200 GPU `0`; one unrelated process is retained. Pre-launch audit: ECC=`0`, about `130 GiB` free; projected post-load reserve exceeds `25 GiB`.
- Candidate bank: read-only `phase6_multiline_candidate_bank_20260712_v1/candidate_bank_raw.jsonl` (`40,632 = 5,079 × 8`); it is never rebuilt, moved, or overwritten.
- Log: `logs/paper_agent/20260728_m4_semantic_particle_assembly_multilinecore_v1.log`.
- Outputs: isolated append-only `outputs_clean/m4_{best,assembly,repair}_multilinecore_20260728_v1/` and compact `analysis_outputs/m4_semantic_particle_assembly_20260728_multilinecore_v1/`.

```bash
tmux new-session -d \
  -s m4-semantic-particle-assembly-sprint-v1 \
  -c /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1 \
  "bash scripts/manual_launch_m4_semantic_particle_assembly_sprint_v1.sh"
```

## Technical Success and Kill Criteria

Technical success requires the committed manifest to have exactly one non-frozen source span in each of `148` base-task groups, then smoke/full to have all three arms `148/148`, unique candidate keys, no missing/duplicate/error rows, repair exactly `64` forwards, canonical outputs append-only/resume no-op, frozen test `sealed` with count `0`, no OOM/ECC/driver/disk error, and no third research GPU process. Fail-stop on a malformed/non-deterministic manifest, duplicate keys, unexpected error rows, frozen intersection, OOM/ECC, persistent driver failure, or a below-25-GiB post-load reserve; do not kill the unrelated process.

## Outcome Handling and Documentation

`analysis/m4_semantic_particle_assembly.py` and its test were committed before launch. Once technical integrity completes, run its fixed grouped analyzer and record task-macro/cluster CI, paired effects, help/harm, fragment/provider/assembly activation, repair hash changes/fallback, forwards/token/wall/memory. Promotion is allowed only if the fixed equal-512 primary comparison has point delta `>0`, help>harm, and activated fragment/provider evidence. Update the method portfolio, runtime status, dashboard, checkpoint, ledger, and compact result report; raw code remains unreported.
