# Phase 5 Manual H200 Launch

Completion update, 2026-07-12 UTC: the user-authorized manual launch completed from baseline `45bead22e3d21daa707be724cf2bdcbbf776592a`. Base bank `1332/1332`, alpha auxiliary `728/728`, and F1–F4 completed. Conditional V0 verdict is `killed_corrected_within_task_gate_failed`. Frozen test remains sealed with `test_evaluation_count=0`. Phase 5 decision: `iterate`.

Historical pre-launch operational decision: `blocked_infrastructure`.

Historical pre-launch scientific decision: `scientific_pending` because no candidate bank or premise result existed yet; this was not a scientific failure.

The exact registered H200 command was retried once after the supervision audit correction and was rejected before process launch by the approval service:

```text
422 Unprocessable Entity: model not found: codex-auto-review
```

Post-failure audit: GPU process started `false`; candidate rows written `0`; frozen test `sealed`; `test_evaluation_count=0`.

The user can launch the corrected pushed experiment manually on the H200 host:

```bash
cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/phase5-method-falsification
bash scripts/manual_launch_phase5_h200_candidate_bank.sh
```

Equivalent explicit tmux command:

```bash
tmux new-session -d -s phase5-randomspanlight-bank \
  -c /home/shx/projects/dllm_infilling/git_workspace/.worktrees/phase5-method-falsification \
  "CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HOME=/home/shx/.cache/huggingface /home/shx/miniconda3/envs/dllm_env/bin/python experiments/phase5_randomspanlight_candidate_bank.py run --dataset-jsonl /home/shx/projects/dllm_infilling/git_workspace/data/HumanEval-RandomSpanInfillingLight.jsonl --output-dir outputs_clean/phase5_randomspanlight_candidate_bank_20260711_v1 --compact-dir analysis_outputs/phase5_randomspanlight_candidate_bank_20260711_v1 --smoke-cases 12 --auto-full 2>&1 | tee logs/paper_agent/20260712_phase5_randomspanlight_candidate_bank.log"
```

Monitor:

```bash
tmux attach -t phase5-randomspanlight-bank
tail -f logs/paper_agent/20260712_phase5_randomspanlight_candidate_bank.log
```

Do not launch F1–F4 until `full_audit.json` reports exactly `1332` base rows, zero missing/duplicate/error/frozen rows, and unchanged test lock. Then run the commands in `docs/paper_agent/current_action.md`. V0 consumes only `f3_deployable_proxy_scores.csv`; pass-trained supervised probe scores are diagnostic-only and rejected by V0.
