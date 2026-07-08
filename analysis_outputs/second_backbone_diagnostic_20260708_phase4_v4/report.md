# Second-Backbone Minimal Diagnostic

Verdict: `diagnostic_subset_oracle_gpu_blocked`.

Backbone: `Dream-org/Dream-Coder-v0-Base-7B`.
Cases: `15` stratified HumanEval SingleLine rows.

Policies:
- primary/control: `primary_cal_lite` from existing full Dream-Coder official-canvas run.
- current best simple length policy: `best_simple_lcal_bounded_repair` from existing full Dream-Coder run.
- oracle-sufficient canvas: fresh oracle run when `run_gpu_oracle=true`; otherwise marked `not_run`.
- E/F/G: not applicable because Dream-Coder runner has no trace-remasking adapter.

Oracle run status:
- run_gpu_oracle: `False`
- blocked reason: `auto_review_codex_auto_review_model_not_found_for_h200_gpu_command`

Key counts:
- missed-long oracle recoveries: `0`
- triggered-long oracle recoveries: `0`
- short-case regressions under best simple policy: `0`
