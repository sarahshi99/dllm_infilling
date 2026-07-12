# AST/Def-Use Bridge Proxy V0 Fixed Formula

Provenance: `fixed_before_outcomes`. Parameters: `hand_fixed_constants_only`.

- `deployable_proxy_prefix_only`: `mean(prefix_candidate_use_coverage, prefix_available_use_coverage, prefix_boundary_indent_match, min(prefix_candidate_def_use_count/3, 1))`
- `deployable_proxy_suffix_only`: `mean(suffix_required_recovery, suffix_boundary_indent_match, min(candidate_suffix_def_use_count/3, 1))`
- `deployable_proxy_token_canvas`: `clip(candidate_canvas_fill_ratio, 0, 1)`
- `deployable_proxy_ordinary_confidence`: `clip(ordinary_confidence, 0, 1)`
- `deployable_proxy_combined`: `0.30*prefix_only + 0.30*suffix_only + 0.20*full_parse_passed + 0.10*min(candidate_control_structure_count/3,1) + 0.10*ordinary_confidence`

Scope boundary: This is a deterministic AST/def-use/boundary proxy, not full program-state analysis, backward obligations, bridge anchors, or denoising intervention.
