# Action Equivalence Audit

This audit separates pass-level effects from candidate-level output changes.

## Aggregate

- task_count: `3`
- candidate_level_canvas_effect_count: `3`
- pass_level_canvas_effect_count: `1`
- output_changed_without_correctness_change_count: `5`
- config_different_but_output_equivalent_count: `5`

## Per Task

| Task | Unique Hashes | Equivalence Classes | Candidate-Level Canvas Effects | Pass-Level Canvas Effects | Output Changed But Correctness Same | Config Different But Output Equivalent |
|---|---:|---|---|---|---|---|
| `SingleLineInfilling/HumanEval/113/L3` | 2 | `['A_primary+B_route2_len32', 'C_oracle_sufficient+D_oracle_sufficient_steps96']` | `['C_oracle_sufficient', 'D_oracle_sufficient_steps96']` | `[]` | `['C_oracle_sufficient', 'D_oracle_sufficient_steps96']` | `[['C_oracle_sufficient', 'D_oracle_sufficient_steps96']]` |
| `SingleLineInfilling/HumanEval/116/L0` | 2 | `['A_primary', 'B_route2_len32+C_oracle_sufficient+D_oracle_sufficient_steps96']` | `['B_route2_len32', 'C_oracle_sufficient', 'D_oracle_sufficient_steps96']` | `['B_route2_len32', 'C_oracle_sufficient', 'D_oracle_sufficient_steps96']` | `[]` | `[['B_route2_len32', 'D_oracle_sufficient_steps96'], ['C_oracle_sufficient', 'D_oracle_sufficient_steps96']]` |
| `SingleLineInfilling/HumanEval/85/L0` | 2 | `['A_primary', 'B_route2_len32+C_oracle_sufficient+D_oracle_sufficient_steps96']` | `['B_route2_len32', 'C_oracle_sufficient', 'D_oracle_sufficient_steps96']` | `[]` | `['B_route2_len32', 'C_oracle_sufficient', 'D_oracle_sufficient_steps96']` | `[['B_route2_len32', 'D_oracle_sufficient_steps96'], ['C_oracle_sufficient', 'D_oracle_sufficient_steps96']]` |

## Required Checks

### `SingleLineInfilling/HumanEval/116/L0`
- `A_primary`: hash `1a26b0bfb1ab1ffa166dd00d968cc8533bb72db54c8ea09fa5eb6accee35f43d`, steps `64`, effective_steps `64`, early_commit `False`, stop_reason `no_remaining_masks`, compile `False`, error `SyntaxError`
- `B_route2_len32`: hash `94d3d9067fb51db3a18f5893eac2fc25f3ae0f54999c46939685b31c7cbe403c`, steps `64`, effective_steps `60`, early_commit `True`, stop_reason `global_gap_early_commit`, compile `True`, error `None`
- `C_oracle_sufficient`: hash `94d3d9067fb51db3a18f5893eac2fc25f3ae0f54999c46939685b31c7cbe403c`, steps `64`, effective_steps `60`, early_commit `True`, stop_reason `global_gap_early_commit`, compile `True`, error `None`
- `D_oracle_sufficient_steps96`: hash `94d3d9067fb51db3a18f5893eac2fc25f3ae0f54999c46939685b31c7cbe403c`, steps `96`, effective_steps `90`, early_commit `True`, stop_reason `global_gap_early_commit`, compile `True`, error `None`
- equivalence_classes: `[{'generated_text_sha256': '1a26b0bfb1ab1ffa166dd00d968cc8533bb72db54c8ea09fa5eb6accee35f43d', 'actions': ['A_primary']}, {'generated_text_sha256': '94d3d9067fb51db3a18f5893eac2fc25f3ae0f54999c46939685b31c7cbe403c', 'actions': ['B_route2_len32', 'C_oracle_sufficient', 'D_oracle_sufficient_steps96']}]`

### `SingleLineInfilling/HumanEval/85/L0`
- `A_primary`: hash `cc0e41f32b21606bed32aca9ec81fa6a472ee89b35412d2d30d0f6304b942afe`, steps `64`, effective_steps `64`, early_commit `False`, stop_reason `no_remaining_masks`, compile `False`, error `SyntaxError`
- `B_route2_len32`: hash `ff82f09561d32c7f5c790a0f7dd3028043ede4ca0269f6d3fc4bee7e663668d0`, steps `64`, effective_steps `63`, early_commit `True`, stop_reason `global_gap_early_commit`, compile `False`, error `SyntaxError`
- `C_oracle_sufficient`: hash `ff82f09561d32c7f5c790a0f7dd3028043ede4ca0269f6d3fc4bee7e663668d0`, steps `64`, effective_steps `63`, early_commit `True`, stop_reason `global_gap_early_commit`, compile `False`, error `SyntaxError`
- `D_oracle_sufficient_steps96`: hash `ff82f09561d32c7f5c790a0f7dd3028043ede4ca0269f6d3fc4bee7e663668d0`, steps `96`, effective_steps `93`, early_commit `True`, stop_reason `global_gap_early_commit`, compile `False`, error `SyntaxError`
- equivalence_classes: `[{'generated_text_sha256': 'cc0e41f32b21606bed32aca9ec81fa6a472ee89b35412d2d30d0f6304b942afe', 'actions': ['A_primary']}, {'generated_text_sha256': 'ff82f09561d32c7f5c790a0f7dd3028043ede4ca0269f6d3fc4bee7e663668d0', 'actions': ['B_route2_len32', 'C_oracle_sufficient', 'D_oracle_sufficient_steps96']}]`

### `SingleLineInfilling/HumanEval/113/L3`
- `A_primary`: hash `b1c9952d0408f69d6e80ebf718b350a5f87e2b406d5382112a4265634fdff27e`, steps `64`, effective_steps `29`, early_commit `True`, stop_reason `global_gap_early_commit`, compile `True`, error `UnitTestFailure`
- `B_route2_len32`: hash `b1c9952d0408f69d6e80ebf718b350a5f87e2b406d5382112a4265634fdff27e`, steps `64`, effective_steps `29`, early_commit `True`, stop_reason `global_gap_early_commit`, compile `True`, error `UnitTestFailure`
- `C_oracle_sufficient`: hash `9cb32a957675503e43d748232733ad1ba7989ccd78b69e4ac20287ad94015bda`, steps `64`, effective_steps `46`, early_commit `True`, stop_reason `global_gap_early_commit`, compile `True`, error `UnitTestFailure`
- `D_oracle_sufficient_steps96`: hash `9cb32a957675503e43d748232733ad1ba7989ccd78b69e4ac20287ad94015bda`, steps `96`, effective_steps `68`, early_commit `True`, stop_reason `global_gap_early_commit`, compile `True`, error `UnitTestFailure`
- equivalence_classes: `[{'generated_text_sha256': 'b1c9952d0408f69d6e80ebf718b350a5f87e2b406d5382112a4265634fdff27e', 'actions': ['A_primary', 'B_route2_len32']}, {'generated_text_sha256': '9cb32a957675503e43d748232733ad1ba7989ccd78b69e4ac20287ad94015bda', 'actions': ['C_oracle_sufficient', 'D_oracle_sufficient_steps96']}]`
