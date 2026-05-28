# Run Registry

This file will track meaningful experiment runs and point to local raw outputs without committing all raw `results.jsonl` files to normal git.

Status values:

- `global_best`
- `env_control`
- `candidate`
- `diagnostic`
- `superseded`
- `obsolete`

Initial entries will be generated from `analysis_outputs/experiment_scoreboard.md` and the local `outputs_clean/` directory.
