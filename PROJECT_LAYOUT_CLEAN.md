# Clean project layout recommendation

## Keep both zones

```
project/
  expvision_dllm/           # legacy experimental code, frozen for reference
  scripts/                  # legacy runners, frozen for reference

  expvision_dllm_clean/     # clean package for the new phase
  clean_scripts/            # dedicated clean runners

  outputs/                  # legacy outputs
  outputs_clean/            # new clean outputs
```

## Why this layout

The next phase only needs a minimal baseline core plus explicit length-control experiments.
By separating the clean package from the legacy package, later CAL-lite and stopping experiments can be added without dragging historical branches back into the critical path.

## Immediate experiment entry points

- `python clean_scripts/run_vanilla_fixed.py ...`
- `python clean_scripts/run_vanilla_oracle.py ...`

## Planned next files

When phase 0 is done, the next additions should be:
- `expvision_dllm_clean/length_probe.py`
- `clean_scripts/run_cal_lite.py`
- `expvision_dllm_clean/stopping.py`
- `clean_scripts/run_cal_lite_stop.py`

The key rule is that each new experiment should add a new module and runner, rather than reopening the legacy monolith.
