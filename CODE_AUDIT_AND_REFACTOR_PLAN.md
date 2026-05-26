# Code audit and refactor plan

## What was found in the uploaded project

### 1. Baseline and historical branches are entangled
- `decode_structured.py` is 1688 lines and contains multiple generations of repair / verifier / remask logic.
- `config.py` currently carries knobs for P0, P1, G, R/M, and S series in one shared dataclass.
- The runner layer duplicates setup across multiple scripts, which makes later rollback and experiment hygiene harder.

### 2. The current baseline chain is not isolated enough
- `decode_vanilla.py` still mixes core decode logic, detailed reconstruction diagnostics, and full trace logging.
- Step-level logging currently captures large text payloads and repeated diagnostics, which is expensive and easy to break.

### 3. Snapshot / selector analysis code has grown into a separate subsystem
- `snapshot_pipeline.py` and `snapshot_selector.py` together exceed 1400 lines.
- This is useful for offline analysis, but it should not remain on the critical path for the next phase.

### 4. The project needs a frozen legacy zone and a clean experiment zone
- The old code should remain available for reproducing legacy results.
- New phase-0 / phase-1 experiments should live in a new directory with a minimal shared core.

## What was created

A new clean package was added under `expvision_dllm_clean/` together with dedicated clean scripts under `clean_scripts/`.

### New clean package
- `expvision_dllm_clean/config.py`
- `expvision_dllm_clean/dataset.py`
- `expvision_dllm_clean/modeling.py`
- `expvision_dllm_clean/verifier.py`
- `expvision_dllm_clean/decode.py`
- `expvision_dllm_clean/evaluation.py`
- `expvision_dllm_clean/logging.py`
- `expvision_dllm_clean/runner.py`

### New clean scripts
- `clean_scripts/run_vanilla_fixed.py`
- `clean_scripts/run_vanilla_oracle.py`

## Design decisions

### A. Legacy code stays untouched
The original `expvision_dllm/` and `scripts/` directories were left in place. They now function as a frozen legacy zone rather than the recommended base for new work.

### B. The clean package only supports the current next step
The new package intentionally only covers:
- dataset loading
- fixed-length vanilla decode
- oracle-length vanilla diagnostic
- shared verifier / logging / evaluation utilities

This is deliberate. It prevents old G/R/M/S logic from leaking into the next 2x2 experimental base.

### C. Mask length selection is now an explicit interface
The clean package has a single explicit switch: `mask_length_source = fixed | oracle`.
This is the correct abstraction boundary for phase 0. Later, CAL-lite can be added as a third value without contaminating the vanilla chain.

## Recommended next coding steps

1. Run the new clean fixed-length baseline and confirm it reproduces the existing P0 result on a small smoke subset.
2. Run the new oracle-length diagnostic and compare sample-level win/loss/tie against the clean fixed baseline.
3. Only after that, add a small `length_probe.py` module for CAL-lite.
4. Only after CAL-lite is stable, add a separate stopping module.

## What not to do now

- Do not add CAL-lite directly into the legacy `decode_vanilla.py`.
- Do not move G/R/M/S logic into the new clean package yet.
- Do not migrate to DreamOn / DAEDAL / token-level stopping before the clean 2x2 base exists.
