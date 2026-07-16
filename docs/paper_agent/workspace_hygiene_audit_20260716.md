# Workspace Hygiene and Coding-Agent Safety Audit — 2026-07-16

## Verdict

This was a non-destructive audit. No worktree was created, removed, replaced, reused, or pruned. No signal, tmux input, GPU launch, output rewrite, deletion, move, truncation, or partial-performance inspection occurred.

The cleanup manifest remains pending:

- files_deleted: 0
- files_moved: 0
- files_truncated: 0
- every approved field: false
- every executed field: false

## Verified audit identity

The initial read-only identity check matched all requested values:

- worktree: /home/shx/projects/dllm_infilling/git_workspace/.worktrees/workspace-hygiene-audit-20260716
- branch: codex/workspace-hygiene-audit-20260716
- baseline HEAD: 394d383194a3a827e8192801d975a97a78257de0
- initial git status: clean

git worktree list reported ten registered worktrees. The active ccfa-execution-sprint-v1 worktree was never used as this audit's working directory and was not edited.

## Active-job metadata snapshot

Tmux and process metadata were re-parsed rather than trusting historical PIDs. The pane/PID checks were taken immediately before the final output-line snapshot at 2026-07-16T04:54:19Z. This section intentionally excludes pass rates, row contents, help/harm, and every other partial-performance field.

### M1 RandomSpanLight

- tmux session: m1-randomspanlight-sprint-v1, live pane 0, pane PID 2251036
- actual Python child PID: 2251038, parent PID 2251036
- cwd: /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1
- process state observed: Rl+
- launcher log path: /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1/logs/paper_agent/20260716_m1_randomspanlight_sprint_v1.log

| Raw output path | Read-only line count |
| --- | ---: |
| outputs_clean/m1_randomspanlight_20260715_v1/stage1/m1_stage1_candidate_bank_raw.jsonl | 184 |
| outputs_clean/m1_randomspanlight_20260715_v1/generic/m1_equal_compute_generic_raw.jsonl | 12 |
| outputs_clean/m1_randomspanlight_20260715_v1/dependency_cone/m1_dependency_cone_raw.jsonl | 12 |

### M2 Constraint-Homotopy

- tmux session: m2-constraint-homotopy-sprint-v1, live pane 0, pane PID and actual Python PID 2254866
- cwd: /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1
- process state observed: Rsl+
- launcher log path: /home/shx/projects/dllm_infilling/git_workspace/.worktrees/ccfa-execution-sprint-v1/logs/paper_agent/20260715_m2_constraint_homotopy_sprint_v1.log

| Raw output path | Read-only line count |
| --- | ---: |
| outputs_clean/m2_vanilla_randomspanlight_20260715_sprint_v1/m2_vanilla_raw.jsonl | 126 |
| outputs_clean/m2_gradual_randomspanlight_20260715_sprint_v1/m2_gradual_raw.jsonl | 12 |
| outputs_clean/m2_abrupt_randomspanlight_20260715_sprint_v1/m2_abrupt_raw.jsonl | 12 |

## Workspace hygiene inventory

- outputs_clean and logs are ignored by .gitignore; no tracked raw output or log files were found.
- analysis_outputs has 495 tracked compact artifacts. This audit made no cleanup recommendation for them.
- git object database: 2,312 loose objects, 23.46 MiB, zero reported garbage.
- available filesystem capacity: 2.5 TiB at audit time.
- The M1/M2 launcher scripts were inspected without execution. The limited static scan over scripts, experiments, analysis, and tests found no worktree-management command, tmux send-keys action, signal command, or destructive cleanup command.

## Safety controls added

- AGENTS.md now defines a workspace-hygiene containment policy: inventory-only by default, no active-worktree cwd switch, no live-job intervention, and an explicit pending-manifest gate for cleanup.
- scripts/workspace_hygiene_audit.py is a read-only snapshot utility. It has no filesystem-write or apply option and permits only fixed read-only prefixes for git, tmux, ps, readlink, and wc.
- tests/test_workspace_hygiene_audit.py checks parser behavior, fixed read-only command allowlisting, and recursive manifest invariants.
- cleanup_apply_manifest.pending.json has no cleanup candidates, marks all live raw outputs as protected, and retains all approval/execution flags as false.

## Verification completed

- python -m unittest tests.test_workspace_hygiene_audit: 5 tests passed.
- python -m json.tool cleanup_apply_manifest.pending.json: passed.
- Recursive pending-manifest validator: passed.
- git diff --check: passed.

## Files prepared for focused commit

- AGENTS.md
- cleanup_apply_manifest.pending.json
- docs/paper_agent/workspace_hygiene_audit_20260716.md
- scripts/workspace_hygiene_audit.py
- tests/test_workspace_hygiene_audit.py

Suggested commit message: docs: add non-destructive workspace hygiene audit guardrails

## Commit handoff: blocked by read-only git metadata

At the final stage attempt, git add failed before any path was staged:

fatal: Unable to create '/home/shx/projects/dllm_infilling/git_workspace/.git/worktrees/workspace-hygiene-audit-20260716/index.lock': Read-only file system

No permission bypass was attempted. No commit or push was attempted after the failed stage.

git status --short after the failed stage:

    M AGENTS.md
    ?? cleanup_apply_manifest.pending.json
    ?? docs/paper_agent/workspace_hygiene_audit_20260716.md
    ?? scripts/workspace_hygiene_audit.py
    ?? tests/test_workspace_hygiene_audit.py

The required commit set is exactly:

- AGENTS.md
- cleanup_apply_manifest.pending.json
- docs/paper_agent/workspace_hygiene_audit_20260716.md
- scripts/workspace_hygiene_audit.py
- tests/test_workspace_hygiene_audit.py

From an external terminal with writable git metadata, run:

    cd /home/shx/projects/dllm_infilling/git_workspace/.worktrees/workspace-hygiene-audit-20260716
    git add -- AGENTS.md cleanup_apply_manifest.pending.json docs/paper_agent/workspace_hygiene_audit_20260716.md scripts/workspace_hygiene_audit.py tests/test_workspace_hygiene_audit.py
    git commit -m "docs: add non-destructive workspace hygiene audit guardrails"
    git push -u origin codex/workspace-hygiene-audit-20260716
