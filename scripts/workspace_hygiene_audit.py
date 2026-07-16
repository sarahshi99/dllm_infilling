#!/usr/bin/env python3
"""Emit a workspace-hygiene snapshot using only read-only host commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


READ_ONLY_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("git", "branch", "--show-current"),
    ("git", "rev-parse", "HEAD"),
    ("git", "status", "--short"),
    ("git", "worktree", "list", "--porcelain"),
    ("tmux", "list-panes"),
    ("ps", "-ww"),
    ("readlink", "-f"),
    ("wc", "-l", "--"),
)
MUTATION_COUNTERS = ("files_deleted", "files_moved", "files_truncated")
TMUX_FORMAT = (
    "#{session_name}\t#{pane_id}\t#{pane_pid}\t#{pane_current_path}\t"
    "#{pane_current_command}\t#{pane_dead}"
)


def command_is_read_only(command: Sequence[str]) -> bool:
    normalized = tuple(command)
    return any(normalized[: len(prefix)] == prefix for prefix in READ_ONLY_PREFIXES)


def run_read_only(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    if not command_is_read_only(command):
        raise ValueError(f"Command is outside the read-only allowlist: {list(command)!r}")
    return subprocess.run(
        list(command),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )


def parse_worktree_porcelain(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        if not raw_line:
            if current:
                records.append(current)
                current = {}
            continue
        if " " in raw_line:
            key, value = raw_line.split(" ", 1)
        else:
            key, value = raw_line, ""
        if key in {"worktree", "HEAD", "branch"}:
            current[key.lower()] = value
        else:
            current[key] = value
    if current:
        records.append(current)
    return records


def parse_tmux_panes(text: str) -> list[dict[str, Any]]:
    panes: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        parts = raw_line.split("\t")
        if len(parts) != 6:
            continue
        session, pane, pane_pid, cwd, command, dead = parts
        panes.append(
            {
                "session": session,
                "pane": pane,
                "pane_pid": int(pane_pid),
                "cwd": cwd,
                "current_command": command,
                "dead": dead == "1",
            }
        )
    return panes


def validate_pending_manifest(value: Any) -> list[str]:
    """Return no-apply invariant violations without changing the manifest."""

    violations: list[str] = []

    def visit(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                child_path = f"{path}.{key}"
                if key in {"approved", "executed"} and child is not False:
                    violations.append(f"{child_path} must be boolean false")
                if key in MUTATION_COUNTERS and (type(child) is not int or child != 0):
                    violations.append(f"{child_path} must be integer 0")
                visit(child, child_path)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")

    visit(value, "$")
    return violations


def command_text(command: Sequence[str], *, cwd: Path) -> tuple[str, str, int]:
    completed = run_read_only(command, cwd=cwd)
    return completed.stdout.strip(), completed.stderr.strip(), completed.returncode


def line_count(path: Path, *, cwd: Path) -> dict[str, Any]:
    stdout, stderr, returncode = command_text(("wc", "-l", "--", str(path)), cwd=cwd)
    record: dict[str, Any] = {"path": str(path), "returncode": returncode}
    if returncode:
        record["error"] = stderr
        return record
    fields = stdout.split(maxsplit=1)
    record["lines"] = int(fields[0])
    return record


def tmux_snapshot(session: str, *, cwd: Path) -> dict[str, Any]:
    stdout, stderr, returncode = command_text(
        ("tmux", "list-panes", "-t", session, "-F", TMUX_FORMAT),
        cwd=cwd,
    )
    snapshot: dict[str, Any] = {"session": session, "returncode": returncode}
    if returncode:
        snapshot["error"] = stderr
    else:
        snapshot["panes"] = parse_tmux_panes(stdout)
    return snapshot


def process_snapshot(pid: int, *, cwd: Path) -> dict[str, Any]:
    ps_stdout, ps_stderr, ps_returncode = command_text(
        ("ps", "-ww", "-p", str(pid), "-o", "pid=,ppid=,stat=,etime=,args="),
        cwd=cwd,
    )
    cwd_stdout, cwd_stderr, cwd_returncode = command_text(
        ("readlink", "-f", f"/proc/{pid}/cwd"),
        cwd=cwd,
    )
    return {
        "pid": pid,
        "ps_returncode": ps_returncode,
        "ps": ps_stdout,
        "ps_error": ps_stderr,
        "cwd_returncode": cwd_returncode,
        "cwd": cwd_stdout,
        "cwd_error": cwd_stderr,
    }


def make_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).resolve()
    branch, _, _ = command_text(("git", "branch", "--show-current"), cwd=repo)
    head, _, _ = command_text(("git", "rev-parse", "HEAD"), cwd=repo)
    status, _, _ = command_text(("git", "status", "--short"), cwd=repo)
    worktrees, _, _ = command_text(("git", "worktree", "list", "--porcelain"), cwd=repo)

    manifest: dict[str, Any] | None = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = {
            "path": str(manifest_path),
            "violations": validate_pending_manifest(manifest_value),
        }

    expected_worktree = str(Path(args.expected_worktree).resolve()) if args.expected_worktree else None
    return {
        "audit_mode": "non_destructive",
        "audit_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(repo),
        "identity": {
            "branch": branch,
            "head": head,
            "status_short": status.splitlines(),
            "expected_worktree": expected_worktree,
            "expected_branch": args.expected_branch,
            "expected_head": args.expected_head,
            "worktree_matches": expected_worktree is None or str(repo) == expected_worktree,
            "branch_matches": args.expected_branch is None or branch == args.expected_branch,
            "head_matches": args.expected_head is None or head == args.expected_head,
        },
        "worktrees": parse_worktree_porcelain(worktrees),
        "tmux": [tmux_snapshot(session, cwd=repo) for session in args.tmux_session],
        "processes": [process_snapshot(pid, cwd=repo) for pid in args.process_pid],
        "line_counts": [line_count(Path(path), cwd=repo) for path in args.line_count],
        "manifest": manifest,
        "mutation_counters": {key: 0 for key in MUTATION_COUNTERS},
        "declared_read_only_prefixes": [list(prefix) for prefix in READ_ONLY_PREFIXES],
    }


def snapshot_is_complete(snapshot: dict[str, Any]) -> bool:
    tmux_ok = all(item["returncode"] == 0 for item in snapshot["tmux"])
    process_ok = all(
        item["ps_returncode"] == 0
        and item["cwd_returncode"] == 0
        and bool(item["ps"])
        and bool(item["cwd"])
        for item in snapshot["processes"]
    )
    line_count_ok = all(item["returncode"] == 0 and "lines" in item for item in snapshot["line_counts"])
    return tmux_ok and process_ok and line_count_ok


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--expected-worktree")
    parser.add_argument("--expected-branch")
    parser.add_argument("--expected-head")
    parser.add_argument("--tmux-session", action="append", default=[])
    parser.add_argument("--process-pid", action="append", type=int, default=[])
    parser.add_argument("--line-count", action="append", default=[])
    parser.add_argument("--manifest")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    snapshot = make_snapshot(parse_args(sys.argv[1:] if argv is None else argv))
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    identity = snapshot["identity"]
    valid_identity = all(
        identity[key] for key in ("worktree_matches", "branch_matches", "head_matches")
    )
    manifest = snapshot["manifest"]
    valid_manifest = manifest is None or not manifest["violations"]
    return 0 if valid_identity and valid_manifest and snapshot_is_complete(snapshot) else 2


if __name__ == "__main__":
    raise SystemExit(main())
