#!/usr/bin/env python3
"""Run one persistent Codex audit session; never retry or spawn another agent."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/dreamon_markov_error_audit_20261013_v1"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(name, value):
    target = OUT / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)


def git(*args):
    try:
        command = ["git", "-C", str(ROOT)]
        if args and args[0] == "ls-remote":
            # This session's port-22 push/verification timed out. Port 443
            # succeeded with the existing key and known-host verification.
            # Keep the runner's final read on that transport; no config edit.
            command += ["-c", "core.sshCommand=ssh -p 443 -o Hostname=ssh.github.com "
                        "-o HostKeyAlias=github.com -o ConnectTimeout=10 "
                        "-o BatchMode=yes -o StrictHostKeyChecking=yes"]
        return subprocess.check_output(
            [*command, *args], text=True, timeout=30,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "session.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This audit already has an active session; refusing a second one.")
        if (OUT / "runner_exit.json").exists():
            raise SystemExit("This finite invocation already ended; inspect checkpoint before resuming.")
        command = [
            "/home/shx/.local/bin/codex", "exec", "--json", "--color", "never",
            "-C", str(ROOT), "-o", str(OUT / "session_final.zh.md"), "-",
        ]
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")
        started = utc()
        with (OUT / "session_prompt.zh.md").open("rb") as prompt, \
                (OUT / "session_events.jsonl").open("wb") as events, \
                (OUT / "session_stderr.log").open("wb") as errors:
            child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                     stdin=prompt, stdout=events, stderr=errors)
            save("session_launch.json", {
                "status": "starting", "started_utc": started,
                "wrapper_pid": os.getpid(), "codex_pid": child.pid,
                "command_argv": command, "branch": git("branch", "--show-current"),
                "source_commit": git("rev-parse", "HEAD"),
                "tmux_session": "dreamon_markov_error_audit_20261013",
                "permissions_and_auth": "inherited existing configuration unchanged",
                "gpu_visibility": "disabled", "retry_count": 0,
            })
            returncode = child.wait()
        branch = git("branch", "--show-current")
        head = git("rev-parse", "HEAD")
        remote = git("ls-remote", "--heads", "origin", "refs/heads/" + branch) if branch else None
        pushed = bool(remote and head and remote.split()[0] == head)
        coverage_path = OUT / "coverage_audit.json"
        coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else {}
        queue_path = OUT / "review_queue.jsonl"
        queue = [json.loads(line) for line in queue_path.read_text().splitlines() if line] if queue_path.exists() else []
        counts = {status: sum(row.get("status") == status for row in queue)
                  for status in ("pending", "evidence_ready", "reviewed", "missing_evidence", "failed_infrastructure")}
        save("runner_exit.json", {
            "started_utc": started, "finished_utc": utc(), "codex_exit_code": returncode,
            "actual_commit": head, "actual_remote_commit": remote,
            "push_verified": pushed, "queue_status_counts": counts,
            "coverage_audit_present": bool(coverage),
            "completion_marker_present": (OUT / "completion.json").exists(),
        })
        if not (OUT / "completion.json").exists():
            save("completion.json", {
                "status": "incomplete_session_ended", "total_to_review": len(queue),
                "reviewed": counts["reviewed"], "missing_evidence": counts["missing_evidence"],
                "unreviewed": counts["pending"] + counts["evidence_ready"],
                "failed_infrastructure": counts["failed_infrastructure"],
                "html_path": str(OUT / "casebook.html") if (OUT / "casebook.html").exists() else None,
                "exit_code": returncode, "actual_commit": head, "push_verified": pushed,
                "blocker": "Codex exited without writing its required completion record; no automatic retry.",
                "finished_utc": utc(),
            })
        return returncode


if __name__ == "__main__":
    raise SystemExit(main())
