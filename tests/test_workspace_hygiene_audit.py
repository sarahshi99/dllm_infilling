from __future__ import annotations

import unittest

from scripts.workspace_hygiene_audit import (
    READ_ONLY_PREFIXES,
    command_is_read_only,
    parse_tmux_panes,
    parse_worktree_porcelain,
    snapshot_is_complete,
    validate_pending_manifest,
)


class WorkspaceHygieneAuditTest(unittest.TestCase):
    def test_worktree_porcelain_parser_preserves_branch_and_head(self) -> None:
        parsed = parse_worktree_porcelain(
            "worktree /repo\nHEAD abc123\nbranch refs/heads/audit\n\n"
            "worktree /repo/.worktrees/live\nHEAD def456\ndetached\n"
        )
        self.assertEqual(parsed[0]["worktree"], "/repo")
        self.assertEqual(parsed[0]["head"], "abc123")
        self.assertEqual(parsed[0]["branch"], "refs/heads/audit")
        self.assertEqual(parsed[1]["detached"], "")

    def test_tmux_parser_reads_metadata_without_pane_content(self) -> None:
        parsed = parse_tmux_panes("m1\t%0\t101\t/repo/.worktrees/live\tpython\t0\n")
        self.assertEqual(
            parsed,
            [
                {
                    "session": "m1",
                    "pane": "%0",
                    "pane_pid": 101,
                    "cwd": "/repo/.worktrees/live",
                    "current_command": "python",
                    "dead": False,
                }
            ],
        )

    def test_pending_manifest_requires_false_flags_and_zero_counters(self) -> None:
        valid = {
            "approved": False,
            "executed": False,
            "files_deleted": 0,
            "files_moved": 0,
            "files_truncated": 0,
            "entries": [{"approved": False, "executed": False}],
        }
        self.assertEqual(validate_pending_manifest(valid), [])

        invalid = {
            "approved": True,
            "executed": False,
            "files_deleted": 1,
            "entries": [{"approved": False, "executed": "not-yet"}],
        }
        violations = validate_pending_manifest(invalid)
        self.assertIn("$.approved must be boolean false", violations)
        self.assertIn("$.files_deleted must be integer 0", violations)
        self.assertIn("$.entries[0].executed must be boolean false", violations)

    def test_only_fixed_read_only_command_prefixes_are_accepted(self) -> None:
        for prefix in READ_ONLY_PREFIXES:
            self.assertTrue(command_is_read_only(prefix))
        self.assertFalse(command_is_read_only(("git", "worktree", "add", "/tmp/unsafe")))
        self.assertFalse(command_is_read_only(("git", "reset", "--hard")))
        self.assertFalse(command_is_read_only(("tmux", "send-keys", "C-c")))
        self.assertFalse(command_is_read_only(("rm", "-rf", "outputs_clean")))

    def test_requested_runtime_checks_fail_closed(self) -> None:
        complete = {
            "tmux": [{"returncode": 0}],
            "processes": [{"ps_returncode": 0, "cwd_returncode": 0, "ps": "python", "cwd": "/repo"}],
            "line_counts": [{"returncode": 0, "lines": 12}],
        }
        self.assertTrue(snapshot_is_complete(complete))

        incomplete = {**complete, "tmux": [{"returncode": 1}]}
        self.assertFalse(snapshot_is_complete(incomplete))


if __name__ == "__main__":
    unittest.main()
