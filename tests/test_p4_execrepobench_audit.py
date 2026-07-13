import tempfile
import unittest
from pathlib import Path

from experiments.p4_execrepobench_audit import AuditError, _license_audit, build_audit, parser


def row(index: int, fill_type: str, repo_name: str) -> dict[str, object]:
    return {
        "repo_name": repo_name,
        "file_name": f"pkg/file_{index}.py",
        "prefix_code": "def f():\n",
        "suffix_code": "\n    return value\n",
        "middle_code": "    value = 1\n",
        "context_code": ["pkg/context.py", "value = 1\n"],
        "fill_type": fill_type,
    }


class ExecRepoBenchAuditTest(unittest.TestCase):
    def test_smoke_plan_covers_all_six_fill_types_and_multiple_repositories(self) -> None:
        records = [row(index, f"type_{index}", "repo_a" if index % 2 else "repo_b") for index in range(6)]
        audit, plan = build_audit(records)
        self.assertEqual(audit["record_count"], 6)
        self.assertEqual(audit["fill_type_count"], 6)
        self.assertEqual(audit["smoke_case_count"], 6)
        self.assertEqual(audit["smoke_repository_count"], 2)
        self.assertEqual({item["fill_type"] for item in plan}, {f"type_{index}" for index in range(6)})
        self.assertNotIn("middle_code", plan[0])

    def test_schema_audit_rejects_missing_public_field(self) -> None:
        records = [row(index, f"type_{index}", "repo_a" if index % 2 else "repo_b") for index in range(6)]
        del records[0]["suffix_code"]
        with self.assertRaisesRegex(AuditError, "suffix_code"):
            build_audit(records)

    def test_schema_audit_rejects_non_six_fill_type_population(self) -> None:
        records = [row(index, "only_type", "repo_a" if index % 2 else "repo_b") for index in range(6)]
        with self.assertRaisesRegex(AuditError, "six fill types"):
            build_audit(records)

    def test_cli_has_no_final_results_switch(self) -> None:
        args = parser().parse_args(
            ["--dataset-root", "dataset", "--evaluator-root", "evaluator", "--output-dir", "output"]
        )
        self.assertEqual(str(args.dataset_root), "dataset")
        self.assertFalse(hasattr(args, "allow_final"))

    def test_license_audit_records_hash_and_declared_hint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")
            audit = _license_audit(root, "README.md")
        self.assertEqual(audit["declared_license_hint"], "mit")
        self.assertEqual(len(audit["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
