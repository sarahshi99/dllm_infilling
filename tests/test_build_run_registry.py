from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.build_run_registry import summarize_run


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class BuildRunRegistryTest(unittest.TestCase):
    def test_summarize_run_reads_singular_oracle_bucket_pass_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "full_example_20260528"
            write_json(
                run_dir / "summary.json",
                {
                    "num_samples": 2,
                    "pass_rate": 0.5,
                    "oracle_bucket_pass_rate": {"<=8": 1.0, "9-12": 0.0},
                },
            )
            write_json(run_dir / "config.json", {"model": {"model_path": "example/model"}})
            (run_dir / "results.jsonl").write_text(
                json.dumps({"metrics": {"passed": True}}) + "\n"
                + json.dumps({"metrics": {"passed": False}}) + "\n",
                encoding="utf-8",
            )

            row = summarize_run(run_dir)

        self.assertEqual(row["oracle_bucket_pass_rates"], {"<=8": 1.0, "9-12": 0.0})


if __name__ == "__main__":
    unittest.main()
