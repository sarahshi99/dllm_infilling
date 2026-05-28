from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from analysis.build_a6000_scoreboard_section import build_section


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class BuildA6000ScoreboardSectionTest(unittest.TestCase):
    def test_build_section_uses_latest_runs_and_pairwise_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            outputs_dir = tmp_path / "outputs_clean"
            analysis_dir = tmp_path / "analysis_outputs" / "a6000_midcons_longrescue"

            write_json(
                outputs_dir
                / "full_lcal_official_bounded_repair_union_s3_off6_9_delta1_8_susp16_a6000_control_20260528_111111"
                / "summary.json",
                {
                    "num_samples": 1033,
                    "pass_rate": 724 / 1033,
                    "oracle_bucket_pass_rate": {
                        "<=8": 0.80,
                        "9-12": 0.60,
                        "13-16": 0.50,
                        "17-24": 0.20,
                        "25+": 0.10,
                    },
                },
            )
            write_json(
                outputs_dir
                / "full_lcal_official_bounded_repair_union_midcons_off11_13_d3_7_r08_a6000_20260528_222222"
                / "summary.json",
                {
                    "num_samples": 1033,
                    "pass_rate": 739 / 1033,
                    "oracle_bucket_pass_rate": {
                        "<=8": 0.81,
                        "9-12": 0.61,
                        "13-16": 0.55,
                        "17-24": 0.20,
                        "25+": 0.10,
                    },
                },
            )
            write_json(
                analysis_dir / "midcons_vs_a6000_control" / "summary.json",
                {"common": 1033, "wins": 18, "losses": 2, "net": 16},
            )

            section = build_section(outputs_dir=outputs_dir, analysis_dir=analysis_dir)

        self.assertIn("`a6000_control`", section)
        self.assertIn("`midcons`", section)
        self.assertIn("`724/1033` `70.09%`", section)
        self.assertIn("`739/1033` `71.54%`", section)
        self.assertIn("`+16`", section)
        self.assertIn("`81.00%`", section)


if __name__ == "__main__":
    unittest.main()
