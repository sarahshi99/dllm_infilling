from __future__ import annotations

import unittest
from pathlib import Path


class A6000LauncherTest(unittest.TestCase):
    def test_baseline_launcher_allows_gpu_ids_override(self) -> None:
        script = Path("clean_scripts/run_lcal_a6000_baselines.sh").read_text(encoding="utf-8")

        self.assertIn("GPU_IDS=${GPU_IDS:-0,1}", script)
        self.assertIn("export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_IDS}", script)
        self.assertNotIn("export CUDA_VISIBLE_DEVICES=0,1", script)

    def test_offline_candidate_launcher_allows_gpu_ids_override(self) -> None:
        script = Path("clean_scripts/run_lcal_a6000_candidates_offline.sh").read_text(encoding="utf-8")

        self.assertIn("GPU_IDS=${GPU_IDS:-0,1}", script)
        self.assertIn("export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU_IDS}", script)
        self.assertNotIn("export CUDA_VISIBLE_DEVICES=0,1", script)

    def test_wait_script_passes_checked_gpu_ids_to_baseline_launcher(self) -> None:
        script = Path("clean_scripts/wait_and_run_lcal_a6000_baselines.sh").read_text(encoding="utf-8")

        self.assertIn("GPU_IDS=${GPU_IDS:-0,1}", script)
        self.assertIn("CUDA_VISIBLE_DEVICES=\"$GPU_IDS\"", script)
        self.assertIn("exec clean_scripts/run_lcal_a6000_baselines.sh", script)


if __name__ == "__main__":
    unittest.main()
