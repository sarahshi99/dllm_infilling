from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clean_scripts import resume_lcal_a6000_policy as resume


def minimal_config(*, experiment_name: str, v2: bool = True) -> dict:
    repair = {
        "official_eval_max_s3_len": 12,
        "repair_max_s3_len": 5,
        "repair_min_official_len": 6,
        "repair_max_official_len": 9,
        "repair_min_delta": 1,
        "repair_max_delta": 8,
        "suspicion_max_s3_len": 5,
        "suspicion_min_official_len": 16,
        "suspicion_max_official_len": 64,
        "suspicion_min_delta": 1,
        "mid_rescue_max_s3_len": 12,
        "mid_rescue_min_official_len": 11,
        "mid_rescue_max_official_len": 13,
        "mid_rescue_min_delta": 3,
        "mid_rescue_max_delta": 7,
        "mid_rescue_min_long_ratio": 0.8,
        "mid_rescue_source": "base",
    }
    if v2:
        repair.update(
            {
                "mid_rescue_min_support_count": 2,
                "mid_rescue_best_long_lens_csv": "13,14,15,16",
                "mid_rescue_veto_s3_le": 5,
                "mid_rescue_veto_official_len": 13,
                "true_long_max_s3_len": 12,
                "true_long_min_official_len": 17,
                "true_long_max_official_len": 64,
                "true_long_min_delta": 8,
                "true_long_min_long_ratio": 0.85,
                "true_long_min_raw_ratio": 0.85,
                "true_long_min_support_count": 2,
                "true_long_best_long_lens_csv": "16,20,24,28,32,40",
                "true_long_source": "any",
            }
        )
    return {
        "model": {"model_path": "GSAI-ML/LLaDA-8B-Base", "torch_dtype": "bfloat16"},
        "data": {
            "split": "test",
            "dataset_subset": "HumanEval-SingleLineInfilling",
            "max_samples": None,
        },
        "decode": {
            "total_steps": 64,
            "seed": 42,
            "save_step_traces": False,
            "save_full_text_per_step": False,
            "cal_lite_probe_lengths_csv": "3,4,5",
            "cal_lite_tie_break": "shorter",
            "cal_lite_score_mode": "length_power",
            "cal_lite_length_alpha": 0.06,
            "lcas_policy": "lcas_v3b",
        },
        "logging": {"output_dir": "outputs_clean", "experiment_name": experiment_name},
        "lcal_official_bounded_repair": {
            "lcal": {
                "base_probe_lengths_csv": "3,4,5",
                "base_alpha": 0.06,
                "weak_probe_lengths_csv": "13,14",
                "strong_probe_lengths_csv": "13,14,20",
                "long_alpha": 0.1,
                "strong_min_len": 13,
                "weak_min_base_len": 8,
                "weak_max_base_len": 12,
                "ratio_trigger_threshold": 0.97,
                "long_score_floor": 0.55,
                "raw_ratio_threshold": 0.97,
                "support_count_threshold": 2,
                "cap_base_le8": 14,
                "cap_base_9_12": 16,
                "short_safe_policy": "s3",
                "correction_selection_rule": "shortest_supported",
                "shortest_supported_ratio": 0.985,
                "tie_break": "shorter",
                "score_mode": "length_power",
            },
            "official": {
                "initial_length": 8,
                "span": 1,
                "max_length": 64,
                "dstep": 4,
                "use_bias": True,
                "bias_params": [1.0, 1.77, 0.56, 0.06, 0.24],
            },
            "repair": repair,
        },
        "baseline_results": "outputs_clean/baseline/results.jsonl",
    }


class LcalA6000ResumeTest(unittest.TestCase):
    def test_deduplicates_existing_results_by_task_id_with_last_row_winning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            rows = [
                {"task_id": "task/1", "metrics": {"passed": False}},
                {"task_id": "task/2", "metrics": {"passed": True}},
                {"task_id": "task/1", "metrics": {"passed": True}},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            loaded = resume.load_existing_results(path)

        self.assertEqual([row["task_id"] for row in loaded], ["task/1", "task/2"])
        self.assertTrue(loaded[0]["metrics"]["passed"])

    def test_builds_v2_settings_from_saved_config(self) -> None:
        module = resume.select_runner_module(minimal_config(experiment_name="combined", v2=True))
        cfg, settings, baseline = resume.build_cfg_settings_from_config(
            minimal_config(experiment_name="combined", v2=True),
            module,
        )

        self.assertEqual(cfg.decode.total_steps, 64)
        self.assertEqual(cfg.decode.lcas_policy, "lcas_v3b")
        self.assertEqual(settings.mid_rescue_min_support_count, 2)
        self.assertEqual(settings.true_long_min_official_len, 17)
        self.assertEqual(baseline, "outputs_clean/baseline/results.jsonl")

    def test_selects_legacy_runner_when_config_has_no_v2_repair_keys(self) -> None:
        module = resume.select_runner_module(minimal_config(experiment_name="midcons", v2=False))
        _cfg, settings, _baseline = resume.build_cfg_settings_from_config(
            minimal_config(experiment_name="midcons", v2=False),
            module,
        )

        self.assertFalse(hasattr(settings, "true_long_min_official_len"))
        self.assertEqual(settings.mid_rescue_source, "base")


if __name__ == "__main__":
    unittest.main()
