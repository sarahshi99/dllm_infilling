from __future__ import annotations

import unittest

from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.length_probe import adjust_length_probe_score


class LengthProbeProportionalScoreTest(unittest.TestCase):
    def test_proportional_score_matches_length_power_below_reference(self) -> None:
        cfg = ExperimentConfig()
        cfg.decode.cal_lite_score_mode = "length_power_proportional"
        cfg.decode.cal_lite_length_alpha = 0.06
        cfg.decode.cal_lite_length_prop_beta = 0.04
        cfg.decode.cal_lite_length_prop_ref_length = 12.0

        proportional = adjust_length_probe_score(raw_score=0.8, mask_length=8, cfg=cfg)

        cfg.decode.cal_lite_score_mode = "length_power"
        baseline = adjust_length_probe_score(raw_score=0.8, mask_length=8, cfg=cfg)
        self.assertAlmostEqual(proportional, baseline)

    def test_proportional_score_rewards_lengths_above_reference(self) -> None:
        cfg = ExperimentConfig()
        cfg.decode.cal_lite_score_mode = "length_power_proportional"
        cfg.decode.cal_lite_length_alpha = 0.06
        cfg.decode.cal_lite_length_prop_beta = 0.04
        cfg.decode.cal_lite_length_prop_ref_length = 12.0

        proportional = adjust_length_probe_score(raw_score=0.8, mask_length=24, cfg=cfg)

        cfg.decode.cal_lite_score_mode = "length_power"
        baseline = adjust_length_probe_score(raw_score=0.8, mask_length=24, cfg=cfg)
        self.assertGreater(proportional, baseline)

    def test_proportional_cap_limits_extra_reward(self) -> None:
        capped_cfg = ExperimentConfig()
        capped_cfg.decode.cal_lite_score_mode = "length_power_proportional"
        capped_cfg.decode.cal_lite_length_alpha = 0.06
        capped_cfg.decode.cal_lite_length_prop_beta = 0.04
        capped_cfg.decode.cal_lite_length_prop_ref_length = 12.0
        capped_cfg.decode.cal_lite_length_prop_cap_length = 24.0

        uncapped_cfg = ExperimentConfig()
        uncapped_cfg.decode.cal_lite_score_mode = "length_power_proportional"
        uncapped_cfg.decode.cal_lite_length_alpha = 0.06
        uncapped_cfg.decode.cal_lite_length_prop_beta = 0.04
        uncapped_cfg.decode.cal_lite_length_prop_ref_length = 12.0
        uncapped_cfg.decode.cal_lite_length_prop_cap_length = None

        capped = adjust_length_probe_score(raw_score=0.8, mask_length=48, cfg=capped_cfg)
        uncapped = adjust_length_probe_score(raw_score=0.8, mask_length=48, cfg=uncapped_cfg)
        self.assertLess(capped, uncapped)


if __name__ == "__main__":
    unittest.main()
