#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.runner_lcas_v3 import run_lcas_v3_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CAL-lite length-adaptive infilling with LCAS-v3 stopping policies."
    )
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--probe-lengths",
        type=str,
        default="3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24",
    )
    parser.add_argument("--tie-break", type=str, default="shorter", choices=["shorter", "longer"])
    parser.add_argument("--score-mode", type=str, default="length_power", choices=["raw", "length_power"])
    parser.add_argument("--length-alpha", type=float, default=0.06)

    parser.add_argument("--lcas-policy", type=str, default="lcas_v3a", choices=["lcas_v3a", "lcas_v3b"])

    parser.add_argument("--output-dir", type=str, default="outputs_clean")
    parser.add_argument("--experiment-name", type=str, default="cal_lite_lcas_v3_clean")
    parser.add_argument("--save-step-traces", action="store_true")
    parser.add_argument("--save-full-text-per-step", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ExperimentConfig()
    cfg.model.model_path = args.model_path
    cfg.data.split = args.split
    cfg.data.dataset_subset = args.dataset_subset
    cfg.data.max_samples = args.max_samples

    cfg.decode.mask_length_source = "cal_lite"
    cfg.decode.total_steps = args.total_steps
    cfg.decode.seed = args.seed

    cfg.decode.cal_lite_probe_lengths_csv = args.probe_lengths
    cfg.decode.cal_lite_tie_break = args.tie_break
    cfg.decode.cal_lite_score_mode = args.score_mode
    cfg.decode.cal_lite_length_alpha = args.length_alpha

    cfg.decode.lcas_policy = args.lcas_policy

    cfg.decode.save_step_traces = args.save_step_traces
    cfg.decode.save_full_text_per_step = args.save_full_text_per_step

    cfg.logging.output_dir = args.output_dir
    cfg.logging.experiment_name = args.experiment_name

    output = run_lcas_v3_experiment(cfg)

    print("===== LCAS-v3 Summary =====")
    for key, value in output["summary"].items():
        print(f"{key}: {value}")
    print(f"\nRun directory: {output['run_dir']}")


if __name__ == "__main__":
    main()