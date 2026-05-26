#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.runner_with_stopping import run_stopping_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-length infilling with global-gap early commit stopping."
    )
    parser.add_argument("--model-path", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-mask-tokens", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--stop-min-step", type=int, default=32)
    parser.add_argument("--stop-gap-threshold", type=float, default=0.50)
    parser.add_argument("--stop-top1-threshold", type=float, default=0.70)
    parser.add_argument("--stop-max-remaining-mask-ratio", type=float, default=0.50)

    parser.add_argument("--output-dir", type=str, default="outputs_clean")
    parser.add_argument("--experiment-name", type=str, default="fixed_stop_clean")
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

    cfg.decode.mask_length_source = "fixed"
    cfg.decode.fixed_mask_length = args.num_mask_tokens
    cfg.decode.total_steps = args.total_steps
    cfg.decode.seed = args.seed

    cfg.decode.stop_min_step = args.stop_min_step
    cfg.decode.stop_gap_threshold = args.stop_gap_threshold
    cfg.decode.stop_top1_threshold = args.stop_top1_threshold
    cfg.decode.stop_max_remaining_mask_ratio = args.stop_max_remaining_mask_ratio

    cfg.decode.save_step_traces = args.save_step_traces
    cfg.decode.save_full_text_per_step = args.save_full_text_per_step

    cfg.logging.output_dir = args.output_dir
    cfg.logging.experiment_name = args.experiment_name

    output = run_stopping_experiment(cfg)

    print("===== Fixed-Length + Stopping Summary =====")
    for key, value in output["summary"].items():
        print(f"{key}: {value}")
    print(f"\nRun directory: {output['run_dir']}")


if __name__ == "__main__":
    main()