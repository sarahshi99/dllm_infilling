#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WRAPPER = ROOT / "clean_scripts" / "run_lcas_v3_model_sweep.py"
DEFAULT_LLMXY = "/home/shx/miniconda3/envs/llmxy/bin/python"
DEFAULT_HUMAN_EVAL = "/home/shx/projects/dllm_llada_infilling/human-eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Queue LCAS-v3 model-sweep jobs and launch them when GPUs have enough free memory."
    )
    parser.add_argument("--models", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--experiment-prefix", type=str, default="full_lcas_v3")
    parser.add_argument("--wrapper", type=str, default=str(DEFAULT_WRAPPER))
    parser.add_argument("--cuda-visible-devices", type=str, default="2,3")
    parser.add_argument("--min-free-mb", type=int, default=24000)
    parser.add_argument("--poll-sec", type=int, default=180)
    parser.add_argument("--python-bin", type=str, default=DEFAULT_LLMXY)
    parser.add_argument("--append-site-packages", type=str, default=DEFAULT_HUMAN_EVAL)
    parser.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--proxy-url", type=str, default="http://127.0.0.1:7890")
    parser.add_argument("--disable-hf-xet", action="store_true", default=True)
    parser.add_argument("--stream-mode", type=str, default="summary", choices=["full", "summary", "none"])
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_gpu_ids(csv_text: str) -> List[int]:
    ids = []
    for raw in csv_text.split(","):
        raw = raw.strip()
        if raw:
            ids.append(int(raw))
    if not ids:
        raise ValueError("--cuda-visible-devices must contain at least one GPU id")
    return ids


def query_free_memory() -> Dict[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free_by_gpu: Dict[int, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        idx_text, free_text = [part.strip() for part in line.split(",", 1)]
        free_by_gpu[int(idx_text)] = int(free_text)
    return free_by_gpu


def wait_for_memory(gpu_ids: List[int], min_free_mb: int, poll_sec: int, status_path: Path) -> None:
    while True:
        free_by_gpu = query_free_memory()
        selected = {gpu_id: free_by_gpu.get(gpu_id, 0) for gpu_id in gpu_ids}
        enough = all(value >= min_free_mb for value in selected.values())
        write_status(
            status_path,
            {
                "event": "memory_check",
                "free_mb": selected,
                "min_free_mb": min_free_mb,
                "enough": enough,
            },
        )
        print(
            f"[{now()}] memory_check free_mb={selected} min_free_mb={min_free_mb} enough={enough}",
            flush=True,
        )
        if enough:
            return
        time.sleep(max(5, poll_sec))


def model_items(models_csv: str) -> List[str]:
    return [item.strip() for item in models_csv.split(",") if item.strip()]


def build_command(args: argparse.Namespace, model: str, log_dir: Path) -> List[str]:
    cmd = [
        args.python_bin,
        str(Path(args.wrapper)),
        "--models",
        model,
        "--python-bin",
        args.python_bin,
        "--append-site-packages",
        args.append_site_packages,
        "--cuda-visible-devices",
        args.cuda_visible_devices,
        "--output-dir",
        args.output_dir,
        "--experiment-prefix",
        args.experiment_prefix,
        "--stream-mode",
        args.stream_mode,
        "--progress-every",
        str(args.progress_every),
        "--child-log-dir",
        str(log_dir),
    ]
    if args.hf_endpoint:
        cmd.extend(["--hf-endpoint", args.hf_endpoint])
    if args.proxy_url:
        cmd.extend(["--proxy-url", args.proxy_url])
    if args.disable_hf_xet:
        cmd.append("--disable-hf-xet")
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    return cmd


def build_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint
    if args.proxy_url:
        env["http_proxy"] = args.proxy_url
        env["https_proxy"] = args.proxy_url
        env["HTTP_PROXY"] = args.proxy_url
        env["HTTPS_PROXY"] = args.proxy_url
    if args.disable_hf_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    return env


def write_status(path: Path, payload: Dict[str, object]) -> None:
    payload = {"time": now(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_model(args: argparse.Namespace, model: str, log_dir: Path, status_path: Path) -> int:
    cmd = build_command(args, model, log_dir)
    env = build_env(args)
    safe_model = model.replace("/", "_")
    wrapper_log = log_dir / f"queue_{safe_model}.log"

    write_status(status_path, {"event": "start_model", "model": model, "command": cmd})
    print(f"[{now()}] start_model model={model}", flush=True)
    print(f"[{now()}] log={wrapper_log}", flush=True)

    with wrapper_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        returncode = process.wait()

    write_status(status_path, {"event": "finish_model", "model": model, "returncode": returncode})
    print(f"[{now()}] finish_model model={model} returncode={returncode}", flush=True)
    return returncode


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    log_dir = output_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "gpu_queue_status.jsonl"
    gpu_ids = parse_gpu_ids(args.cuda_visible_devices)

    write_status(
        status_path,
        {
            "event": "queue_start",
            "models": model_items(args.models),
            "cuda_visible_devices": args.cuda_visible_devices,
            "min_free_mb": args.min_free_mb,
            "poll_sec": args.poll_sec,
        },
    )

    for model in model_items(args.models):
        wait_for_memory(gpu_ids, args.min_free_mb, args.poll_sec, status_path)
        returncode = run_model(args, model, log_dir, status_path)
        if returncode != 0 and args.stop_on_error:
            raise SystemExit(returncode)

    write_status(status_path, {"event": "queue_done"})
    print(f"[{now()}] queue_done", flush=True)


if __name__ == "__main__":
    main()
