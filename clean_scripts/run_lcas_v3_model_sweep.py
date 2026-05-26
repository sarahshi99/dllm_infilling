#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = ROOT / "clean_scripts" / "run_cal_lite_lcas_v3.py"
COMPAT_SITE = ROOT / "clean_scripts" / "compat_site"

MODEL_ALIASES: Dict[str, str] = {
    "llada-base": "GSAI-ML/LLaDA-8B-Base",
    "llada-instruct": "GSAI-ML/LLaDA-8B-Instruct",
    "dream-base": "Dream-org/Dream-v0-Base-7B",
    "dream-instruct": "Dream-org/Dream-v0-Instruct-7B",
    "dream-coder-base": "Dream-org/Dream-Coder-v0-Base-7B",
    "dream-coder-instruct": "Dream-org/Dream-Coder-v0-Instruct-7B",
    "dreamon": "Dream-org/DreamOn-v0-7B",
}

RUN_DIR_RE = re.compile(r"Run directory:\s*(?P<path>\S+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current clean LCAS-v3 entrypoint across multiple masked-diffusion "
            "models without modifying existing experiment files."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        default="llada-base,llada-instruct,dream-coder-base,dream-coder-instruct",
        help=(
            "Comma-separated model aliases or Hugging Face paths. Known aliases: "
            + ",".join(sorted(MODEL_ALIASES))
        ),
    )
    parser.add_argument("--runner", type=str, default=str(DEFAULT_RUNNER))
    parser.add_argument("--cuda-visible-devices", type=str, default="2,3")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument(
        "--append-site-packages",
        type=str,
        default=None,
        help=(
            "Optional os.pathsep-separated site-packages paths appended inside the child "
            "process after its own environment packages. Useful for reusing datasets/"
            "human_eval from dllm_env while running a newer transformers env."
        ),
    )
    parser.add_argument("--hf-endpoint", type=str, default=os.environ.get("HF_ENDPOINT"))
    parser.add_argument("--proxy-url", type=str, default=None)
    parser.add_argument("--disable-hf-xet", action="store_true")
    parser.add_argument("--unset-proxy-vars", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--stream-mode", type=str, default="full", choices=["full", "summary", "none"])
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--child-log-dir", type=str, default=None)

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
    parser.add_argument("--experiment-prefix", type=str, default="model_sweep_lcas_v3")
    parser.add_argument("--save-step-traces", action="store_true")
    parser.add_argument("--save-full-text-per-step", action="store_true")
    parser.add_argument(
        "--aggregate-path",
        type=str,
        default=None,
        help="Optional JSON path for aggregate sweep status. Defaults under output-dir.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    text = text.strip().replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._-").lower() or "model"


def resolve_models(models_csv: str) -> List[Dict[str, str]]:
    specs: List[Dict[str, str]] = []
    seen = set()
    for raw in models_csv.split(","):
        key = raw.strip()
        if not key:
            continue
        model_path = MODEL_ALIASES.get(key, key)
        alias = key if key in MODEL_ALIASES else slugify(model_path)
        if model_path in seen:
            continue
        seen.add(model_path)
        specs.append({"alias": alias, "model_path": model_path})
    if not specs:
        raise ValueError("--models did not contain any valid model alias or path")
    return specs


def build_command(args: argparse.Namespace, model_alias: str, model_path: str) -> List[str]:
    experiment_name = f"{args.experiment_prefix}_{slugify(model_alias)}"
    cmd = [
        args.python_bin,
        str(Path(args.runner)),
        "--model-path",
        model_path,
        "--split",
        args.split,
        "--dataset-subset",
        args.dataset_subset,
        "--total-steps",
        str(args.total_steps),
        "--seed",
        str(args.seed),
        "--probe-lengths",
        args.probe_lengths,
        "--tie-break",
        args.tie_break,
        "--score-mode",
        args.score_mode,
        "--length-alpha",
        str(args.length_alpha),
        "--lcas-policy",
        args.lcas_policy,
        "--output-dir",
        args.output_dir,
        "--experiment-name",
        experiment_name,
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.save_step_traces:
        cmd.append("--save-step-traces")
    if args.save_full_text_per_step:
        cmd.append("--save-full-text-per-step")
    return cmd


def should_stream_line(line: str, mode: str, progress_every: int) -> bool:
    if mode == "full":
        return True
    if mode == "none":
        return False

    stripped = line.strip()
    if not stripped:
        return False
    important_prefixes = (
        "Starting ",
        "experiment_name =",
        "model_path       =",
        "max_samples      =",
        "Run directory:",
        "LCAS-v3 experiment finished",
        "===== LCAS-v3 Summary =====",
        "num_samples:",
        "pass_rate:",
        "stop_rate:",
        "avg_effective_steps:",
        "Aggregate summary:",
    )
    if stripped.startswith(important_prefixes):
        return True
    if stripped.startswith("=" * 20) or stripped.startswith("-" * 20):
        return True

    if stripped.startswith("[") and "|" in stripped and ("PASS" in stripped or "FAIL" in stripped):
        match = re.match(r"\[(?P<idx>\d+)/(?P<total>\d+)\]", stripped)
        if not match:
            return True
        idx = int(match.group("idx"))
        total = int(match.group("total"))
        every = max(1, int(progress_every))
        return idx == 1 or idx == total or idx % every == 0

    return False


def extract_run_dir(stdout: str) -> Optional[str]:
    matches = RUN_DIR_RE.findall(stdout)
    return matches[-1] if matches else None


def load_summary(run_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    if not run_dir:
        return None
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_aggregate_path(output_dir: str, experiment_prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(output_dir) / f"{experiment_prefix}_aggregate_{timestamp}.json"


def build_child_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    if args.unset_proxy_vars:
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            env.pop(key, None)
    if args.proxy_url:
        env["http_proxy"] = args.proxy_url
        env["https_proxy"] = args.proxy_url
        env["HTTP_PROXY"] = args.proxy_url
        env["HTTPS_PROXY"] = args.proxy_url
    if args.hf_endpoint:
        env["HF_ENDPOINT"] = args.hf_endpoint
    if args.disable_hf_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    if args.append_site_packages:
        env["DLLM_SWEEP_APPEND_SITE_PACKAGES"] = args.append_site_packages
        existing_pythonpath = env.get("PYTHONPATH")
        pythonpath_parts = [str(COMPAT_SITE)]
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run_one(args: argparse.Namespace, model_spec: Dict[str, str]) -> Dict[str, Any]:
    cmd = build_command(args, model_spec["alias"], model_spec["model_path"])
    env = build_child_env(args)

    printable = " ".join(cmd)
    print("=" * 80, flush=True)
    print(f"model_alias = {model_spec['alias']}", flush=True)
    print(f"model_path  = {model_spec['model_path']}", flush=True)
    print(f"gpus        = {args.cuda_visible_devices}", flush=True)
    print(f"hf_endpoint = {env.get('HF_ENDPOINT')}", flush=True)
    print(f"proxy_url   = {args.proxy_url}", flush=True)
    print(f"disable_xet = {env.get('HF_HUB_DISABLE_XET')}", flush=True)
    print(f"append_sp   = {args.append_site_packages}", flush=True)
    print(f"command     = CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} {printable}", flush=True)

    experiment_name = f"{args.experiment_prefix}_{slugify(model_spec['alias'])}"
    log_dir = Path(args.child_log_dir) if args.child_log_dir else Path(args.output_dir) / "_logs"
    log_path = log_dir / f"{experiment_name}.log"

    if args.dry_run:
        return {
            "model_alias": model_spec["alias"],
            "model_path": model_spec["model_path"],
            "status": "dry_run",
            "command": cmd,
            "cuda_visible_devices": args.cuda_visible_devices,
            "log_path": str(log_path),
        }

    log_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    stdout_parts: List[str] = []
    try:
        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8") as log_handle:
            for line in process.stdout:
                log_handle.write(line)
                log_handle.flush()
                if should_stream_line(line, args.stream_mode, args.progress_every):
                    print(line, end="", flush=True)
                stdout_parts.append(line)
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise

    stdout = "".join(stdout_parts)
    run_dir = extract_run_dir(stdout)
    summary = load_summary(run_dir)
    status = "ok" if returncode == 0 else "failed"
    return {
        "model_alias": model_spec["alias"],
        "model_path": model_spec["model_path"],
        "status": status,
        "returncode": returncode,
        "run_dir": run_dir,
        "summary": summary,
        "command": cmd,
        "cuda_visible_devices": args.cuda_visible_devices,
        "log_path": str(log_path),
    }


def main() -> None:
    args = parse_args()
    runner = Path(args.runner)
    if not runner.exists():
        raise FileNotFoundError(f"Runner not found: {runner}")

    model_specs = resolve_models(args.models)
    aggregate: Dict[str, Any] = {
        "runner": str(runner),
        "cuda_visible_devices": args.cuda_visible_devices,
        "models": model_specs,
        "args": vars(args),
        "determinism_note": (
            "Current LCAS-v3 wrapper uses deterministic argmax/remask decoding. "
            "The seed is fixed for reproducibility and environment initialization; "
            "multi-seed mean/std is not expected to measure decoding stochasticity."
        ),
        "results": [],
    }

    for model_spec in model_specs:
        result = run_one(args, model_spec)
        aggregate["results"].append(result)
        if result["status"] == "failed" and not args.continue_on_error:
            break

    aggregate_path = Path(args.aggregate_path) if args.aggregate_path else default_aggregate_path(
        args.output_dir,
        args.experiment_prefix,
    )
    if not args.dry_run:
        aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        with aggregate_path.open("w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2, ensure_ascii=False)
        print(f"Aggregate summary: {aggregate_path}", flush=True)
    else:
        print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)

    failed = [item for item in aggregate["results"] if item["status"] == "failed"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
