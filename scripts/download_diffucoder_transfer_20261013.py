#!/usr/bin/env python3
"""Bounded CPU-only, resumable download of the exact registered Base revision.

Metadata comes directly from the pinned HF mirror resolver. This machine needs
direct access to that resolver and its configured proxy for the Xet CDN.
Existing partial downloads are preserved. No GPU or head files are touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REVISION = "a2c33054cbe99b9ab7af074f09b1d13943157ff1"
MODEL_ID = "apple/DiffuCoder-7B-Base"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-wall-sec", type=int, default=1200)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("bounded at eight CPU download workers")
    if args.max_wall_sec <= 0:
        raise ValueError("download wall time must be positive")
    deadline = time.monotonic() + args.max_wall_sec
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.mkdir(parents=True, exist_ok=True)
    chunks = args.model_path / ".download_chunks"
    chunks.mkdir(exist_ok=True)
    direct = requests.Session()
    direct.trust_env = False
    response = direct.get(f"https://hf-mirror.com/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true", timeout=45)
    response.raise_for_status()
    metadata = response.json()
    if metadata["sha"] != REVISION:
        raise RuntimeError("unexpected model revision")
    (args.output_dir / "hf_diffucoder_exact_blobs.json").write_text(json.dumps(metadata, indent=2))
    entries = [entry for entry in metadata["siblings"] if entry["rfilename"].endswith(".safetensors")]
    jobs = []
    chunk_bytes = 64 * 1024 * 1024
    for entry in entries:
        destination = args.model_path / entry["rfilename"]
        if destination.exists() and destination.stat().st_size == entry["size"]:
            continue
        for start in range(0, entry["size"], chunk_bytes):
            jobs.append((entry["rfilename"], start, min(start + chunk_bytes, entry["size"]) - 1))

    def download(job):
        name, start, end = job
        path = chunks / f"{name}.{start}-{end}"
        expected = end - start + 1
        if path.exists() and path.stat().st_size == expected:
            return
        last_error = None
        for attempt in range(3):
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError("bounded download session wall time exhausted")
                current = path.stat().st_size if path.exists() else 0
                if current > expected:
                    raise RuntimeError("existing range is larger than requested")
                range_start = start + current
                resolver = requests.Session()
                resolver.trust_env = False
                # Cache-busting avoids an expired signed CDN redirect on retries.
                r = resolver.get(f"https://hf-mirror.com/{MODEL_ID}/resolve/{REVISION}/{name}?download=true&request_time={time.time_ns()}",
                                 allow_redirects=False, timeout=30)
                r.raise_for_status()
                if r.status_code not in (302, 303, 307):
                    raise RuntimeError("resolver did not return a CDN redirect")
                with requests.get(r.headers["Location"], headers={"Range": f"bytes={range_start}-{end}"},
                                  stream=True, timeout=(30, 90)) as data:
                    data.raise_for_status()
                    if data.status_code != 206 or not data.headers.get("Content-Range", "").startswith(f"bytes {range_start}-{end}/"):
                        raise RuntimeError("CDN did not honor the exact requested range")
                    with path.open("ab" if current else "wb") as handle:
                        for block in data.iter_content(1024 * 1024):
                            if time.monotonic() >= deadline:
                                raise TimeoutError("bounded download session wall time exhausted")
                            handle.write(block)
                if path.stat().st_size != expected:
                    raise RuntimeError("downloaded range size mismatch")
                return
            except Exception as exc:
                last_error = f"{type(exc).__name__}: download/range failure on attempt {attempt + 1}"
                if time.monotonic() >= deadline:
                    break
        raise RuntimeError(f"{name} range {start}-{end}: {last_error}")

    started = time.time()
    print(f"Downloading {len(jobs)} ranges using {args.workers} CPU workers; GPU hours=0", flush=True)
    completed = 0
    errors = []
    with ThreadPoolExecutor(args.workers) as executor:
        from concurrent.futures import as_completed
        futures = {executor.submit(download, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                future.result()
                completed += 1
                if completed % 16 == 0:
                    print(f"Completed {completed}/{len(jobs)} ranges", flush=True)
            except Exception as exc:
                errors.append(str(exc))
    for entry in entries:
        destination = args.model_path / entry["rfilename"]
        if destination.exists() and destination.stat().st_size == entry["size"]:
            continue
        required = [chunks / f"{entry['rfilename']}.{start}-{min(start + chunk_bytes, entry['size']) - 1}"
                    for start in range(0, entry["size"], chunk_bytes)]
        if not all(path.exists() and path.stat().st_size == min(chunk_bytes, entry["size"] - int(path.name.split(".")[-1].split("-")[0])) for path in required):
            continue
        assembling = destination.with_suffix(".assembling")
        digest = hashlib.sha256()
        with assembling.open("wb") as target:
            for path in required:
                with path.open("rb") as source:
                    while block := source.read(8 * 1024 * 1024):
                        target.write(block)
                        digest.update(block)
        if assembling.stat().st_size != entry["size"]:
            raise RuntimeError("assembled weight size mismatch")
        if digest.hexdigest() != entry["lfs"]["sha256"]:
            raise RuntimeError("assembled weight differs from exact HF LFS sha256")
        entry["assembled_sha256_verified"] = True
        assembling.rename(destination)
    state = {"id": MODEL_ID, "revision": REVISION, "local_path": str(args.model_path),
             "elapsed_cpu_download_sec": time.time() - started, "gpu_hours": 0,
             "completed_ranges_this_run": completed, "errors": errors,
             "shards": [{"name": e["rfilename"], "expected_size": e["size"],
                         "complete": (args.model_path / e["rfilename"]).exists(), "lfs": e.get("lfs"),
                         "assembled_sha256_verified_this_run": e.get("assembled_sha256_verified", False)} for e in entries]}
    (args.output_dir / "model_download.json").write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
