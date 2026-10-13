"""Download pinned official Base shards, with bounded retries and partial resume."""
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/zero_training_transfer_20261013/b_dream"
SNAPSHOT = Path("/home/shx/.cache/dllm_infilling/zero_training_transfer_20261013/dream_base/6572adb5535263e4d1a337b56942ba48b6dee2a9")


def download(item):
    name = item["rfilename"]
    destination = SNAPSHOT / name
    temporary = destination.with_suffix(destination.suffix + ".part")
    expected = item["lfs"]
    if destination.exists():
        if destination.stat().st_size != expected["size"]:
            raise RuntimeError(f"existing final shard size differs: {name}")
        return {"name": name, "status": "reused", "size": expected["size"]}
    for attempt in range(3):
        try:
            url = f"https://hf-mirror.com/Dream-org/Dream-v0-Base-7B/resolve/{SNAPSHOT.name}/{name}"
            command = ["curl", "--fail", "--location", "--silent", "--show-error", "--noproxy", "*",
                       "--connect-timeout", "20", "--max-time", "180", "--speed-limit", "131072",
                       "--speed-time", "30", "--continue-at", "-", "--output", str(temporary), url]
            # curl enforces wall and low-speed limits even when a response trickles bytes.
            subprocess.run(command, check=True, timeout=190)
            digest = hashlib.sha256()
            with temporary.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 << 20), b""):
                    digest.update(chunk)
            if temporary.stat().st_size != expected["size"] or digest.hexdigest() != expected["sha256"]:
                raise RuntimeError("downloaded shard size or official LFS digest differs")
            temporary.rename(destination)
            result = {"name": name, "status": "verified", **expected}
            print(json.dumps(result), flush=True)
            return result
        except Exception as exc:
            print(json.dumps({"name": name, "attempt": attempt + 1, "error": str(exc)}), flush=True)
            if attempt == 2:
                return {"name": name, "status": "blocked", "error": str(exc)}
            time.sleep(2)


def main():
    metadata = json.loads((OUT / "dream_base_api.json").read_text())
    if metadata["sha"] != SNAPSHOT.name:
        raise RuntimeError("model revision differs")
    items = [row for row in metadata["siblings"] if row["rfilename"].endswith(".safetensors")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(download, items))
    (OUT / "weight_download_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    if any(row["status"] == "blocked" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
