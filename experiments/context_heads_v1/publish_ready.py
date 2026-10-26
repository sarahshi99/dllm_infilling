"""Publish only B_done, after commit/push verification; never mutate another ready.

The user explicitly requests manifest hashing for this cross-worktree handoff.
Manifest intentionally has no self-referential commit/hash; B_done supplies them.
"""
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .contract import CONTRACT_ID, SCHEMA_VERSION

BRANCH = "codex/dreamon-context-heads-20261015-v1"
BASE = "d39a51d494efc2a1d1661eb28622b18b3b741fa6"
MODULE = Path(__file__).resolve().parent
ROOT = MODULE.parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def payload_files():
    return sorted(p for p in MODULE.rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.name != "manifest.json")


def build_manifest():
    manifest = dict(contract_id=CONTRACT_ID, schema_version=SCHEMA_VERSION,
                    branch=BRANCH, base_commit=BASE, status="CPU-only implementation complete",
                    scope="seven heads; ordinary-support reference executors; no training/GPU/backbone/answers",
                    data=[dict(path=str(p.relative_to(ROOT)), sha256=sha_bytes(p.read_bytes()))
                          for p in payload_files()])
    (MODULE/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def publish():
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("wrong producer branch")
    if git("status", "--porcelain", "--", "experiments/context_heads_v1", "docs/paper_agent/activity_ledger.zh.md"):
        raise RuntimeError("B payload/log must be committed before publishing")
    commit = git("rev-parse", "HEAD")
    remote = git("ls-remote", "origin", "refs/heads/"+BRANCH).split()
    if not remote or remote[0] != commit:
        raise RuntimeError("remote HEAD must match local commit")
    manifest_path = MODULE/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["contract_id"] != CONTRACT_ID or manifest["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("manifest contract mismatch")
    actual_paths = {str(p.relative_to(ROOT)) for p in payload_files()}
    if actual_paths != {entry["path"] for entry in manifest["data"]}:
        raise RuntimeError("manifest payload list incomplete")
    for entry in manifest["data"]:
        p = ROOT/entry["path"]
        if sha_bytes(p.read_bytes()) != entry["sha256"]:
            raise RuntimeError("manifest payload hash mismatch: "+entry["path"])
        committed = subprocess.check_output(["git", "show", commit+":"+entry["path"]], cwd=ROOT)
        if sha_bytes(committed) != entry["sha256"]:
            raise RuntimeError("manifest payload differs from commit")
    verification = json.loads((MODULE/"artifacts/verification.json").read_text())
    if verification["status"] != "passed" or verification["gpu_used"] or verification["training_started"]:
        raise RuntimeError("CPU verification not complete")
    digest = sha_bytes(manifest_path.read_bytes())
    ready = dict(contract_id=CONTRACT_ID, schema_version=SCHEMA_VERSION,
                 status="B_done_CPU_only", commit=commit, branch=BRANCH,
                 manifest_path=str(manifest_path), manifest_sha256=digest,
                 # Aliases for A's existing ready-reader naming convention.
                 source_commit=commit, producer_branch=BRANCH,
                 manifest_absolute_path=str(manifest_path),
                 created_at=datetime.now(timezone.utc).isoformat())
    common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir"))
    out = common/"research_runs/dreamon_context_20261015_v1"
    out.mkdir(parents=True, exist_ok=True)
    dest = out/"B_done.json"
    if dest.exists():
        existing = json.loads(dest.read_text())
        if existing.get("commit") == commit and existing.get("manifest_sha256") == digest:
            print(str(dest))
            return
        raise FileExistsError("preserving existing B_done; no ready/lock overwritten")
    # Write a complete private file, then link atomically without overwriting dest.
    temporary = out/(".B_done."+str(os.getpid())+".tmp")
    with temporary.open("x") as f:
        json.dump(ready, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.link(temporary, dest)
    temporary.unlink()  # only our own temporary link, never another ready/lock
    print(str(dest))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-manifest", action="store_true")
    args = parser.parse_args()
    build_manifest() if args.build_manifest else publish()


if __name__ == "__main__":
    main()
