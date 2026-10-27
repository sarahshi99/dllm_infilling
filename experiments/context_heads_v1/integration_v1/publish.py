"""Publish additive interop evidence; preserve first-batch B_done and other ready."""
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..contract import CONTRACT_ID, SCHEMA_VERSION
from ..publish_ready import BRANCH, sha_bytes
from .verify_cpu import check_original_payload, sha

MODULE = Path(__file__).resolve().parent
ROOT = MODULE.parents[2]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def files():
    return sorted(p for p in MODULE.rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and p.name != "manifest.json")


def build_manifest():
    value = dict(contract_id=CONTRACT_ID, schema_version=SCHEMA_VERSION, branch=BRANCH,
                 base_commit="ec8b159ce276c7a919a924c1b6991ffb64a5b8bf",
                 supplements="B_done.json", scope="additive CPU A/B keyword/sampler interop only",
                 data=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in files()])
    (MODULE/"manifest.json").write_text(json.dumps(value, indent=2)+"\n")


def publish():
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("wrong B producer branch")
    if git("status", "--porcelain", "--", "experiments/context_heads_v1/integration_v1",
           "docs/paper_agent/activity_ledger.zh.md"):
        raise RuntimeError("commit the supplementary payload first")
    check_original_payload()
    commit = git("rev-parse", "HEAD")
    command = ["git", "-c", "core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12 "
               "-o ConnectionAttempts=1 -o ServerAliveInterval=5 -o ServerAliveCountMax=2",
               "ls-remote", "origin", "refs/heads/"+BRANCH]
    remote = subprocess.check_output(command, cwd=ROOT, text=True, timeout=35).split()
    if not remote or remote[0] != commit:
        raise RuntimeError("remote B branch HEAD must match the supplementary commit")
    manifest = MODULE/"manifest.json"
    value = json.loads(manifest.read_text())
    if value["contract_id"] != CONTRACT_ID or value["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError("supplement manifest contract mismatch")
    if {str(p.relative_to(ROOT)) for p in files()} != {x["path"] for x in value["data"]}:
        raise RuntimeError("supplement payload list incomplete")
    for entry in value["data"]:
        if sha(ROOT/entry["path"]) != entry["sha256"]:
            raise RuntimeError("supplement payload hash mismatch")
        committed = subprocess.check_output(["git", "show", commit+":"+entry["path"]], cwd=ROOT)
        if sha_bytes(committed) != entry["sha256"]:
            raise RuntimeError("supplement payload differs from commit")
    evidence = json.loads((MODULE/"artifacts/verification.json").read_text())
    if evidence["status"] != "passed" or evidence["gpu_used"] or evidence["training_started"]:
        raise RuntimeError("supplement CPU verification incomplete")
    common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir"))/"research_runs/dreamon_context_20261015_v1"
    if sha(common/"B_done.json") != evidence["B_done_sha256_unchanged"]:
        raise RuntimeError("first-batch B_done changed since the audit")
    ready = dict(contract_id=CONTRACT_ID, schema_version=SCHEMA_VERSION,
                 status="B_integration_done_CPU_only", supplements="B_done.json",
                 commit=commit, branch=BRANCH, source_commit=commit, producer_branch=BRANCH,
                 manifest_path=str(manifest), manifest_absolute_path=str(manifest),
                 manifest_sha256=sha(manifest), created_at=datetime.now(timezone.utc).isoformat())
    destination = common/"B_integration_done.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous.get("commit") == commit and previous.get("manifest_sha256") == ready["manifest_sha256"]:
            print(destination)
            return
        raise FileExistsError("preserving existing supplementary ready")
    temporary = common/(".B_integration_done."+str(os.getpid())+".tmp")
    with temporary.open("x") as stream:
        json.dump(ready, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, destination)
    temporary.unlink()
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-manifest", action="store_true")
    arguments = parser.parse_args()
    build_manifest() if arguments.build_manifest else publish()
