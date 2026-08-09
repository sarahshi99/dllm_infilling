from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
SHARED_ROOT = Path("/home/shx/projects/dllm_infilling")
DATA_PATH = SHARED_ROOT / "git_workspace/data/HumanEval-MultiLineInfilling.jsonl"
DREAMON_ROOT = SHARED_ROOT / "DreamOn"
MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub/models--Dream-org--DreamOn-v0-7B"
    / "snapshots/8ccc74750e43177327f29dab9e91882ba759e194"
)
PYTHON = SHARED_ROOT / ".venvs/dreamon-repro/bin/python"

MODEL_REVISION = "8ccc74750e43177327f29dab9e91882ba759e194"
DREAMON_COMMIT = "8a0a54918412eda9402a327646f7f067f7160ec8"
EVALUATOR_COMMIT = "88062ff9859c875d04db115b698ed4b0f0395170"

INITIAL_MASK_COUNT = 64
MAX_NEW_TOKENS = 64
NUMBER_TRANSFER_TOKENS = 1
MAX_FORWARDS = 256
MAX_CONTEXT_TOKENS = 2048
TEMPERATURE = 0.2
TOP_P = 0.9
TOP_K = None
ALGORITHM = "entropy"
ALGORITHM_TEMPERATURE = 0.0
WIDTHS = (1, 4, 8, 16, None)
WIDTH_LABELS = {1: "1", 4: "4", 8: "8", 16: "16", None: "inf"}
PILOT_GATE_PASSES = 16
PILOT_ROWS = 30
FIXED_FULL_ROWS = 1000
EVALUATION_TIMEOUT_SECONDS = 3.0
EVALUATION_WORKERS = 8

PILOT_MANIFEST_PATH = EXPERIMENT_DIR / "pilot30_manifest.jsonl"
PILOT_METADATA_PATH = EXPERIMENT_DIR / "pilot30_manifest.meta.json"
PILOT_CHECKSUM_PATH = EXPERIMENT_DIR / "pilot30_manifest.sha256"
FIXED_MANIFEST_PATH = EXPERIMENT_DIR / "fixed_full_1000_manifest.jsonl"
FIXED_METADATA_PATH = EXPERIMENT_DIR / "fixed_full_1000_manifest.meta.json"
FIXED_CHECKSUM_PATH = EXPERIMENT_DIR / "fixed_full_1000_manifest.sha256"

SEED_NAMESPACE = "frontier-gated-dreamon-v0-seed-20260809"
FIXED_SELECTION_SEED = "frontier-gated-dreamon-fixed1000-v0-20260809"
BOOTSTRAP_SEED = 20260809
BOOTSTRAP_REPLICATES = 10000


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def width_label(width: int | None) -> str:
    return WIDTH_LABELS[width]


def stable_sample_seed(sample_id: str, width: int | None = None) -> int:
    del width
    digest = hashlib.sha256(f"{SEED_NAMESPACE}\0{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def generation_config() -> dict[str, Any]:
    return {
        "experiment": "Frontier-Gated DreamOn V0",
        "model_path": str(MODEL_PATH),
        "model_revision": MODEL_REVISION,
        "dreamon_repository_commit": DREAMON_COMMIT,
        "dtype": "bfloat16",
        "initial_mask_count": INITIAL_MASK_COUNT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "number_transfer_tokens": NUMBER_TRANSFER_TOKENS,
        "max_forwards": MAX_FORWARDS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "algorithm": ALGORITHM,
        "algorithm_temperature": ALGORITHM_TEMPERATURE,
        "delete_eos_token": True,
        "pad_eos_to_right": True,
        "full_sequence_attention": True,
        "widths": [width_label(width) for width in WIDTHS],
        "seed_namespace": SEED_NAMESPACE,
    }


def config_hash() -> str:
    return sha256_bytes(canonical_json(generation_config()).encode())
