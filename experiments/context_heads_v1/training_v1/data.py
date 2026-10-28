"""Read only common tensor fields plus frozen fresh_logits from A's cached bank."""
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..contract import HeadBatch, validate
from ..integration_v1.adapter import config_from_a_schema

FIELDS = ("old_hidden", "position_ids", "chain_ids", "selected_mask", "query_indices",
          "old_logits", "token_ids", "token_state", "history_mask")
META = ("state_id", "raw_path", "raw_sha256", "teacher_path", "teacher_sha256",
        "group", "split", "span", "order", "arm", "history_length")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f".tmp-{os.getpid()}")
    with temp.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def metadata(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            source = json.loads(line)
            # No effect/quality-based filtering or access to generation scoring.
            rows.append({key: source[key] for key in META})
    rows.sort(key=lambda r: r["state_id"])
    if len(rows) != len({r["state_id"] for r in rows}):
        raise ValueError("duplicate teacher state")
    if any(r["split"] not in ("train", "validation") for r in rows):
        raise ValueError("pilot accepts only original train/validation splits")
    train = {r["group"] for r in rows if r["split"] == "train"}
    val = {r["group"] for r in rows if r["split"] == "validation"}
    if not train or not val or train & val:
        raise ValueError("empty split or source-group leakage")
    return rows


def load_arrays(row, cache, config):
    state, teacher = Path(cache)/row["raw_path"], Path(cache)/row["teacher_path"]
    if sha(state) != row["raw_sha256"] or sha(teacher) != row["teacher_sha256"]:
        raise ValueError("cached input changed: "+row["state_id"])
    with np.load(state, allow_pickle=False) as package:
        arrays = {key: package[key] for key in FIELDS}
    with np.load(teacher, allow_pickle=False) as package:
        fresh = package["fresh_logits"]
    batch = HeadBatch(**{key: torch.from_numpy(value) for key, value in arrays.items()})
    validate(batch, config)
    if arrays["old_hidden"].shape[0] != 1 or arrays["query_indices"].shape != (1, 1):
        raise ValueError("pilot requires A's B=Q=1 cached states")
    if int(batch.history_mask.sum()) != row["history_length"] or row["history_length"] not in (1, 2, 3):
        raise ValueError("history metadata mismatch")
    # Teacher bank must contain the complete legal chain prefix, never holes.
    q = int(batch.query_indices[0, 0])
    possible = (batch.selected_mask[0] & (batch.chain_ids[0] == batch.chain_ids[0, q])
                & (batch.position_ids[0] < batch.position_ids[0, q]))
    if not torch.equal(batch.history_mask[0, 0], possible):
        raise ValueError("teacher input is not a complete committed chain prefix")
    for key in ("old_hidden", "old_logits"):
        if arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
            raise ValueError(key+" must be lossless finite FP32")
    if fresh.shape != (config.vocab_size,) or fresh.dtype != np.float32 or not np.isfinite(fresh).all():
        raise ValueError("fresh teacher must be finite full-support FP32 logits")
    arrays["teacher_logits"] = fresh[None, None]
    return arrays


def shapes(config):
    k, h, v = config.max_slots, config.hidden_size, config.vocab_size
    return {"old_hidden": (k, h), "position_ids": (k,), "chain_ids": (k,),
            "selected_mask": (k,), "query_indices": (1,), "old_logits": (1, v),
            "token_ids": (k,), "token_state": (k,), "history_mask": (1, k),
            "teacher_logits": (1, v)}


def prepare(a_output, destination):
    """Lossless memmap pack, restartable every 128 verified rows, no GPU needed."""
    a_output, destination = Path(a_output), Path(destination)
    source = a_output/"state_metrics.jsonl.gz"
    source_sha = sha(source)
    if (destination/"manifest.json").exists():
        manifest = json.loads((destination/"manifest.json").read_text())
        if manifest["source_sha256"] != source_sha:
            raise ValueError("existing pack comes from different A metadata")
        return manifest
    rows = metadata(source)
    schema = json.loads((a_output/"protocol_schema.json").read_text())
    # Reuse B's local tokenizer verification; never load a backbone/tokenizer here.
    tokenizer = json.loads((Path(__file__).resolve().parents[1]/"artifacts/dreamon_metadata.json").read_text())
    if schema["H"] != tokenizer["hidden_H"] or schema["V"] != tokenizer["model_logit_V"]:
        raise ValueError("A schema differs from the verified frozen backbone")
    config = config_from_a_schema(schema, tokenizer_size=tokenizer["tokenizer_size"])
    cache = Path(json.loads((a_output/"runtime_paths.local.json").read_text())["cache_root"])
    destination.mkdir(parents=True, exist_ok=True)
    progress_path = destination/"prepare_progress.json"
    done = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["source_sha256"] != source_sha:
            raise ValueError("incomplete pack belongs to different source")
        done = progress["rows_done"]
    maps = {}
    for name, shape in shapes(config).items():
        dtype = np.bool_ if name in ("selected_mask", "history_mask") else np.float32 if name in ("old_hidden", "old_logits", "teacher_logits") else np.int64
        path = destination/(name+".npy")
        if done and not path.exists():
            raise ValueError("incomplete resumed pack")
        array = np.lib.format.open_memmap(path, mode="r+" if path.exists() else "w+", dtype=dtype,
                                         shape=(len(rows), *shape))
        if array.shape != (len(rows), *shape) or array.dtype != dtype:
            raise ValueError("pack shape/dtype mismatch")
        maps[name] = array
    for i in range(done, len(rows)):
        values = load_arrays(rows[i], cache, config)
        k = values["old_hidden"].shape[1]
        for name, array in maps.items():
            array[i] = -1 if name == "chain_ids" else 0
            if name == "old_hidden":
                array[i, :k] = values[name][0]
            elif name == "history_mask":
                array[i, :, :k] = values[name][0]
            elif name in ("query_indices", "old_logits", "teacher_logits"):
                array[i] = values[name][0]
            else:
                array[i, :k] = values[name][0]
        if (i+1) % 128 == 0 or i+1 == len(rows):
            for array in maps.values():
                array.flush()
            atomic_json(progress_path, dict(source_sha256=source_sha, rows_done=i+1, total=len(rows)))
            print(f"prepare {i+1}/{len(rows)}", flush=True)
    atomic_json(destination/"records.json", rows)
    groups = {s: sorted({r["group"] for r in rows if r["split"] == s}) for s in ("train", "validation")}
    manifest = dict(source_sha256=source_sha, source_file="state_metrics.jsonl.gz",
                    schema_sha256=sha(a_output/"protocol_schema.json"),
                    backbone_revision=tokenizer["revision"],
                    contract_id=schema["contract_id"], schema_version=schema["schema_version"],
                    config=vars(config), count=len(rows),
                    splits=dict(collections.Counter(r["split"] for r in rows)),
                    source_groups={s: len(values) for s, values in groups.items()}, group_overlap=0,
                    cells=dict(collections.Counter("|".join(str(r[k]) for k in ("split", "order", "span", "arm", "history_length")) for r in rows)),
                    fp32_lossless=True, fields=list(maps),
                    packed_bytes=sum(a.nbytes for a in maps.values()),
                    excluded="counterfactual/DCD/generation/evaluation answers/tests/passed; no new teacher collection")
    atomic_json(destination/"manifest.json", manifest)
    return manifest


class PackedBank:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads((self.path/"manifest.json").read_text())
        self.rows = json.loads((self.path/"records.json").read_text())
        self.arrays = {name: np.load(self.path/(name+".npy"), mmap_mode="r", allow_pickle=False)
                       for name in self.manifest["fields"]}

    def indices(self, split):
        return np.array([i for i, row in enumerate(self.rows) if row["split"] == split], dtype=np.int64)

    def batch(self, indices, device):
        # Advanced indexing makes writable copies; original cache/memmaps stay readonly.
        values = {key: torch.from_numpy(array[indices]).to(device) for key, array in self.arrays.items()}
        teacher = values.pop("teacher_logits")
        return HeadBatch(**values), teacher
