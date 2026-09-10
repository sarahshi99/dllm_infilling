#!/usr/bin/env python3
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments.markov_data_identity import connected_groups, human_index, human_matches


SPLIT_SEED = 20260901
HEAD_SEED = 42
DEFAULT_RANK = 256


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalized_tests(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    return sorted(item for item in (normalize_text(item) for item in values) if item)


def _ast_fingerprint(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return None
    return _sha256_text(ast.dump(tree, annotate_fields=True, include_attributes=False))


def _assert_fingerprints(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return set()
    output = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            output.add(
                _sha256_text(ast.dump(node.test, annotate_fields=True, include_attributes=False))
            )
    return output


def build_humaneval_fingerprints(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base_tasks: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        parts = task_id.split("/")
        base_id = "/".join(parts[:3]) if len(parts) >= 3 else task_id
        if base_id in base_tasks:
            continue
        prompt = normalize_text(row.get("prompt"))
        middle = normalize_text(row.get("canonical_solution"))
        suffix = normalize_text(row.get("suffix"))
        full_code = normalize_text(
            str(row.get("prompt") or "")
            + str(row.get("canonical_solution") or "")
            + str(row.get("suffix") or "")
        )
        test = normalize_text(row.get("test"))
        base_tasks[base_id] = {
            "entry_point": normalize_text(row.get("entry_point")),
            "prompt": prompt,
            "middle": middle,
            "suffix": suffix,
            "full_code": full_code,
            "test": test,
        }
    code_text_hashes: set[str] = set()
    code_ast_hashes: set[str] = set()
    assert_by_entry: dict[str, set[str]] = {}
    prompt_hashes: set[str] = set()
    for item in base_tasks.values():
        for code in (item["middle"], item["full_code"]):
            if code:
                code_text_hashes.add(_sha256_text(code))
                fingerprint = _ast_fingerprint(code)
                if fingerprint:
                    code_ast_hashes.add(fingerprint)
        if item["prompt"]:
            prompt_hashes.add(_sha256_text(item["prompt"]))
        entry = item["entry_point"]
        assert_by_entry.setdefault(entry, set()).update(_assert_fingerprints(item["test"]))
    return {
        "base_task_count": len(base_tasks),
        "code_text_hashes": code_text_hashes,
        "code_ast_hashes": code_ast_hashes,
        "assert_by_entry": assert_by_entry,
        "prompt_hashes": prompt_hashes,
    }


def _contaminated_by_humaneval(
    record: Mapping[str, Any], fingerprints: Mapping[str, Any]
) -> tuple[bool, str | None]:
    if not fingerprints or not fingerprints.get("base_task_count"):
        return False, None
    code = str(record["code"])
    if _sha256_text(code) in fingerprints["code_text_hashes"]:
        return True, "exact_normalized_code"
    code_ast = _ast_fingerprint(code)
    if code_ast and code_ast in fingerprints["code_ast_hashes"]:
        return True, "python_ast_code"
    instruction = str(record["instruction"])
    if instruction and _sha256_text(instruction) in fingerprints["prompt_hashes"]:
        return True, "exact_normalized_prompt"
    entry = str(record["entry_point"])
    human_asserts = fingerprints["assert_by_entry"].get(entry, set())
    record_asserts: set[str] = set()
    for test in record["testcase"]:
        record_asserts.update(_assert_fingerprints(test))
    if human_asserts and record_asserts.intersection(human_asserts):
        return True, "matching_entry_point_and_assert_ast"
    return False, None


def deterministic_split(problem_key: str, *, seed: int = SPLIT_SEED) -> str:
    digest = hashlib.sha256(f"{int(seed)}:{problem_key}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if bucket < 0.8:
        return "train"
    if bucket < 0.9:
        return "validation"
    return "external_test"


def _normalize_record(row: Mapping[str, Any]) -> dict[str, Any]:
    tests = _normalized_tests(row.get("testcase"))
    return {
        "source_seq_id": int(row.get("seq_id", -1)),
        "instruction": normalize_text(row.get("instruction")),
        "output": normalize_text(row.get("output")),
        "code": normalize_text(row.get("code")),
        "entry_point": normalize_text(row.get("entry_point")),
        "testcase": tests,
    }


def deduplicate_and_split_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    human_eval_rows: Sequence[Mapping[str, Any]],
    split_seed: int = SPLIT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    human_eval = build_humaneval_fingerprints(human_eval_rows)
    stronger_human_eval = human_index(human_eval_rows)
    exact_seen: dict[str, dict[str, Any]] = {}
    duplicate_count = 0
    invalid_count = 0
    for source in rows:
        record = _normalize_record(source)
        if not record["code"] or not record["instruction"]:
            invalid_count += 1
            continue
        exact_key = _sha256_text(
            _canonical_json(
                {
                    "instruction": record["instruction"],
                    "code": record["code"],
                    "entry_point": record["entry_point"],
                    "testcase": record["testcase"],
                }
            )
        )
        if exact_key in exact_seen:
            duplicate_count += 1
            exact_seen[exact_key]["duplicate_source_seq_ids"].append(record["source_seq_id"])
            continue
        record["record_id"] = exact_key
        record["duplicate_source_seq_ids"] = []
        exact_seen[exact_key] = record

    output: list[dict[str, Any]] = []
    records = sorted(exact_seen.values(), key=lambda r: r["record_id"])
    groups = connected_groups(records)
    direct_matches: dict[str, dict[str, Any]] = {}
    contamination_reasons: dict[str, int] = {}
    for record in records:
        legacy_contaminated, legacy_reason = _contaminated_by_humaneval(record, human_eval)
        matches = human_matches(record, stronger_human_eval)
        reasons = {match["reason"] for match in matches}
        if legacy_contaminated and legacy_reason:
            reasons.add(str(legacy_reason))
        if not reasons:
            continue
        for reason in reasons:
            contamination_reasons[reason] = contamination_reasons.get(reason, 0) + 1
        direct_matches[record["record_id"]] = {
            "record_id": record["record_id"],
            "source_seq_id": record["source_seq_id"],
            "duplicate_source_seq_ids": record["duplicate_source_seq_ids"],
            "reasons": sorted(reasons),
            "matches": matches,
        }
    members_by_group: dict[int, list[dict[str, Any]]] = {}
    for record, representative in zip(records, groups):
        members_by_group.setdefault(representative, []).append(record)
    excluded_groups = {
        representative
        for representative, members in members_by_group.items()
        if any(record["record_id"] in direct_matches for record in members)
    }
    exclusion_details = []
    for representative in sorted(excluded_groups):
        members = members_by_group[representative]
        exclusion_details.append(
            {
                "problem_group_id": records[representative]["record_id"],
                "member_record_ids": [record["record_id"] for record in members],
                "member_source_seq_ids": [record["source_seq_id"] for record in members],
                "direct_matches": [
                    direct_matches[record["record_id"]]
                    for record in members
                    if record["record_id"] in direct_matches
                ],
            }
        )
    for record, representative in zip(records, groups):
        if representative in excluded_groups:
            continue
        # Existing record IDs suffice. Never regenerate a completed v1 manifest:
        # this corrected grouping applies only to newly versioned datasets.
        problem_key = records[representative]["record_id"]
        record["problem_group_id"] = problem_key
        record["split"] = deterministic_split(problem_key, seed=split_seed)
        output.append(record)
    output.sort(key=lambda item: (item["problem_group_id"], item["record_id"]))
    split_counts: dict[str, int] = {"train": 0, "validation": 0, "external_test": 0}
    for record in output:
        split_counts[record["split"]] += 1
    audit = {
        "grouping_version": "connected_instruction_code_tests_v2",
        "source_rows": len(rows),
        "normalized_duplicate_rows_removed": duplicate_count,
        "invalid_rows_removed": invalid_count,
        "humaneval_base_tasks": human_eval["base_task_count"],
        "humaneval_variant_consistency": stronger_human_eval["variant_consistency"],
        "humaneval_direct_candidate_rows": len(direct_matches),
        "humaneval_excluded_groups": len(excluded_groups),
        "humaneval_decontaminated_rows": sum(
            len(members_by_group[representative]) for representative in excluded_groups
        ),
        "humaneval_decontamination_reasons": contamination_reasons,
        "humaneval_exclusion_details": exclusion_details,
        "available_rows_after_filters": len(output),
        "problem_group_count": len({row["problem_group_id"] for row in output}),
        "split_seed": int(split_seed),
        "split_counts": split_counts,
        "cross_split_problem_groups": 0,
    }
    return output, audit


def deterministic_line_split(code: str, *, row_seed: int) -> tuple[str, str, str]:
    normalized = normalize_text(code)
    lines = normalized.split("\n")
    if len(lines) < 3:
        raise ValueError("code has fewer than three lines")
    rng = random.Random(int(row_seed))
    candidates: list[tuple[int, int]] = []
    for _ in range(5):
        start = rng.randint(1, len(lines) - 2)
        end = rng.randint(start + 1, len(lines) - 1)
        candidates.append((start, end))
    candidates.extend(
        (start, end)
        for start in range(1, len(lines) - 1)
        for end in range(start + 1, len(lines))
    )
    for start, end in candidates:
        middle = "\n".join(lines[start:end]) + "\n"
        if len(middle.split()) >= 2 and "def" not in middle:
            prefix = "\n".join(lines[:start]) + "\n"
            suffix = "\n".join(lines[end:])
            return prefix, middle, suffix
    raise ValueError("no legal line middle with at least two ordinary tokens")


class MarkovHead(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        rank: int = DEFAULT_RANK,
        structural_token_ids: Sequence[int],
        seed: int = HEAD_SEED,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.rank = int(rank)
        ids = sorted({int(token_id) for token_id in structural_token_ids})
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in ids):
            raise ValueError("structural token id is outside the vocabulary")
        self.embedding = nn.Embedding(self.vocab_size, self.rank)
        self.output = nn.Linear(self.rank, self.vocab_size, bias=False)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        with torch.no_grad():
            self.embedding.weight.normal_(mean=0.0, std=1.0, generator=generator)
            self.output.weight.zero_()
        mask = torch.ones(self.vocab_size, dtype=torch.float32)
        if ids:
            mask[torch.tensor(ids, dtype=torch.long)] = 0.0
        self.register_buffer("output_mask", mask, persistent=True)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, previous_token_ids: torch.Tensor) -> torch.Tensor:
        bias = self.output(self.embedding(previous_token_ids))
        return bias * self.output_mask.to(device=bias.device, dtype=bias.dtype)


def distribution_loss(q: torch.Tensor, p_fresh: torch.Tensor, *, kind: str) -> torch.Tensor:
    q32 = q.float()
    p32 = p_fresh.float()
    if kind == "tv":
        return torch.abs(q32 - p32).sum(dim=-1).mean()
    if kind == "kl":
        tiny = torch.finfo(torch.float32).tiny
        terms = torch.where(
            p32 > 0,
            p32 * (torch.clamp_min(p32, tiny).log() - torch.clamp_min(q32, tiny).log()),
            torch.zeros_like(p32),
        )
        return terms.sum(dim=-1).mean()
    raise ValueError(f"unknown distribution loss: {kind}")


def aligned_cross_entropy(
    q_logits: torch.Tensor, reference_token_ids: torch.Tensor, aligned_mask: torch.Tensor
) -> torch.Tensor:
    mask = aligned_mask.bool()
    if not bool(mask.any().item()):
        return q_logits.float().sum() * 0.0
    return F.cross_entropy(q_logits.float()[mask], reference_token_ids.long()[mask], reduction="mean")


def markov_total_loss(
    q: torch.Tensor,
    p_fresh: torch.Tensor,
    q_logits: torch.Tensor,
    reference_token_ids: torch.Tensor,
    aligned_mask: torch.Tensor,
    *,
    kind: str,
) -> torch.Tensor:
    return 0.9 * distribution_loss(q, p_fresh, kind=kind) + 0.1 * aligned_cross_entropy(
        q_logits, reference_token_ids, aligned_mask
    )


def accumulated_markov_loss(
    q: torch.Tensor, p_fresh: torch.Tensor, q_logits: torch.Tensor,
    reference_token_ids: torch.Tensor, aligned_mask: torch.Tensor, *,
    kind: str, total_rows: int, total_aligned: int,
) -> torch.Tensor:
    """One microbatch contribution to a whole optimizer-batch mean.

    Distribution and reference losses have different denominators. The latter
    counts only aligned rows, including across microbatches with zero such rows.
    """
    count = int(aligned_mask.bool().sum().item())
    if total_rows < len(q) or total_aligned < count or total_aligned > total_rows:
        raise ValueError("invalid optimizer-batch denominators")
    dist = distribution_loss(q, p_fresh, kind=kind) * (len(q) / total_rows)
    ce = aligned_cross_entropy(q_logits, reference_token_ids, aligned_mask)
    ce = ce * (count / total_aligned if total_aligned else 0.0)
    return 0.9 * dist + 0.1 * ce


def replay_target_logits(
    hidden_states: torch.Tensor, lm_head: nn.Module, target_positions: torch.Tensor
) -> torch.Tensor:
    positions = target_positions.long()
    if bool((positions < 0).any().item()) or bool(
        (positions >= hidden_states.shape[1]).any().item()
    ):
        raise ValueError("target position is outside hidden states")
    source_positions = torch.clamp_min(positions - 1, 0)
    selected = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), source_positions]
    return lm_head(selected)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def backbone_state_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save_checkpoint(
    path: Path,
    *,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int,
    epoch: int,
    best_metric: float,
    manifest_version: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": int(global_step),
            "epoch": int(epoch),
            "best_metric": float(best_metric),
            "manifest_version": str(manifest_version),
            "rng_state": _rng_state(),
            "extra": dict(extra or {}),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    *,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    head.load_state_dict(state["head"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    _restore_rng_state(state["rng_state"])
    return state
