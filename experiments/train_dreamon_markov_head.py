#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.dreamon_markov_head_training import (
    HEAD_SEED,
    MarkovHead,
    aligned_cross_entropy,
    atomic_save_checkpoint,
    distribution_loss,
    freeze_module,
    load_checkpoint,
    replay_target_logits,
)


MODEL_REVISION = "8ccc74750e43177327f29dab9e91882ba759e194"
SOURCE_REVISION = "8a0a54918412eda9402a327646f7f067f7160ec8"
RANK = 256
EFFECTIVE_BATCH = 128
PILOT_TRAIN = 20000
PILOT_VALIDATION = 4000
MAX_EPOCHS = 5
PATIENCE = 2
LEARNING_RATE = 3e-4
BETAS = (0.9, 0.95)
CLIP = 1.0
LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_csv(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def head_config(config: Any) -> dict[str, Any]:
    structural = sorted(
        {int(config.mask_token_id), int(config.expand_token_id), int(config.eos_token_id)}
    )
    return {
        "vocab_size": int(config.vocab_size),
        "rank": RANK,
        "structural_token_ids": structural,
        "seed": HEAD_SEED,
        "parameter_count": int(config.vocab_size) * RANK * 2,
    }


def make_head(config: Any, device: str) -> MarkovHead:
    values = head_config(config)
    head = MarkovHead(
        vocab_size=values["vocab_size"],
        rank=values["rank"],
        structural_token_ids=values["structural_token_ids"],
        seed=values["seed"],
    ).to(device)
    if head.parameter_count != values["parameter_count"]:
        raise RuntimeError("Markov head parameter count mismatch")
    return head


def init_common(args: argparse.Namespace) -> int:
    snapshot = Path(args.model_snapshot).resolve()
    config = AutoConfig.from_pretrained(snapshot, trust_remote_code=True, local_files_only=True)
    head = make_head(config, "cpu")
    path = Path(args.common_init).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "head": head.state_dict(),
            "head_config": head_config(config),
            "model_revision": MODEL_REVISION,
            "source_revision": SOURCE_REVISION,
        },
        temporary,
    )
    os.replace(temporary, path)
    atomic_json(
        Path(args.result_dir).resolve() / "common_initialization.json",
        {
            **head_config(config),
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
            "w2_all_zero": bool(torch.count_nonzero(head.output.weight).item() == 0),
        },
    )
    return 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReplayBank:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)

    def keys(self, split: str, limit: int | None = None, purpose: str = "full") -> list[str]:
        keys = [
            str(row[0])
            for row in self.connection.execute(
                "SELECT sample_key FROM samples WHERE split=? ORDER BY sample_key", (split,)
            )
        ]
        if limit is not None and len(keys) > limit:
            keys.sort(
                key=lambda key: hashlib.sha256(f"20260901:{purpose}:{key}".encode()).digest()
            )
            keys = keys[:limit]
            keys.sort()
        return keys

    def rows(self, keys: Sequence[str]) -> list[dict[str, Any]]:
        output = []
        for key in keys:
            payload = self.connection.execute(
                "SELECT payload FROM samples WHERE sample_key=?", (key,)
            ).fetchone()
            if payload is None:
                raise RuntimeError(f"missing replay key: {key}")
            output.append(json.loads(zlib.decompress(payload[0])))
        return output

    def close(self) -> None:
        self.connection.close()


def collate(rows: Sequence[Mapping[str, Any]], *, pad_id: int, device: str) -> dict[str, Any]:
    batch = len(rows)
    lengths = torch.tensor([len(row["stale_input_ids"]) for row in rows], device=device)
    maximum = int(lengths.max().item())
    combined = torch.full((batch * 2, maximum), int(pad_id), dtype=torch.long, device=device)
    attention = torch.zeros((batch * 2, maximum), dtype=torch.bool, device=device)
    targets = torch.empty(batch * 2, dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        length = len(row["stale_input_ids"])
        combined[index, :length] = torch.tensor(row["stale_input_ids"], device=device)
        combined[index + batch, :length] = torch.tensor(row["fresh_input_ids"], device=device)
        attention[index, :length] = True
        attention[index + batch, :length] = True
        targets[index] = int(row["target_position"])
        targets[index + batch] = int(row["target_position"])
    position_ids = attention.long().cumsum(-1) - 1
    position_ids.masked_fill_(~attention, 1)
    pairwise = torch.logical_and(
        attention.unsqueeze(1).unsqueeze(-2), attention.unsqueeze(1).unsqueeze(-1)
    )
    return {
        "input_ids": combined,
        "pairwise_attention": pairwise,
        "position_ids": position_ids,
        "target_positions": targets,
        "previous_token_ids": torch.tensor(
            [int(row["previous_token_id"]) for row in rows], device=device
        ),
        "reference_token_ids": torch.tensor(
            [max(0, int(row["reference_token_id"])) for row in rows], device=device
        ),
        "aligned": torch.tensor(
            [bool(row["reference_prefix_aligned_through_predecessor"]) for row in rows],
            device=device,
        ),
        "expand_masked": torch.tensor(
            [bool(row["expand_action_masked"]) for row in rows], device=device
        ),
        "rows": rows,
    }


@torch.inference_mode()
def teacher_logits(model: Any, batch: Mapping[str, Any], *, expand_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["pairwise_attention"],
        position_ids=batch["position_ids"],
        use_cache=False,
        return_dict=True,
    )
    logits = replay_target_logits(
        outputs.last_hidden_state, model.lm_head, batch["target_positions"]
    ).float()
    size = logits.shape[0] // 2
    stale, fresh = logits[:size], logits[size:]
    if bool(batch["expand_masked"].any().item()):
        stale[batch["expand_masked"], int(expand_id)] -= 1e9
        fresh[batch["expand_masked"], int(expand_id)] -= 1e9
    return stale, fresh


def batch_metrics(
    stale_logits: torch.Tensor,
    fresh_logits: torch.Tensor,
    head: MarkovHead,
    batch: Mapping[str, Any],
    *,
    kind: str,
    lambda_value: float,
    extended: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    bias = head(batch["previous_token_ids"]).float()
    q_logits = stale_logits.float() + float(lambda_value) * bias
    p_fresh = torch.softmax(fresh_logits.float(), dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    loss = 0.9 * distribution_loss(q, p_fresh, kind=kind) + 0.1 * aligned_cross_entropy(
        q_logits, batch["reference_token_ids"], batch["aligned"]
    )
    stale = torch.softmax(stale_logits.float(), dim=-1)
    baseline_tv = 0.5 * torch.abs(stale - p_fresh).sum(-1)
    head_tv = 0.5 * torch.abs(q - p_fresh).sum(-1)
    stale_top = stale.argmax(-1)
    fresh_top = p_fresh.argmax(-1)
    q_top = q.argmax(-1)
    stale_logp = F.log_softmax(stale_logits.float(), -1)
    q_logp = F.log_softmax(q_logits.float(), -1)
    stale_support = q_support = None
    if extended:
        def top_p_support(logits: torch.Tensor) -> torch.Tensor:
            scaled = logits.float() / 0.2
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            cumulative = torch.cumsum(torch.softmax(sorted_logits, -1), -1)
            remove = cumulative > 0.9
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            removed = torch.zeros_like(remove).scatter_(-1, sorted_indices, remove)
            return ~removed

        stale_support = top_p_support(stale_logits)
        q_support = top_p_support(q_logits)
    diagnostics = []
    for index, row in enumerate(batch["rows"]):
        aligned = bool(batch["aligned"][index].item())
        reference = int(batch["reference_token_ids"][index].item())
        item = {
                "sample_key": row["sample_key"],
                "problem_group_id": row["problem_group_id"],
                "trajectory_policy": row["trajectory_policy"],
                "generation_stage": row["generation_stage"],
                "previous_token_id": row["previous_token_id"],
                "baseline_tv": float(baseline_tv[index].item()),
                "head_tv": float(head_tv[index].item()),
                "tv_improvement": float((baseline_tv[index] - head_tv[index]).item()),
                "baseline_top1_agreement": bool(stale_top[index] == fresh_top[index]),
                "head_top1_agreement": bool(q_top[index] == fresh_top[index]),
                "mismatch_recovery": bool(
                    stale_top[index] != fresh_top[index] and q_top[index] == fresh_top[index]
                ),
                "stable_corruption": bool(
                    stale_top[index] == fresh_top[index] and q_top[index] != fresh_top[index]
                ),
                "reference_aligned": aligned,
                "baseline_reference_nll": (
                    float(-stale_logp[index, reference].item()) if aligned else None
                ),
                "head_reference_nll": float(-q_logp[index, reference].item()) if aligned else None,
            }
        if extended:
            item.update(
                {
                    "baseline_reference_rank": (
                        int((stale_logits[index] > stale_logits[index, reference]).sum().item()) + 1
                        if aligned
                        else None
                    ),
                    "head_reference_rank": (
                        int((q_logits[index] > q_logits[index, reference]).sum().item()) + 1
                        if aligned
                        else None
                    ),
                    "top_p_support_enter_count": int(
                        ((~stale_support[index]) & q_support[index]).sum().item()
                    ),
                    "top_p_support_exit_count": int(
                        (stale_support[index] & (~q_support[index])).sum().item()
                    ),
                }
            )
        diagnostics.append(item)
    return loss, diagnostics


def clustered_ci(rows: Sequence[Mapping[str, Any]], field: str, *, seed: int, reps: int) -> tuple[float, float, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            groups[str(row["problem_group_id"])].append(float(value))
    values = np.asarray([np.mean(items) for items in groups.values()], dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(values), size=(reps, len(values)))
    bootstrap = values[indexes].mean(1)
    return float(values.mean()), float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def summarize(rows: Sequence[Mapping[str, Any]], *, bootstrap_reps: int) -> dict[str, Any]:
    baseline = float(np.mean([row["baseline_tv"] for row in rows]))
    head = float(np.mean([row["head_tv"] for row in rows]))
    improvement, low, high = clustered_ci(
        rows, "tv_improvement", seed=20260901, reps=bootstrap_reps
    )
    aligned = [row for row in rows if row["reference_aligned"]]
    mismatch = [row for row in rows if not row["baseline_top1_agreement"]]
    stable = [row for row in rows if row["baseline_top1_agreement"]]
    return {
        "transitions": len(rows),
        "problem_groups": len({row["problem_group_id"] for row in rows}),
        "baseline_raw_tv": baseline,
        "head_raw_tv": head,
        "relative_raw_tv_improvement": (baseline - head) / baseline if baseline else 0.0,
        "cluster_tv_improvement": improvement,
        "cluster_tv_improvement_ci95_low": low,
        "cluster_tv_improvement_ci95_high": high,
        "baseline_top1_agreement": float(np.mean([row["baseline_top1_agreement"] for row in rows])),
        "head_top1_agreement": float(np.mean([row["head_top1_agreement"] for row in rows])),
        "mismatch_transitions": len(mismatch),
        "stable_transitions": len(stable),
        "mismatch_recovery": (
            float(np.mean([row["mismatch_recovery"] for row in mismatch])) if mismatch else 0.0
        ),
        "stable_corruption": (
            float(np.mean([row["stable_corruption"] for row in stable])) if stable else 0.0
        ),
        "aligned_transitions": len(aligned),
        "baseline_reference_nll": (
            float(np.mean([row["baseline_reference_nll"] for row in aligned])) if aligned else None
        ),
        "head_reference_nll": (
            float(np.mean([row["head_reference_nll"] for row in aligned])) if aligned else None
        ),
    }


def gate(metrics: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    if float(metrics["relative_raw_tv_improvement"]) < 0.10:
        failures.append("relative_raw_tv_improvement_below_10pct")
    if float(metrics["cluster_tv_improvement_ci95_low"]) <= 0:
        failures.append("cluster_ci_lower_not_positive")
    if float(metrics["head_top1_agreement"]) < float(metrics["baseline_top1_agreement"]):
        failures.append("fresh_top1_agreement_decreased")
    if float(metrics["mismatch_recovery"]) <= float(metrics["stable_corruption"]):
        failures.append("mismatch_recovery_not_above_stable_corruption")
    if metrics["head_reference_nll"] is not None and float(metrics["head_reference_nll"]) > float(
        metrics["baseline_reference_nll"]
    ):
        failures.append("aligned_reference_nll_worsened")
    return not failures, failures


def evaluate(
    bank: ReplayBank,
    keys: Sequence[str],
    *,
    model: Any,
    head: MarkovHead,
    config: Any,
    micro_batch: int,
    kind: str,
    lambda_value: float,
    device: str,
    bootstrap_reps: int,
    diagnostic_path: Path | None = None,
    extended: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows_out = []
    for start in range(0, len(keys), micro_batch):
        rows = bank.rows(keys[start : start + micro_batch])
        batch = collate(rows, pad_id=int(config.pad_token_id), device=device)
        stale, fresh = teacher_logits(model, batch, expand_id=int(config.expand_token_id))
        with torch.no_grad():
            _, diagnostics = batch_metrics(
                stale,
                fresh,
                head,
                batch,
                kind=kind,
                lambda_value=lambda_value,
                extended=extended,
            )
        rows_out.extend(diagnostics)
    if diagnostic_path is not None:
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(diagnostic_path, "wt", encoding="utf-8") as handle:
            for row in rows_out:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return summarize(rows_out, bootstrap_reps=bootstrap_reps), rows_out


def scheduler_for(optimizer: torch.optim.Optimizer, total_steps: int):
    warmup = max(1, math.ceil(total_steps * 0.05))

    def scale(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def load_common(head: MarkovHead, path: Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=False)
    head.load_state_dict(state["head"])
    return file_sha256(path)


def train_phase(
    *,
    phase: str,
    train_keys: Sequence[str],
    validation_keys: Sequence[str],
    bank: ReplayBank,
    model: Any,
    config: Any,
    head: MarkovHead,
    kind: str,
    micro_batch: int,
    checkpoint_dir: Path,
    manifest_version: str,
    curve_path: Path,
    device: str,
    bootstrap_reps: int,
) -> tuple[dict[str, Any], Path]:
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=LEARNING_RATE, betas=BETAS, weight_decay=0
    )
    steps_per_epoch = math.ceil(len(train_keys) / EFFECTIVE_BATCH)
    scheduler = scheduler_for(optimizer, steps_per_epoch * MAX_EPOCHS)
    best_metric = -float("inf")
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    no_improve = 0
    global_step = 0
    start_epoch = 0
    if last_path.exists():
        resumed = load_checkpoint(
            last_path, head=head, optimizer=optimizer, scheduler=scheduler
        )
        global_step = int(resumed["global_step"])
        start_epoch = int(resumed["epoch"])
        best_metric = float(resumed["best_metric"])
        no_improve = int(resumed.get("extra", {}).get("no_improve", 0))
    for epoch in range(start_epoch, MAX_EPOCHS):
        order = list(train_keys)
        random.Random(HEAD_SEED + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        losses = []
        for start in range(0, len(order), micro_batch):
            rows = bank.rows(order[start : start + micro_batch])
            batch = collate(rows, pad_id=int(config.pad_token_id), device=device)
            stale, fresh = teacher_logits(model, batch, expand_id=int(config.expand_token_id))
            loss, _ = batch_metrics(stale, fresh, head, batch, kind=kind, lambda_value=1.0)
            (loss * len(rows) / EFFECTIVE_BATCH).backward()
            accumulated += len(rows)
            losses.append(float(loss.item()))
            if accumulated >= EFFECTIVE_BATCH or start + micro_batch >= len(order):
                torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accumulated = 0
                global_step += 1
        validation, _ = evaluate(
            bank,
            validation_keys,
            model=model,
            head=head,
            config=config,
            micro_batch=micro_batch,
            kind=kind,
            lambda_value=1.0,
            device=device,
            bootstrap_reps=bootstrap_reps,
        )
        metric = float(validation["relative_raw_tv_improvement"])
        curve = {
            "head": kind,
            "phase": phase,
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": float(np.mean(losses)),
            "validation_baseline_raw_tv": validation["baseline_raw_tv"],
            "validation_head_raw_tv": validation["head_raw_tv"],
            "validation_relative_raw_tv_improvement": metric,
            "validation_ci95_low": validation["cluster_tv_improvement_ci95_low"],
            "validation_top1_agreement": validation["head_top1_agreement"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        append_csv(curve_path, curve)
        atomic_save_checkpoint(
            last_path,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch=epoch + 1,
            best_metric=max(best_metric, metric),
            manifest_version=manifest_version,
            extra={
                "kind": kind,
                "phase": phase,
                "validation": validation,
                "no_improve": no_improve,
            },
        )
        if metric > best_metric:
            best_metric = metric
            no_improve = 0
            atomic_save_checkpoint(
                best_path,
                head=head,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                epoch=epoch + 1,
                best_metric=best_metric,
                manifest_version=manifest_version,
                extra={
                    "kind": kind,
                    "phase": phase,
                    "validation": validation,
                    "no_improve": no_improve,
                },
            )
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
    load_checkpoint(best_path, head=head, optimizer=optimizer, scheduler=scheduler)
    final_validation, _ = evaluate(
        bank,
        validation_keys,
        model=model,
        head=head,
        config=config,
        micro_batch=micro_batch,
        kind=kind,
        lambda_value=1.0,
        device=device,
        bootstrap_reps=bootstrap_reps,
    )
    passed, failures = gate(final_validation)
    return {
        "phase": phase,
        "passed": passed,
        "failures": failures,
        "validation_lambda_1": final_validation,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "global_step": global_step,
    }, best_path


def real_overfit_smoke(
    bank: ReplayBank,
    keys: Sequence[str],
    *,
    model: Any,
    config: Any,
    head: MarkovHead,
    kind: str,
    micro_batch: int,
    device: str,
) -> dict[str, Any]:
    selected = list(keys[:32])
    cached = []
    for start in range(0, len(selected), micro_batch):
        rows = bank.rows(selected[start : start + micro_batch])
        batch = collate(rows, pad_id=int(config.pad_token_id), device=device)
        stale, fresh = teacher_logits(model, batch, expand_id=int(config.expand_token_id))
        cached.append((batch, stale, fresh))
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=0)
    losses = []
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for batch, stale, fresh in cached:
            loss, _ = batch_metrics(stale, fresh, head, batch, kind=kind, lambda_value=1.0)
            (loss / len(cached)).backward()
            total += float(loss.item()) / len(cached)
        torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP)
        optimizer.step()
        losses.append(total)
    return {
        "samples": len(selected),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "decreased": losses[-1] < losses[0] * 0.9,
        "finite": all(math.isfinite(value) for value in losses),
    }


def determine_microbatch(args: argparse.Namespace) -> int:
    config = AutoConfig.from_pretrained(
        Path(args.model_snapshot).resolve(), trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        Path(args.model_snapshot).resolve(),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    freeze_module(model)
    bank = ReplayBank(Path(args.bank_db).resolve())
    keys = bank.keys("train", limit=32, purpose="microbatch")
    chosen = None
    attempts = []
    for candidate in (16, 8, 4, 2, 1):
        try:
            head = make_head(config, args.device)
            rows = bank.rows(keys[:candidate])
            batch = collate(rows, pad_id=int(config.pad_token_id), device=args.device)
            stale, fresh = teacher_logits(model, batch, expand_id=int(config.expand_token_id))
            loss, _ = batch_metrics(stale, fresh, head, batch, kind="tv", lambda_value=1.0)
            loss.backward()
            attempts.append({"micro_batch": candidate, "status": "passed"})
            chosen = candidate
            del head, batch, stale, fresh, loss
            torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            attempts.append({"micro_batch": candidate, "status": "oom"})
            torch.cuda.empty_cache()
    if chosen is None:
        raise RuntimeError("no micro-batch fits H200")
    payload = {
        "micro_batch": chosen,
        "effective_batch": EFFECTIVE_BATCH,
        "gradient_accumulation": EFFECTIVE_BATCH // chosen,
        "attempts": attempts,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    atomic_json(Path(args.common_training_config).resolve(), payload)
    bank.close()
    return 0


def run_head(args: argparse.Namespace) -> int:
    started = time.time()
    random.seed(HEAD_SEED)
    np.random.seed(HEAD_SEED)
    torch.manual_seed(HEAD_SEED)
    torch.cuda.manual_seed_all(HEAD_SEED)
    config = AutoConfig.from_pretrained(
        Path(args.model_snapshot).resolve(), trust_remote_code=True, local_files_only=True
    )
    common = json.loads(Path(args.common_training_config).read_text())
    micro_batch = int(common["micro_batch"])
    model = AutoModel.from_pretrained(
        Path(args.model_snapshot).resolve(),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    freeze_module(model)
    versions_before = [parameter._version for parameter in model.parameters()]
    bank = ReplayBank(Path(args.bank_db).resolve())
    train_full = bank.keys("train")
    validation_full = bank.keys("validation")
    train_pilot = bank.keys("train", limit=PILOT_TRAIN, purpose="pilot_train")
    validation_pilot = bank.keys(
        "validation", limit=PILOT_VALIDATION, purpose="pilot_validation"
    )
    result_dir = Path(args.result_dir).resolve()
    checkpoint_root = Path(args.checkpoint_root).resolve() / args.kind
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    curve_path = result_dir / f"training_curves_{args.kind}.csv"
    if curve_path.exists() and not any(checkpoint_root.glob("*/last.pt")):
        curve_path.unlink()
    head = make_head(config, args.device)
    common_sha = load_common(head, Path(args.common_init).resolve())
    zero_equivalence = bool(torch.count_nonzero(head.output.weight).item() == 0)
    smoke = real_overfit_smoke(
        bank,
        train_pilot,
        model=model,
        config=config,
        head=head,
        kind=args.kind,
        micro_batch=micro_batch,
        device=args.device,
    )
    if not smoke["decreased"] or not smoke["finite"]:
        raise RuntimeError("real 32-transition overfit smoke failed")
    head = make_head(config, args.device)
    load_common(head, Path(args.common_init).resolve())
    manifest_version = file_sha256(Path(args.bank_db).resolve())
    pilot, _ = train_phase(
        phase="pilot",
        train_keys=train_pilot,
        validation_keys=validation_pilot,
        bank=bank,
        model=model,
        config=config,
        head=head,
        kind=args.kind,
        micro_batch=micro_batch,
        checkpoint_dir=checkpoint_root / "pilot",
        manifest_version=manifest_version,
        curve_path=curve_path,
        device=args.device,
        bootstrap_reps=2000,
    )
    evaluate(
        bank,
        validation_pilot,
        model=model,
        head=head,
        config=config,
        micro_batch=micro_batch,
        kind=args.kind,
        lambda_value=1.0,
        device=args.device,
        bootstrap_reps=500,
        diagnostic_path=result_dir / f"pilot_validation_diagnostics_{args.kind}.jsonl.gz",
    )
    full = None
    lambda_metrics = []
    chosen_lambda = 0.0
    best_path = None
    if pilot["passed"]:
        head = make_head(config, args.device)
        load_common(head, Path(args.common_init).resolve())
        full, best_path = train_phase(
            phase="full",
            train_keys=train_full,
            validation_keys=validation_full,
            bank=bank,
            model=model,
            config=config,
            head=head,
            kind=args.kind,
            micro_batch=micro_batch,
            checkpoint_dir=checkpoint_root / "full",
            manifest_version=manifest_version,
            curve_path=curve_path,
            device=args.device,
            bootstrap_reps=10000,
        )
        if full["passed"]:
            for lambda_value in LAMBDA_GRID:
                metrics, _ = evaluate(
                    bank,
                    validation_full,
                    model=model,
                    head=head,
                    config=config,
                    micro_batch=micro_batch,
                    kind=args.kind,
                    lambda_value=lambda_value,
                    device=args.device,
                    bootstrap_reps=2000,
                    diagnostic_path=(
                        result_dir / f"validation_diagnostics_{args.kind}.jsonl.gz"
                        if lambda_value == 1.0
                        else None
                    ),
                )
                lambda_metrics.append({"lambda": lambda_value, **metrics})
            lambda_metrics.sort(
                key=lambda row: (
                    row["head_raw_tv"],
                    row["stable_corruption"],
                    row["head_reference_nll"]
                    if row["head_reference_nll"] is not None
                    else float("inf"),
                )
            )
            chosen_lambda = float(lambda_metrics[0]["lambda"])
    versions_after = [parameter._version for parameter in model.parameters()]
    if versions_before != versions_after:
        raise RuntimeError("DreamOn weights changed")
    status = {
        "head": args.kind,
        "common_initialization_sha256": common_sha,
        "zero_init_no_head_equivalence": zero_equivalence,
        "smoke": smoke,
        "pilot": pilot,
        "full": full,
        "lambda_validation": lambda_metrics,
        "chosen_lambda": chosen_lambda,
        "deployable_gain": bool(full and full["passed"] and chosen_lambda != 0),
        "best_checkpoint": str(best_path) if best_path else None,
        "best_checkpoint_size": best_path.stat().st_size if best_path else None,
        "dreamon_requires_grad_false": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "dreamon_parameter_versions_unchanged": True,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": time.time() - started,
        "micro_batch": micro_batch,
        "effective_batch": EFFECTIVE_BATCH,
    }
    atomic_json(result_dir / f"{args.kind}_training_status.json", status)
    bank.close()
    del head, model
    torch.cuda.empty_cache()
    return 0


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-snapshot", required=True)
    common.add_argument("--result-dir", required=True)
    init = sub.add_parser("init-common", parents=[common])
    init.add_argument("--common-init", required=True)
    micro = sub.add_parser("determine-microbatch", parents=[common])
    micro.add_argument("--bank-db", required=True)
    micro.add_argument("--common-training-config", required=True)
    micro.add_argument("--device", default="cuda")
    run = sub.add_parser("run-head", parents=[common])
    run.add_argument("--kind", choices=("tv", "kl"), required=True)
    run.add_argument("--bank-db", required=True)
    run.add_argument("--common-init", required=True)
    run.add_argument("--common-training-config", required=True)
    run.add_argument("--checkpoint-root", required=True)
    run.add_argument("--device", default="cuda")
    return parser


def main(args: argparse.Namespace) -> int:
    if args.command == "init-common":
        return init_common(args)
    if args.command == "determine-microbatch":
        return determine_microbatch(args)
    return run_head(args)


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
