"""Single-worker, seven-head, one-epoch frozen-cache KL pilot. No backbone calls."""
import argparse
import gc
import json
import math
import os
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..contract import HeadConfig
from ..heads import ContextHead, MODEL_NAMES
from ..loss import teacher_kl
from .data import PackedBank, atomic_json, prepare, sha

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN = Path("/home/shx/.cache/dllm_infilling/context_heads_training_20261028_v1")


def utc():
    return datetime.now(timezone.utc).isoformat()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".tmp-{os.getpid()}")
    with temporary.open("wb") as f:
        torch.save(cpu_tree(value), f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def update(head, optimizer, batch, teacher):
    head.train()
    optimizer.zero_grad(set_to_none=True)
    output = head(batch)
    loss = teacher_kl(batch.old_logits, output.delta_logits, teacher, batch.history_mask.any(-1))
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite common teacher KL")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)


@torch.inference_mode()
def evaluate(head, bank, indices, device, batch_size):
    if head is not None:
        head.eval()
    buckets = defaultdict(list)
    groups = defaultdict(list)
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start+batch_size]
        batch, teacher = bank.batch(selected, device)
        scores = batch.old_logits if head is None else batch.old_logits + head(batch).delta_logits
        lp, lq = teacher.float().log_softmax(-1), scores.float().log_softmax(-1)
        p, q = lp.exp(), lq.exp()
        kl = (p*(lp-lq)).sum(-1).flatten().cpu().tolist()
        tv = ((p-q).abs().sum(-1)/2).flatten().cpu().tolist()
        for index, a, b in zip(selected, kl, tv):
            if not math.isfinite(a) or not math.isfinite(b):
                raise FloatingPointError("nonfinite validation metric")
            row = bank.rows[int(index)]
            values = (a, b)
            buckets["all"].append(values)
            for key in ("history_length", "order", "span", "arm"):
                buckets[f"{key}={row[key]}"].append(values)
            buckets[f"order={row['order']}|history={row['history_length']}"].append(values)
            groups[row["group"]].append(a)
    return dict(by_bucket={key: dict(count=len(values), kl_T1=float(np.mean([v[0] for v in values])),
                                    tv_T1=float(np.mean([v[1] for v in values])))
                           for key, values in buckets.items()},
                group_mean_kl_T1=float(np.mean([np.mean(v) for v in groups.values()])),
                source_groups=len(groups))


def train_one(name, config, bank, plan, args, status, run_start):
    folder = args.run_dir/"checkpoints"/name
    folder.mkdir(parents=True, exist_ok=True)
    result_path = args.run_dir/"results"/(name+".json")
    if result_path.exists():
        return json.loads(result_path.read_text())
    device = torch.device(args.device)
    head = ContextHead(name, config).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate,
                                 betas=(.9, .999), eps=1e-8, weight_decay=0.)
    state = None
    cursor, steps, loss_sum, nseen = 0, 0, 0., 0
    last_path = folder/"last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["model"] != name or state["plan"] != plan:
            raise ValueError("checkpoint differs from common training plan")
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        cursor, steps = state["cursor"], state["steps"]
        loss_sum, nseen = state["loss_sum"], state["nseen"]
        initial = state["initial_validation"]
    else:
        initial = evaluate(head, bank, bank.indices("validation"), device, args.batch_size)
    del state
    initial_kl = initial["by_bucket"]["all"]["kl_T1"]
    if not (folder/"best.pt").exists():
        save_checkpoint(folder/"best.pt", dict(model=name, config=vars(config), plan=plan,
                                               head=head.state_dict(), step=0, validation=initial))
    order = bank.indices("train")
    order = np.random.RandomState(args.seed).permutation(order)
    atomic_json(args.run_dir/"status.json", dict(status="running", model=name, cursor=cursor,
                steps=steps, initial_validation_kl_T1=initial_kl, pid=os.getpid(), at=utc(), **status))
    print(json.dumps(dict(event="model_started", model=name, initial_kl_T1=initial_kl,
                          cursor=cursor, train_rows=len(order)), ensure_ascii=False), flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    batch = teacher = None
    while cursor < len(order):
        if time.monotonic()-run_start >= args.max_wall_seconds:
            raise TimeoutError("pilot wall-time budget reached; last checkpoint retained")
        selected = order[cursor:cursor+args.batch_size]
        batch, teacher = bank.batch(selected, device)
        loss, norm = update(head, optimizer, batch, teacher)
        cursor += len(selected)
        steps += 1
        loss_sum += loss*len(selected)
        nseen += len(selected)
        if steps % args.checkpoint_every == 0 or cursor == len(order):
            save_checkpoint(last_path, dict(model=name, plan=plan, config=vars(config),
                head=head.state_dict(), optimizer=optimizer.state_dict(), cursor=cursor,
                steps=steps, loss_sum=loss_sum, nseen=nseen, initial_validation=initial))
        if steps % 16 == 0 or cursor == len(order):
            progress = dict(status="running", model=name, cursor=cursor, steps=steps,
                            total_train_rows=len(order), mean_train_kl_T1=loss_sum/nseen,
                            last_gradient_norm=norm, pid=os.getpid(), at=utc(), **status)
            atomic_json(args.run_dir/"status.json", progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
    final = evaluate(head, bank, bank.indices("validation"), device, args.batch_size)
    final_kl = final["by_bucket"]["all"]["kl_T1"]
    if final_kl < initial_kl:
        save_checkpoint(folder/"best.pt", dict(model=name, config=vars(config), plan=plan,
                                               head=head.state_dict(), step=steps, validation=final))
    elapsed = time.monotonic()-started
    result = dict(model=name, status="complete", optimizer_steps=steps, examples_seen=nseen,
                  epochs=1, initial_validation=initial, final_validation=final,
                  final_minus_initial_kl_T1=final_kl-initial_kl,
                  mean_training_kl_T1=loss_sum/nseen, best_step=steps if final_kl < initial_kl else 0,
                  parameter_count=sum(p.numel() for p in head.parameters()),
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                  peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
                  train_and_final_validation_wall_seconds=elapsed,
                  checkpoint_last=str(last_path), checkpoint_best=str(folder/"best.pt"), at=utc())
    atomic_json(result_path, result)
    print(json.dumps(dict(event="model_complete", model=name, initial_kl_T1=initial_kl,
                          final_kl_T1=final_kl, steps=steps), ensure_ascii=False), flush=True)
    del head, optimizer, batch, teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-output", type=Path, default=ROOT.parent/"dreamon-context-foundation-20261015-v1"/"analysis_outputs/dreamon_context_foundation_20261015_v1")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit-gib", type=float, default=8.)
    parser.add_argument("--checkpoint-every", type=int, default=64)
    parser.add_argument("--max-wall-seconds", type=float, default=3600.)
    args = parser.parse_args()
    if args.batch_size < 1 or args.checkpoint_every < 1 or args.max_wall_seconds <= 0:
        raise ValueError("invalid pilot schedule")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.run_dir/"summary.json"
    if not args.prepare_only and summary_path.exists():
        if json.loads(summary_path.read_text())["status"] == "complete":
            print("pilot already complete; no GPU initialized or result overwritten", flush=True)
            return
        raise ValueError("existing non-complete summary needs inspection")
    torch.set_num_threads(args.threads)
    manifest = prepare(args.a_output, args.run_dir/"data")
    if args.prepare_only:
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
        return
    # User authorized sharing physical GPU0, single worker. No other device used.
    device = torch.device(args.device)
    if device.type == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise ValueError("this pilot is authorized only for visible physical GPU0")
        free, total = torch.cuda.mem_get_info(device)
        if free < args.memory_limit_gib*(1024**3):
            raise RuntimeError("not enough free GPU memory; no other process touched")
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib*(1024**3)/total, device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    config_dict = dict(manifest["config"])
    config_dict["special_token_ids"] = tuple(config_dict["special_token_ids"])
    config = HeadConfig(**config_dict)
    plan = dict(contract_id=manifest["contract_id"], schema_version=1, models=list(MODEL_NAMES),
                epochs=1, batch_size=args.batch_size, learning_rate=args.learning_rate,
                training_seed=args.seed, initialization_seed=config.seed, rank=256, lambda_value=1.,
                objective="T1 full-vocabulary KL(teacher||softmax(old_logits+delta)); no CE/DCD",
                optimizer="AdamW betas(.9,.999), eps1e-8, weight_decay0, grad_clip1",
                data_source_sha256=manifest["source_sha256"])
    old_plan = args.run_dir/"plan.json"
    if old_plan.exists() and json.loads(old_plan.read_text()) != plan:
        raise ValueError("run directory belongs to a different pilot plan")
    atomic_json(old_plan, plan)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    common = dict(source_commit=commit, device=args.device, models=list(MODEL_NAMES))
    environment = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                       CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),
                       CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                       gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       cpu_threads=torch.get_num_threads(), pid=os.getpid(), created_at=utc(),
                       memory_limit_gib=args.memory_limit_gib, shared_gpu_authorized=True,
                       backbone_loaded=False, teacher_collection_started=False, human_eval_answers_generated=False)
    atomic_json(args.run_dir/"environment.json", environment)
    bank = PackedBank(args.run_dir/"data")
    start = time.monotonic()
    try:
        stale_path = args.run_dir/"stale_validation.json"
        if not stale_path.exists():
            atomic_json(stale_path, evaluate(None, bank, bank.indices("validation"), device, args.batch_size))
        results = [train_one(name, config, bank, plan, args, common, start) for name in MODEL_NAMES]
        summary = dict(status="complete", results=results, plan=plan, environment=environment,
                       stale_validation=json.loads(stale_path.read_text()),
                       worker_wall_seconds=time.monotonic()-start, at=utc(), **common)
        atomic_json(args.run_dir/"summary.json", summary)
        atomic_json(args.run_dir/"status.json", dict(status="complete", at=utc(), pid=os.getpid(), **common))
    except Exception as error:
        atomic_json(args.run_dir/"status.json", dict(status="stopped", reason=repr(error), at=utc(),
                    pid=os.getpid(), checkpoint_scope="last completed checkpoint; deterministic prefix replay on resume", **common))
        raise


if __name__ == "__main__":
    main()
