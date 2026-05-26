#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import CodeTask, compute_oracle_mask_length, infer_reference_middle_text, load_humaneval_infilling
from expvision_dllm_clean.decode import build_reconstruction_diagnostics
from expvision_dllm_clean.length_probe import adjust_length_probe_score, parse_probe_lengths
from expvision_dllm_clean.logging import JsonlLogger
from expvision_dllm_clean.modeling import get_torch_dtype, set_global_seed
from expvision_dllm_clean.verifier import run_verifier_stack


FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Dream-Coder on HumanEval infilling using its native FIM prompt format "
            "and model.diffusion_generate instead of the LLaDA logits/remask loop."
        )
    )
    parser.add_argument("--model-path", type=str, default="Dream-org/Dream-Coder-v0-Base-7B")
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="fim",
        choices=["fim", "chat_fill"],
        help=(
            "fim uses Dream/Qwen-style FIM tokens. chat_fill uses the tokenizer chat template "
            "and asks an instruct model to return only the missing span."
        ),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mask-length-source", type=str, default="cal_lite", choices=["fixed", "oracle", "cal_lite"])
    parser.add_argument("--fixed-mask-length", type=int, default=16)
    parser.add_argument(
        "--probe-lengths",
        type=str,
        default="3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24",
    )
    parser.add_argument("--tie-break", type=str, default="shorter", choices=["shorter", "longer"])
    parser.add_argument("--score-mode", type=str, default="length_power", choices=["raw", "length_power"])
    parser.add_argument("--length-alpha", type=float, default=0.06)

    parser.add_argument("--dream-steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--alg", type=str, default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--eos-penalty", type=float, default=3.0)

    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="model_generalization_runs/dream_fim_diffusion")
    parser.add_argument("--experiment-name", type=str, default="dream_fim_diffusion")
    return parser.parse_args()


def _avg(values: Iterable[float | int | None]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    return sum(filtered) / len(filtered) if filtered else None


def _rate(values: Iterable[bool]) -> Optional[float]:
    values = list(values)
    return sum(1 for value in values if value) / len(values) if values else None


def _hist(values: Iterable[Any]) -> Dict[str, int]:
    counter = Counter(str(value) for value in values)
    return dict(sorted(counter.items()))


def resolve_model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def resolve_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise ValueError(f"Tokenizer does not know token: {token}")
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == unk_id:
        raise ValueError(f"Tokenizer mapped {token} to unk_token_id")
    return int(token_id)


def build_fim_prompt(task: CodeTask) -> str:
    return f"{FIM_PREFIX}{task.prefix}{FIM_SUFFIX}{task.suffix}{FIM_MIDDLE}"


def build_chat_fill_messages(task: CodeTask) -> List[Dict[str, str]]:
    content = (
        "Complete the code at <FILL_ME>. Return only the exact missing code, "
        "including indentation. Do not include markdown fences or explanations.\n\n"
        f"{task.prefix}<FILL_ME>{task.suffix}"
    )
    return [{"role": "user", "content": content}]


def encode_prompt(task: CodeTask, tokenizer, prompt_mode: str) -> Dict[str, Any]:
    if prompt_mode == "fim":
        prompt = build_fim_prompt(task)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        attention_mask = [1] * len(prompt_ids)
        return {
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "attention_mask": attention_mask,
            "prompt_length": len(prompt_ids),
            "prompt_format": "dream_fim",
        }

    if prompt_mode == "chat_fill":
        messages = build_chat_fill_messages(task)
        encoded = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        prompt_ids = encoded.input_ids[0].tolist()
        attention_mask = encoded.attention_mask[0].tolist()
        return {
            "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=False),
            "messages": messages,
            "prompt_ids": prompt_ids,
            "attention_mask": attention_mask,
            "prompt_length": len(prompt_ids),
            "prompt_format": "dream_chat_fill",
        }

    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def encode_fim_prompt(task: CodeTask, tokenizer) -> Dict[str, Any]:
    prompt = build_fim_prompt(task)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return {
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "attention_mask": [1] * len(prompt_ids),
        "prompt_length": len(prompt_ids),
        "prompt_format": "dream_fim",
    }


def shifted_logits_for_dream(model, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if attention_mask is not None and torch.all(attention_mask == 1):
        attention_mask = None
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def probe_length_score(
    task: CodeTask,
    tokenizer,
    model,
    cfg: ExperimentConfig,
    mask_length: int,
    prompt_mode: str,
) -> Dict[str, Any]:
    encoded = encode_prompt(task, tokenizer, prompt_mode)
    mask_token_id = resolve_token_id(tokenizer, "<|mask|>")
    input_ids = encoded["prompt_ids"] + [mask_token_id] * mask_length
    attention_ids = encoded["attention_mask"] + [1] * mask_length
    device = resolve_model_device(model)
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([attention_ids], dtype=torch.long, device=device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = shifted_logits_for_dream(model, tensor, attention_mask=attention_mask)
    probe_sec = time.perf_counter() - start

    middle_logits = logits[:, -mask_length:, :]
    probs = torch.softmax(middle_logits, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1)
    top2_values, _ = torch.topk(probs, k=2, dim=-1)

    raw_score = float(max_probs.mean().item())
    adjusted_score = adjust_length_probe_score(raw_score, mask_length, cfg)
    mean_top2_gap = float((top2_values[..., 0] - top2_values[..., 1]).mean().item())

    return {
        "mask_length": int(mask_length),
        "score": float(adjusted_score),
        "raw_score": raw_score,
        "adjusted_score": float(adjusted_score),
        "mean_top1_prob": raw_score,
        "mean_top2_gap": mean_top2_gap,
        "score_mode": cfg.decode.cal_lite_score_mode,
        "length_alpha": float(cfg.decode.cal_lite_length_alpha),
        "probe_sec": probe_sec,
        "prompt_format": encoded["prompt_format"],
        "logit_alignment": "dream_shifted_logits",
    }


def pick_best_candidate(candidates: List[Dict[str, Any]], tie_break: str) -> Dict[str, Any]:
    if tie_break == "shorter":
        return sorted(candidates, key=lambda item: (-float(item["score"]), int(item["mask_length"])))[0]
    if tie_break == "longer":
        return sorted(candidates, key=lambda item: (-float(item["score"]), -int(item["mask_length"])))[0]
    raise ValueError(f"Unsupported tie_break: {tie_break}")


def length_diff(selected_length: Optional[int], oracle_length: Optional[int]) -> Dict[str, Optional[int]]:
    if selected_length is None or oracle_length is None:
        return {"selected_minus_oracle_length": None, "abs_selected_minus_oracle_length": None}
    diff = int(selected_length) - int(oracle_length)
    return {"selected_minus_oracle_length": diff, "abs_selected_minus_oracle_length": abs(diff)}


def resolve_mask_length(task: CodeTask, tokenizer, model, cfg: ExperimentConfig, prompt_mode: str) -> Dict[str, Any]:
    oracle = compute_oracle_mask_length(task, tokenizer, add_special_tokens=False)
    oracle = None if oracle is None else int(oracle)
    source = cfg.decode.mask_length_source

    base = {
        "oracle_mask_length": oracle,
        "mask_length_source": source,
        "selected_mask_length": None,
        "selected_score": None,
        "selected_raw_score": None,
        "selected_adjusted_score": None,
        "candidate_scores": None,
        "length_probe_sec": 0.0,
        "probe_lengths": None,
        "tie_break": None,
        "score_mode": None,
        "length_alpha": None,
    }

    if source == "fixed":
        selected = int(cfg.decode.fixed_mask_length)
        return {**base, "mask_length": selected, "selected_mask_length": selected, **length_diff(selected, oracle)}

    if source == "oracle":
        if oracle is None:
            raise ValueError(f"Oracle mask length unavailable for task {task.task_id}")
        selected = int(oracle)
        return {**base, "mask_length": selected, "selected_mask_length": selected, **length_diff(selected, oracle)}

    if source == "cal_lite":
        candidates: List[Dict[str, Any]] = []
        total_probe_sec = 0.0
        for mask_length in parse_probe_lengths(cfg.decode.cal_lite_probe_lengths_csv):
            score = probe_length_score(task, tokenizer, model, cfg, mask_length, prompt_mode)
            candidates.append(score)
            total_probe_sec += float(score["probe_sec"])
        best = pick_best_candidate(candidates, cfg.decode.cal_lite_tie_break)
        selected = int(best["mask_length"])
        return {
            **base,
            "mask_length": selected,
            "selected_mask_length": selected,
            "selected_score": float(best["score"]),
            "selected_raw_score": float(best["raw_score"]),
            "selected_adjusted_score": float(best["adjusted_score"]),
            "candidate_scores": candidates,
            "length_probe_sec": total_probe_sec,
            "probe_lengths": [int(item["mask_length"]) for item in candidates],
            "tie_break": cfg.decode.cal_lite_tie_break,
            "score_mode": cfg.decode.cal_lite_score_mode,
            "length_alpha": float(cfg.decode.cal_lite_length_alpha),
            **length_diff(selected, oracle),
        }

    raise ValueError(f"Unsupported mask length source: {source}")


def clean_generated_middle(text: str, tokenizer) -> str:
    stops = [
        tokenizer.eos_token,
        "<|endoftext|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|fim_prefix|>",
        "<|fim_pad|>",
        "<|im_end|>",
        "<|mask|>",
    ]
    cleaned = text
    for stop in stops:
        if stop and stop in cleaned:
            cleaned = cleaned.split(stop, 1)[0]
    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
            if cleaned.lstrip().startswith("python"):
                cleaned = cleaned.lstrip()[len("python") :]
                if cleaned.startswith("\n"):
                    cleaned = cleaned[1:]
    return cleaned


def strip_markdown_code_fence(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith("```"):
        return text

    lines = stripped.splitlines()
    if not lines:
        return text
    body = "\n".join(lines[1:])
    if "```" in body:
        body = body.split("```", 1)[0]
    return body


def recover_middle_from_generation(raw_text: str, task: CodeTask, tokenizer) -> Dict[str, Any]:
    candidates = [
        raw_text,
        clean_generated_middle(raw_text, tokenizer),
        strip_markdown_code_fence(raw_text),
        strip_markdown_code_fence(clean_generated_middle(raw_text, tokenizer)),
    ]

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if task.prefix in candidate and task.suffix in candidate:
            after_prefix = candidate.split(task.prefix, 1)[1]
            middle = after_prefix.split(task.suffix, 1)[0]
            return {
                "middle_text": middle,
                "postprocess": "slice_between_prefix_suffix",
                "postprocess_candidate": candidate,
            }

    return {
        "middle_text": clean_generated_middle(raw_text, tokenizer),
        "postprocess": "direct_generated_middle",
        "postprocess_candidate": None,
    }


def build_diagnostics(task: CodeTask, prefix_text: str, middle_text: str, suffix_text: str, prompt_format: str) -> Dict[str, Any]:
    return {
        **build_reconstruction_diagnostics(
            task,
            {
                "prefix_text": prefix_text,
                "middle_text": middle_text,
                "suffix_text": suffix_text,
                "full_text": prefix_text + middle_text + suffix_text,
            },
        ),
        "prompt_format": prompt_format,
    }


def run_task(task: CodeTask, tokenizer, model, cfg: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    length_start = time.perf_counter()
    length_meta = resolve_mask_length(task, tokenizer, model, cfg, args.prompt_mode)
    length_probe_sec = float(length_meta["length_probe_sec"])
    encoded = encode_prompt(task, tokenizer, args.prompt_mode)

    device = resolve_model_device(model)
    input_ids = torch.tensor([encoded["prompt_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long, device=device)

    selected_len = int(length_meta["mask_length"])
    mask_token_id = resolve_token_id(tokenizer, "<|mask|>")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id
    bos_token_id = tokenizer.bos_token_id

    decode_start = time.perf_counter()
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=selected_len,
            steps=int(args.dream_steps),
            temperature=float(args.temperature),
            top_p=None if args.top_p is None else float(args.top_p),
            top_k=args.top_k,
            alg=args.alg,
            alg_temp=args.alg_temp,
            eos_penalty=float(args.eos_penalty),
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            return_dict_in_generate=True,
            output_history=False,
        )
    decode_sec = time.perf_counter() - decode_start

    generated_ids = output.sequences[0, encoded["prompt_length"] :].tolist()
    raw_middle = tokenizer.decode(generated_ids, skip_special_tokens=False)
    recovered = recover_middle_from_generation(raw_middle, task, tokenizer)
    middle_text = recovered["middle_text"]
    full_text = task.prefix + middle_text + task.suffix

    verification = run_verifier_stack(task, full_text, middle_text)
    verification_sec = sum(item.duration_sec for item in verification.values())
    total_sec = decode_sec + verification_sec
    total_sec_including_probe = total_sec + length_probe_sec
    tier3 = verification.get("tier3_unit_tests")
    passed = bool(tier3.passed) if tier3 else False

    diagnostics = build_diagnostics(task, task.prefix, middle_text, task.suffix, encoded["prompt_format"])
    diagnostics.update(
        {
            "raw_generated_middle_text": raw_middle,
            "postprocess": recovered["postprocess"],
            "postprocess_candidate": recovered["postprocess_candidate"],
            "generated_token_count": len(generated_ids),
            "prompt_token_count": int(encoded["prompt_length"]),
            "fim_prompt": encoded["prompt"],
            "prompt_mode": args.prompt_mode,
            "prompt_format": encoded["prompt_format"],
            "reference_middle_text": infer_reference_middle_text(task),
        }
    )

    return {
        "task_id": task.task_id,
        "dataset_subset": cfg.data.dataset_subset,
        "metrics": {
            "passed": passed,
            "decode_sec": decode_sec,
            "verification_sec": verification_sec,
            "total_sec": total_sec,
            "total_sec_including_probe": total_sec_including_probe,
            "length_probe_sec": length_probe_sec,
            "total_steps": int(args.dream_steps),
            "effective_steps": int(args.dream_steps),
            "stopped": False,
            "stop_step": None,
            "stop_reason": "dream_diffusion_generate_fixed_canvas",
            "mask_length": selected_len,
            "oracle_mask_length": length_meta["oracle_mask_length"],
            "selected_mask_length": selected_len,
            "selected_score": length_meta["selected_score"],
            "selected_raw_score": length_meta["selected_raw_score"],
            "selected_adjusted_score": length_meta["selected_adjusted_score"],
            "selected_minus_oracle_length": length_meta["selected_minus_oracle_length"],
            "abs_selected_minus_oracle_length": length_meta["abs_selected_minus_oracle_length"],
            "mask_length_source": length_meta["mask_length_source"],
            "prompt_format": encoded["prompt_format"],
            "decode_backend": "dream_diffusion_generate",
            "dream_steps": int(args.dream_steps),
            "dream_alg": args.alg,
            "dream_temperature": float(args.temperature),
            "dream_top_p": args.top_p,
            "dream_top_k": args.top_k,
            "dream_alg_temp": args.alg_temp,
            "dream_eos_penalty": float(args.eos_penalty),
        },
        "verification": {key: value.to_dict() for key, value in verification.items()},
        "length_probe": {
            "candidate_scores": length_meta["candidate_scores"],
            "length_probe_sec": length_probe_sec,
            "selected_mask_length": selected_len,
            "selected_score": length_meta["selected_score"],
            "selected_raw_score": length_meta["selected_raw_score"],
            "selected_adjusted_score": length_meta["selected_adjusted_score"],
            "selected_minus_oracle_length": length_meta["selected_minus_oracle_length"],
            "abs_selected_minus_oracle_length": length_meta["abs_selected_minus_oracle_length"],
            "probe_lengths": length_meta["probe_lengths"],
            "tie_break": length_meta["tie_break"],
            "score_mode": length_meta["score_mode"],
            "length_alpha": length_meta["length_alpha"],
            "prompt_format": encoded["prompt_format"],
            "logit_alignment": "dream_shifted_logits",
        },
        "stopping": {
            "enabled": False,
            "method": "dream_native_diffusion_generate",
            "policy": None,
            "stopped": False,
            "stop_step": None,
            "stop_reason": "dream_diffusion_generate_fixed_canvas",
            "effective_steps": int(args.dream_steps),
            "config": {
                "steps": int(args.dream_steps),
                "alg": args.alg,
                "temperature": float(args.temperature),
                "top_p": args.top_p,
                "top_k": args.top_k,
                "alg_temp": args.alg_temp,
                "eos_penalty": float(args.eos_penalty),
            },
        },
        "code": full_text,
        "diagnostics": diagnostics,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {"num_samples": 0}
    metrics = [row["metrics"] for row in results]
    num_samples = len(results)
    return {
        "num_samples": num_samples,
        "pass_rate": sum(1 for row in results if row["metrics"]["passed"]) / num_samples,
        "avg_decode_sec": _avg(m.get("decode_sec") for m in metrics),
        "avg_verification_sec": _avg(m.get("verification_sec") for m in metrics),
        "avg_total_sec": _avg(m.get("total_sec") for m in metrics),
        "avg_length_probe_sec": _avg(m.get("length_probe_sec") for m in metrics),
        "avg_total_sec_including_probe": _avg(m.get("total_sec_including_probe") for m in metrics),
        "avg_selected_mask_length": _avg(m.get("selected_mask_length") for m in metrics),
        "avg_oracle_mask_length": _avg(m.get("oracle_mask_length") for m in metrics),
        "avg_selected_minus_oracle_length": _avg(m.get("selected_minus_oracle_length") for m in metrics),
        "avg_abs_selected_minus_oracle_length": _avg(m.get("abs_selected_minus_oracle_length") for m in metrics),
        "under_select_rate": _rate(
            m.get("selected_minus_oracle_length") is not None and m.get("selected_minus_oracle_length") < 0
            for m in metrics
            if m.get("selected_minus_oracle_length") is not None
        ),
        "over_select_rate": _rate(
            m.get("selected_minus_oracle_length") is not None and m.get("selected_minus_oracle_length") > 0
            for m in metrics
            if m.get("selected_minus_oracle_length") is not None
        ),
        "exact_length_match_rate": _rate(
            m.get("selected_minus_oracle_length") == 0 for m in metrics if m.get("selected_minus_oracle_length") is not None
        ),
        "selected_length_histogram": _hist(m.get("selected_mask_length") for m in metrics),
        "oracle_length_histogram": _hist(m.get("oracle_mask_length") for m in metrics),
        "prompt_formats": sorted({str(m.get("prompt_format")) for m in metrics}),
        "decode_backends": sorted({str(m.get("decode_backend")) for m in metrics}),
        "dream_algs": sorted({str(m.get("dream_alg")) for m in metrics}),
        "dream_temperatures": sorted({float(m.get("dream_temperature")) for m in metrics}),
        "dream_eos_penalties": sorted({float(m.get("dream_eos_penalty")) for m in metrics}),
        "determinism_note": (
            "Dream-Coder native diffusion_generate is stochastic when temperature > 0. "
            "Use temperature=0 for deterministic smoke, or run multiple seeds for reported stochastic settings."
        ),
    }


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.model_path = args.model_path
    cfg.model.torch_dtype = args.torch_dtype
    cfg.model.device_map = args.device_map
    cfg.data.split = args.split
    cfg.data.dataset_subset = args.dataset_subset
    cfg.data.max_samples = args.max_samples
    cfg.decode.mask_length_source = args.mask_length_source
    cfg.decode.fixed_mask_length = args.fixed_mask_length
    cfg.decode.seed = args.seed
    cfg.decode.total_steps = args.dream_steps
    cfg.decode.cal_lite_probe_lengths_csv = args.probe_lengths
    cfg.decode.cal_lite_tie_break = args.tie_break
    cfg.decode.cal_lite_score_mode = args.score_mode
    cfg.decode.cal_lite_length_alpha = args.length_alpha
    cfg.logging.output_dir = args.output_dir
    cfg.logging.experiment_name = args.experiment_name
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    set_global_seed(args.seed)

    from transformers import AutoModel, AutoTokenizer

    logger = JsonlLogger(cfg.logging.output_dir, cfg.logging.experiment_name)
    config_payload = cfg.to_dict()
    config_payload["dream_generation"] = {
        "prompt_mode": args.prompt_mode,
        "decode_backend": "dream_diffusion_generate",
        "dream_steps": args.dream_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "alg": args.alg,
        "alg_temp": args.alg_temp,
        "eos_penalty": args.eos_penalty,
    }
    logger.save_config(config_payload)

    print("=" * 80, flush=True)
    print("Starting Dream FIM diffusion experiment", flush=True)
    print(f"experiment_name = {cfg.logging.experiment_name}", flush=True)
    print(f"output_dir       = {cfg.logging.output_dir}", flush=True)
    print(f"model_path       = {cfg.model.model_path}", flush=True)
    print(f"prompt_mode      = {args.prompt_mode}", flush=True)
    print(f"dataset_subset   = {cfg.data.dataset_subset}", flush=True)
    print(f"max_samples      = {cfg.data.max_samples}", flush=True)
    print(f"mask_length_src  = {cfg.decode.mask_length_source}", flush=True)
    print(f"probe_lengths    = {cfg.decode.cal_lite_probe_lengths_csv}", flush=True)
    print(f"dream_steps      = {args.dream_steps}", flush=True)
    print(f"temperature      = {args.temperature}", flush=True)
    print(f"top_p            = {args.top_p}", flush=True)
    print(f"alg              = {args.alg}", flush=True)
    print(f"eos_penalty      = {args.eos_penalty}", flush=True)
    print(f"seed             = {args.seed}", flush=True)
    print("=" * 80, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        cfg.model.model_path,
        trust_remote_code=True,
        torch_dtype=get_torch_dtype(cfg.model.torch_dtype),
        device_map=cfg.model.device_map,
    )
    model.eval()

    tasks = load_humaneval_infilling(
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
        dataset_subset=cfg.data.dataset_subset,
    )
    print(f"Loaded {len(tasks)} tasks", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("-" * 80, flush=True)

    results: List[Dict[str, Any]] = []
    total = len(tasks)
    for idx, task in enumerate(tasks, start=1):
        print(f"[{idx}/{total}] task_id={task.task_id} | start", flush=True)
        result = run_task(task, tokenizer, model, cfg, args)
        results.append(result)
        logger.log_result(result)
        m = result["metrics"]
        print(
            f"[{idx}/{total}] task_id={task.task_id} | "
            f"{'PASS' if m['passed'] else 'FAIL'} | "
            f"total_sec={m['total_sec']:.3f} | "
            f"total_sec_including_probe={m['total_sec_including_probe']:.3f} | "
            f"mask_len={m['mask_length']} | "
            f"oracle_mask_len={m['oracle_mask_length']} | "
            f"diff={m['selected_minus_oracle_length']} | "
            f"backend={m['decode_backend']}",
            flush=True,
        )

    summary = summarize(results)
    logger.save_json("summary.json", summary)

    print("-" * 80, flush=True)
    print("Dream FIM diffusion experiment finished", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
