#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from expvision_dllm_clean.config import ExperimentConfig
from expvision_dllm_clean.dataset import (
    CodeTask,
    compute_oracle_mask_length,
    infer_reference_middle_text,
    load_humaneval_infilling,
)
from expvision_dllm_clean.decode import build_reconstruction_diagnostics
from expvision_dllm_clean.length_probe import adjust_length_probe_score, parse_probe_lengths
from expvision_dllm_clean.logging import JsonlLogger
from expvision_dllm_clean.modeling import get_torch_dtype, set_global_seed
from expvision_dllm_clean.verifier import run_verifier_stack

DEFAULT_COMPACT_GRID = "3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24"
DEFAULT_WEAK_GRID = "13,14,15,16"
DEFAULT_STRONG_GRID = "13,14,15,16,20,24,28,32,40"
DEFAULT_RATIO_TRIGGER_THRESHOLD = 0.97
DEFAULT_LONG_SCORE_FLOOR = 0.55
DEFAULT_RAW_RATIO_THRESHOLD = 0.97
DEFAULT_SUPPORT_COUNT_THRESHOLD = 2
DEFAULT_BIAS_PARAMS = "1.0,1.77,0.56,0.06,0.24"


def ensure_modeling_rope_utils_available() -> None:
    import transformers.utils as transformers_utils

    if os.environ.get("DLLM_DISABLE_FLASH_ATTN", "0").strip().lower() in {"1", "true", "yes", "on"}:
        def _flash_attn_unavailable(*_args, **_kwargs) -> bool:
            return False

        transformers_utils.is_flash_attn_2_available = _flash_attn_unavailable  # type: ignore[attr-defined]
        transformers_utils.is_flash_attn_greater_or_equal = _flash_attn_unavailable  # type: ignore[attr-defined]
        transformers_utils.is_flash_attn_greater_or_equal_2_10 = _flash_attn_unavailable  # type: ignore[attr-defined]
        try:
            import transformers.utils.import_utils as transformers_import_utils

            transformers_import_utils.is_flash_attn_2_available = _flash_attn_unavailable  # type: ignore[attr-defined]
            transformers_import_utils.is_flash_attn_greater_or_equal = _flash_attn_unavailable  # type: ignore[attr-defined]
            transformers_import_utils.is_flash_attn_greater_or_equal_2_10 = _flash_attn_unavailable  # type: ignore[attr-defined]
        except Exception:
            pass

    if not hasattr(transformers_utils, "is_torchdynamo_compiling"):
        def _is_torchdynamo_compiling() -> bool:
            try:
                import torch

                compiler = getattr(torch, "compiler", None)
                if compiler is not None and hasattr(compiler, "is_compiling"):
                    return bool(compiler.is_compiling())
                dynamo = getattr(torch, "_dynamo", None)
                return bool(dynamo is not None and hasattr(dynamo, "is_compiling") and dynamo.is_compiling())
            except Exception:
                return False

        transformers_utils.is_torchdynamo_compiling = _is_torchdynamo_compiling  # type: ignore[attr-defined]

    try:
        import transformers.modeling_rope_utils  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    candidates = [
        os.environ.get("DLLM_MODELING_ROPE_UTILS_PATH", ""),
        "/home/shx/miniconda3/envs/llmxy/lib/python3.10/site-packages/transformers/modeling_rope_utils.py",
    ]
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        spec = importlib.util.spec_from_file_location("transformers.modeling_rope_utils", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["transformers.modeling_rope_utils"] = module
        spec.loader.exec_module(module)
        return

    raise ModuleNotFoundError(
        "transformers.modeling_rope_utils is unavailable; set DLLM_MODELING_ROPE_UTILS_PATH "
        "or use an environment with a newer Transformers package."
    )


@dataclass
class DreamBoundedRepairSettings:
    base_probe_lengths_csv: str
    base_alpha: float
    weak_probe_lengths_csv: str
    strong_probe_lengths_csv: str
    long_alpha: float
    strong_min_len: int
    weak_min_base_len: int
    weak_max_base_len: int
    ratio_trigger_threshold: float
    long_score_floor: float
    raw_ratio_threshold: float
    support_count_threshold: int
    cap_base_le8: int
    cap_base_9_12: int
    short_safe_policy: str
    correction_selection_rule: str
    shortest_supported_ratio: float
    official_initial_length: int
    official_span: int
    official_max_length: int
    official_dstep: int
    official_use_bias: bool
    official_bias_params: Tuple[float, float, float, float, float]
    official_eval_max_s3_len: Optional[int]
    repair_max_s3_len: int
    repair_min_official_len: int
    repair_max_official_len: int
    repair_min_delta: int
    repair_max_delta: int
    suspicion_max_s3_len: Optional[int]
    suspicion_min_official_len: Optional[int]
    suspicion_max_official_len: int
    suspicion_min_delta: int
    mid_rescue_max_s3_len: Optional[int]
    mid_rescue_min_official_len: Optional[int]
    mid_rescue_max_official_len: int
    mid_rescue_min_delta: int
    mid_rescue_max_delta: int
    mid_rescue_min_long_ratio: Optional[float]
    mid_rescue_source: str


def parse_bias_params(text: str) -> Tuple[float, float, float, float, float]:
    parts = [float(item.strip()) for item in text.split(",") if item.strip()]
    if len(parts) != 5:
        raise ValueError("--official-bias-params must contain five comma-separated floats")
    return tuple(parts)  # type: ignore[return-value]


def length_bias(length: int, params: Tuple[float, float, float, float, float]) -> float:
    a, b, c, d, e = params
    return float(a * math.exp(-b * length) + c * math.exp(-d * length) + e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Dream-Coder with its native fixed-canvas infilling interface: "
            "BOS + prefix + <|mask|>*N + suffix + EOS, then slice the filled middle span."
        )
    )
    parser.add_argument("--model-path", type=str, default="Dream-org/Dream-Coder-v0-Base-7B")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-subset", type=str, default="HumanEval-SingleLineInfilling")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--mask-length-source",
        type=str,
        default="cal_lite",
        choices=["fixed", "oracle", "cal_lite", "lcal_official_bounded_repair"],
    )
    parser.add_argument("--fixed-mask-length", type=int, default=16)
    parser.add_argument(
        "--probe-lengths",
        type=str,
        default="3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,32,48,64",
    )
    parser.add_argument("--tie-break", type=str, default="shorter", choices=["shorter", "longer"])
    parser.add_argument("--score-mode", type=str, default="length_power", choices=["raw", "length_power"])
    parser.add_argument("--length-alpha", type=float, default=0.10)

    parser.add_argument("--base-probe-lengths", type=str, default=DEFAULT_COMPACT_GRID)
    parser.add_argument("--base-alpha", type=float, default=0.06)
    parser.add_argument("--weak-probe-lengths", type=str, default=DEFAULT_WEAK_GRID)
    parser.add_argument("--strong-probe-lengths", type=str, default=DEFAULT_STRONG_GRID)
    parser.add_argument("--long-alpha", type=float, default=0.10)
    parser.add_argument("--strong-min-len", type=int, default=13)
    parser.add_argument("--weak-min-base-len", type=int, default=8)
    parser.add_argument("--weak-max-base-len", type=int, default=12)
    parser.add_argument("--ratio-trigger-threshold", type=float, default=DEFAULT_RATIO_TRIGGER_THRESHOLD)
    parser.add_argument("--long-score-floor", type=float, default=DEFAULT_LONG_SCORE_FLOOR)
    parser.add_argument("--raw-ratio-threshold", type=float, default=DEFAULT_RAW_RATIO_THRESHOLD)
    parser.add_argument("--support-count-threshold", type=int, default=DEFAULT_SUPPORT_COUNT_THRESHOLD)
    parser.add_argument("--cap-base-le8", type=int, default=14)
    parser.add_argument("--cap-base-9-12", type=int, default=16)
    parser.add_argument("--short-safe-policy", type=str, default="s3", choices=["s1", "s2", "s3", "s4"])
    parser.add_argument(
        "--correction-selection-rule",
        type=str,
        default="shortest_supported",
        choices=["argmax", "shortest_supported"],
    )
    parser.add_argument("--shortest-supported-ratio", type=float, default=0.985)

    parser.add_argument("--official-initial-length", type=int, default=8)
    parser.add_argument("--official-span", type=int, default=1)
    parser.add_argument("--official-max-length", type=int, default=64)
    parser.add_argument("--official-dstep", type=int, default=4)
    parser.add_argument("--official-no-bias", action="store_true")
    parser.add_argument("--official-bias-params", type=str, default=DEFAULT_BIAS_PARAMS)
    parser.add_argument("--official-eval-max-s3-len", type=int, default=None)
    parser.add_argument("--repair-max-s3-len", type=int, default=5)
    parser.add_argument("--repair-min-official-len", type=int, default=6)
    parser.add_argument("--repair-max-official-len", type=int, default=11)
    parser.add_argument("--repair-min-delta", type=int, default=1)
    parser.add_argument("--repair-max-delta", type=int, default=8)
    parser.add_argument("--suspicion-max-s3-len", type=int, default=None)
    parser.add_argument("--suspicion-min-official-len", type=int, default=None)
    parser.add_argument("--suspicion-max-official-len", type=int, default=64)
    parser.add_argument("--suspicion-min-delta", type=int, default=1)
    parser.add_argument("--mid-rescue-max-s3-len", type=int, default=None)
    parser.add_argument("--mid-rescue-min-official-len", type=int, default=None)
    parser.add_argument("--mid-rescue-max-official-len", type=int, default=13)
    parser.add_argument("--mid-rescue-min-delta", type=int, default=3)
    parser.add_argument("--mid-rescue-max-delta", type=int, default=7)
    parser.add_argument("--mid-rescue-min-long-ratio", type=float, default=None)
    parser.add_argument("--mid-rescue-source", type=str, default="base")

    parser.add_argument("--dream-steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--alg", type=str, default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--eos-penalty", type=float, default=3.0)
    parser.add_argument(
        "--right-pad-new-tokens",
        type=int,
        default=1,
        help=(
            "Dream-Coder remote code requires max_length > input length. We add this many right-pad "
            "positions and force them to EOS so only the existing middle masks are filled."
        ),
    )
    parser.add_argument("--no-force-right-pad-eos", action="store_true")
    parser.add_argument("--no-bos", action="store_true")
    parser.add_argument("--no-eos", action="store_true")

    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="model_generalization_runs/20260513_dreamcoder_official_infilling")
    parser.add_argument("--experiment-name", type=str, default="dreamcoder_official_infilling")
    parser.add_argument("--baseline-results", type=str, default=None)
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


def resolve_special_ids(tokenizer) -> Dict[str, int]:
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = resolve_token_id(tokenizer, "<|mask|>")

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = resolve_token_id(tokenizer, "<|endoftext|>")

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = resolve_token_id(tokenizer, "<|beginoftext|>")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id

    return {
        "mask_token_id": int(mask_token_id),
        "eos_token_id": int(eos_token_id),
        "bos_token_id": int(bos_token_id),
        "pad_token_id": int(pad_token_id),
    }


def build_canvas(task: CodeTask, tokenizer, mask_length: int, args: argparse.Namespace) -> Dict[str, Any]:
    if mask_length <= 0:
        raise ValueError(f"mask_length must be positive, got {mask_length}")

    ids = resolve_special_ids(tokenizer)
    prefix_ids = tokenizer.encode(task.prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(task.suffix, add_special_tokens=False)

    if not args.no_bos:
        prefix_ids = [ids["bos_token_id"]] + prefix_ids
    if not args.no_eos:
        suffix_ids = suffix_ids + [ids["eos_token_id"]]

    middle_start = len(prefix_ids)
    middle_end = middle_start + mask_length
    input_ids = prefix_ids + [ids["mask_token_id"]] * mask_length + suffix_ids

    return {
        **ids,
        "prefix_ids": prefix_ids,
        "suffix_ids": suffix_ids,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "middle_start": middle_start,
        "middle_end": middle_end,
        "mask_length": int(mask_length),
        "input_length": len(input_ids),
        "use_bos": not args.no_bos,
        "use_eos": not args.no_eos,
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
    args: argparse.Namespace,
    mask_length: int,
) -> Dict[str, Any]:
    canvas = build_canvas(task, tokenizer, mask_length, args)
    device = resolve_model_device(model)
    input_tensor = torch.tensor([canvas["input_ids"]], dtype=torch.long, device=device)
    attention_tensor = torch.tensor([canvas["attention_mask"]], dtype=torch.long, device=device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = shifted_logits_for_dream(model, input_tensor, attention_mask=attention_tensor)
    probe_sec = time.perf_counter() - start

    middle_logits = logits[:, canvas["middle_start"] : canvas["middle_end"], :]
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
        "probe_sec": float(probe_sec),
        "logit_alignment": "dream_shifted_logits_middle_canvas",
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


def _candidate_score(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("score", candidate.get("adjusted_score", candidate.get("raw_score"))))


def _raw_score(candidate: Dict[str, Any]) -> float:
    return float(candidate.get("raw_score", candidate.get("mean_top1_prob", candidate.get("score"))))


def _candidate_length(candidate: Dict[str, Any]) -> int:
    return int(candidate["mask_length"])


def _best_candidate(candidates: List[Dict[str, Any]], tie_break: str, score_fn=_candidate_score) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("Cannot select from an empty candidate list")
    if tie_break == "shorter":
        return sorted(candidates, key=lambda item: (-score_fn(item), _candidate_length(item)))[0]
    if tie_break == "longer":
        return sorted(candidates, key=lambda item: (-score_fn(item), -_candidate_length(item)))[0]
    raise ValueError(f"Unsupported tie_break: {tie_break}")


def _clone_probe_cfg(cfg: ExperimentConfig, probe_lengths_csv: str, alpha: float, args: argparse.Namespace) -> ExperimentConfig:
    probe_cfg = copy.deepcopy(cfg)
    probe_cfg.decode.cal_lite_probe_lengths_csv = probe_lengths_csv
    probe_cfg.decode.cal_lite_length_alpha = float(alpha)
    probe_cfg.decode.cal_lite_tie_break = args.tie_break
    probe_cfg.decode.cal_lite_score_mode = args.score_mode
    return probe_cfg


def select_dream_cal_lite(
    task: CodeTask,
    tokenizer,
    model,
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    *,
    probe_lengths_csv: str,
    alpha: float,
) -> Dict[str, Any]:
    probe_cfg = _clone_probe_cfg(cfg, probe_lengths_csv, alpha, args)
    candidates: List[Dict[str, Any]] = []
    total_probe_sec = 0.0
    for mask_length in parse_probe_lengths(probe_lengths_csv):
        score = probe_length_score(task, tokenizer, model, probe_cfg, args, mask_length)
        candidates.append(score)
        total_probe_sec += float(score["probe_sec"])
    best = _best_candidate(candidates, args.tie_break)
    return {
        "selected_mask_length": int(best["mask_length"]),
        "selected_score": float(best["score"]),
        "selected_raw_score": float(best["raw_score"]),
        "selected_adjusted_score": float(best["adjusted_score"]),
        "candidate_scores": candidates,
        "length_probe_sec": float(total_probe_sec),
        "probe_lengths": [int(item["mask_length"]) for item in candidates],
        "tie_break": args.tie_break,
        "score_mode": args.score_mode,
        "length_alpha": float(alpha),
    }


def _select_correction_candidate(
    correction_selection: Dict[str, Any],
    base_selected_length: int,
    settings: DreamBoundedRepairSettings,
    tie_break: str,
) -> Dict[str, Any]:
    candidates = [
        item
        for item in correction_selection["candidate_scores"]
        if _candidate_length(item) >= max(int(base_selected_length), int(settings.strong_min_len))
    ]
    if not candidates:
        raise ValueError("Correction probe produced no usable candidates")
    best = _best_candidate(candidates, tie_break)
    if settings.correction_selection_rule == "argmax":
        return best
    best_score = _candidate_score(best)
    supported = [
        item
        for item in candidates
        if _candidate_score(item) >= float(settings.shortest_supported_ratio) * best_score
    ]
    return min(supported, key=_candidate_length) if supported else best


def _apply_correction_selection(
    correction_selection: Dict[str, Any],
    base_selected_length: int,
    settings: DreamBoundedRepairSettings,
    tie_break: str,
) -> Dict[str, Any]:
    selected_candidate = _select_correction_candidate(correction_selection, base_selected_length, settings, tie_break)
    adjusted = copy.deepcopy(correction_selection)
    adjusted["selected_mask_length"] = int(selected_candidate["mask_length"])
    adjusted["selected_score"] = float(selected_candidate.get("score", selected_candidate.get("adjusted_score")))
    adjusted["selected_raw_score"] = float(
        selected_candidate.get("raw_score", selected_candidate.get("mean_top1_prob"))
    )
    adjusted["selected_adjusted_score"] = float(
        selected_candidate.get("adjusted_score", selected_candidate.get("score", adjusted["selected_score"]))
    )
    adjusted["correction_selection_rule"] = settings.correction_selection_rule
    adjusted["shortest_supported_ratio"] = float(settings.shortest_supported_ratio)
    adjusted["correction_argmax_selected_mask_length"] = int(correction_selection["selected_mask_length"])
    return adjusted


def _base_best_info(base_selection: Dict[str, Any], settings: DreamBoundedRepairSettings, tie_break: str) -> Dict[str, Any]:
    candidates = list(base_selection["candidate_scores"])
    long_candidates = [item for item in candidates if int(item["mask_length"]) >= int(settings.strong_min_len)]
    best = _best_candidate(candidates, tie_break)
    best_long = _best_candidate(long_candidates, tie_break)
    best_raw = _best_candidate(candidates, tie_break, score_fn=_raw_score)
    best_long_raw = _best_candidate(long_candidates, tie_break, score_fn=_raw_score)
    best_score = _candidate_score(best)
    best_long_score = _candidate_score(best_long)
    best_raw_score = _raw_score(best_raw)
    best_long_raw_score = _raw_score(best_long_raw)
    support_count = sum(
        1
        for item in long_candidates
        if _candidate_score(item) >= float(settings.ratio_trigger_threshold) * best_score
    )
    return {
        "best_len": int(best["mask_length"]),
        "best_score": best_score,
        "best_long_len": int(best_long["mask_length"]),
        "best_long_score": best_long_score,
        "long_ratio": best_long_score / best_score if best_score else None,
        "best_raw_len": int(best_raw["mask_length"]),
        "best_raw_score": best_raw_score,
        "best_long_raw_len": int(best_long_raw["mask_length"]),
        "best_long_raw_score": best_long_raw_score,
        "raw_long_ratio": best_long_raw_score / best_raw_score if best_raw_score else None,
        "support_count": int(support_count),
    }


def _effective_weak_grid(base_len: int, settings: DreamBoundedRepairSettings) -> str:
    weak_lengths = parse_probe_lengths(settings.weak_probe_lengths_csv)
    if settings.short_safe_policy == "s1":
        return ",".join(str(length) for length in weak_lengths)
    cap = None
    if base_len <= 8:
        cap = int(settings.cap_base_le8)
    elif 9 <= base_len <= 12:
        cap = int(settings.cap_base_9_12)
    if cap is not None:
        weak_lengths = [length for length in weak_lengths if length <= cap]
    if not weak_lengths:
        raise ValueError(f"Empty weak grid after applying cap for base_len={base_len}")
    return ",".join(str(length) for length in weak_lengths)


def _trigger_decision(
    base_selected_length: int,
    best_info: Dict[str, Any],
    settings: DreamBoundedRepairSettings,
) -> Dict[str, Any]:
    long_score_floor_passed = float(best_info["best_long_score"]) >= float(settings.long_score_floor)
    long_ratio = best_info["long_ratio"]
    ratio_passed = (
        long_ratio is not None
        and float(long_ratio) >= float(settings.ratio_trigger_threshold)
        and long_score_floor_passed
    )
    raw_ratio = best_info["raw_long_ratio"]
    raw_ratio_passed = raw_ratio is not None and float(raw_ratio) >= float(settings.raw_ratio_threshold)
    support_passed = int(best_info["support_count"]) >= int(settings.support_count_threshold)
    strong_trigger = int(base_selected_length) >= int(settings.strong_min_len)
    weak_window = int(settings.weak_min_base_len) <= int(base_selected_length) <= int(settings.weak_max_base_len)

    if strong_trigger:
        return {
            "triggered": True,
            "strong_triggered": True,
            "weak_triggered": False,
            "trigger_reason": "strong_base_len",
            "correction_kind": "strong_full",
            "correction_probe_lengths_csv": settings.strong_probe_lengths_csv,
            "long_score_floor_passed": bool(long_score_floor_passed),
            "ratio_passed": bool(ratio_passed),
            "raw_ratio_passed": bool(raw_ratio_passed),
            "support_passed": bool(support_passed),
            "weak_window": bool(weak_window),
        }

    weak_triggered = False
    trigger_reason = "none"
    if weak_window and ratio_passed:
        if settings.short_safe_policy == "s1":
            weak_triggered = True
            trigger_reason = "weak_ratio"
        elif settings.short_safe_policy == "s2":
            weak_triggered = True
            trigger_reason = "weak_ratio_jump_cap"
        elif settings.short_safe_policy == "s3" and raw_ratio_passed:
            weak_triggered = True
            trigger_reason = "weak_ratio_raw_confirmed_jump_cap"
        elif settings.short_safe_policy == "s4" and support_passed:
            weak_triggered = True
            trigger_reason = "weak_ratio_support_count_jump_cap"

    return {
        "triggered": bool(weak_triggered),
        "strong_triggered": False,
        "weak_triggered": bool(weak_triggered),
        "trigger_reason": trigger_reason,
        "correction_kind": "weak_grid_capped" if weak_triggered else "none",
        "correction_probe_lengths_csv": _effective_weak_grid(base_selected_length, settings) if weak_triggered else None,
        "long_score_floor_passed": bool(long_score_floor_passed),
        "ratio_passed": bool(ratio_passed),
        "raw_ratio_passed": bool(raw_ratio_passed),
        "support_passed": bool(support_passed),
        "weak_window": bool(weak_window),
    }


def probe_official_cal_score(
    task: CodeTask,
    tokenizer,
    model,
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    settings: DreamBoundedRepairSettings,
    mask_length: int,
) -> Dict[str, Any]:
    canvas = build_canvas(task, tokenizer, mask_length, args)
    device = resolve_model_device(model)
    input_tensor = torch.tensor([canvas["input_ids"]], dtype=torch.long, device=device)
    attention_tensor = torch.tensor([canvas["attention_mask"]], dtype=torch.long, device=device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = shifted_logits_for_dream(model, input_tensor, attention_mask=attention_tensor)
    probe_sec = time.perf_counter() - start

    middle_logits = logits[:, canvas["middle_start"] : canvas["middle_end"], :]
    probs = torch.softmax(middle_logits, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1)
    top2_values, _ = torch.topk(probs, k=2, dim=-1)
    raw_phi = float(max_probs.mean().item())
    bias = length_bias(mask_length, settings.official_bias_params)
    calibrated_phi = raw_phi / bias if settings.official_use_bias else raw_phi
    return {
        "mask_length": int(mask_length),
        "score": float(calibrated_phi),
        "calibrated_confidence": float(calibrated_phi),
        "raw_score": float(raw_phi),
        "raw_phi": float(raw_phi),
        "adjusted_score": float(calibrated_phi),
        "bias": float(bias),
        "mean_top1_prob": float(raw_phi),
        "mean_top2_gap": float((top2_values[..., 0] - top2_values[..., 1]).mean().item()),
        "score_mode": "official_cal_bias" if settings.official_use_bias else "official_cal_raw",
        "probe_sec": float(probe_sec),
        "logit_alignment": "dream_shifted_logits_middle_canvas",
    }


def select_dream_official_cal(
    task: CodeTask,
    tokenizer,
    model,
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    settings: DreamBoundedRepairSettings,
) -> Dict[str, Any]:
    candidate_scores: List[Dict[str, Any]] = []
    seen_lengths: set[int] = set()
    total_probe_sec = 0.0
    search_events: List[Dict[str, Any]] = []

    def get_conf_at_len(length: int, direction: str) -> Optional[Dict[str, Any]]:
        nonlocal total_probe_sec
        if length < 1 or length > settings.official_max_length or length in seen_lengths:
            return None
        seen_lengths.add(length)
        candidate = probe_official_cal_score(task, tokenizer, model, cfg, args, settings, length)
        candidate["search_direction"] = direction
        candidate_scores.append(candidate)
        total_probe_sec += float(candidate["probe_sec"])
        return candidate

    first = get_conf_at_len(settings.official_initial_length, "initial")
    if first is None:
        raise RuntimeError("failed to probe initial official-CAL length")
    best = first
    search_events.append(
        {
            "length": int(first["mask_length"]),
            "score": float(first["score"]),
            "raw_phi": float(first["raw_phi"]),
            "bias": float(first["bias"]),
            "direction": "initial",
            "is_new_best": True,
            "consecutive_decreases": 0,
        }
    )

    curr_l = settings.official_initial_length + settings.official_span
    consecutive_decreases = 0
    while curr_l <= settings.official_max_length:
        candidate = get_conf_at_len(curr_l, "up")
        if candidate is None:
            curr_l += settings.official_span
            continue
        is_new_best = float(candidate["score"]) > float(best["score"])
        if is_new_best:
            best = candidate
            consecutive_decreases = 0
        else:
            consecutive_decreases += 1
        search_events.append(
            {
                "length": int(candidate["mask_length"]),
                "score": float(candidate["score"]),
                "raw_phi": float(candidate["raw_phi"]),
                "bias": float(candidate["bias"]),
                "direction": "up",
                "is_new_best": is_new_best,
                "consecutive_decreases": consecutive_decreases,
            }
        )
        if consecutive_decreases >= settings.official_dstep:
            break
        curr_l += settings.official_span

    curr_l = settings.official_initial_length - settings.official_span
    consecutive_decreases = 0
    while curr_l >= 1:
        candidate = get_conf_at_len(curr_l, "down")
        if candidate is None:
            curr_l -= settings.official_span
            continue
        is_new_best = float(candidate["score"]) > float(best["score"])
        if is_new_best:
            best = candidate
            consecutive_decreases = 0
        else:
            consecutive_decreases += 1
        search_events.append(
            {
                "length": int(candidate["mask_length"]),
                "score": float(candidate["score"]),
                "raw_phi": float(candidate["raw_phi"]),
                "bias": float(candidate["bias"]),
                "direction": "down",
                "is_new_best": is_new_best,
                "consecutive_decreases": consecutive_decreases,
            }
        )
        if consecutive_decreases >= settings.official_dstep:
            break
        curr_l -= settings.official_span

    return {
        "selected_mask_length": int(best["mask_length"]),
        "selected_score": float(best["score"]),
        "selected_raw_score": float(best["raw_score"]),
        "selected_adjusted_score": float(best["score"]),
        "candidate_scores": sorted(candidate_scores, key=lambda item: int(item["mask_length"])),
        "length_probe_sec": float(total_probe_sec),
        "probe_lengths": [int(item["mask_length"]) for item in candidate_scores],
        "tie_break": "first_strict_greater",
        "score_mode": "official_cal_bias" if settings.official_use_bias else "official_cal_raw",
        "length_alpha": None,
        "official_cal": {
            "version": "dream_official_cal",
            "initial_length": settings.official_initial_length,
            "span": settings.official_span,
            "max_length": settings.official_max_length,
            "dstep": settings.official_dstep,
            "use_bias": settings.official_use_bias,
            "bias_params": list(settings.official_bias_params),
            "search_steps": len(candidate_scores),
            "probed_lengths": [int(item["mask_length"]) for item in candidate_scores],
            "search_events": search_events,
            "selected_mask_length": int(best["mask_length"]),
            "selected_score": float(best["score"]),
            "selected_raw_score": float(best["raw_score"]),
            "selected_bias": float(best["bias"]),
            "selected_calibrated_confidence": float(best["calibrated_confidence"]),
            "logit_alignment": "dream_shifted_logits_middle_canvas",
        },
    }


def _source_matches(source: Optional[str], allowed: str) -> bool:
    if not allowed or allowed == "any":
        return True
    return str(source or "") in {item.strip() for item in allowed.split(",") if item.strip()}


def _repair_decision(
    s3_selected: int,
    official_selected: Optional[int],
    settings: DreamBoundedRepairSettings,
    *,
    pre_repair_source: Optional[str] = None,
    long_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    official_eval_max = (
        int(settings.official_eval_max_s3_len)
        if settings.official_eval_max_s3_len is not None
        else int(settings.repair_max_s3_len)
    )
    considered = int(s3_selected) <= official_eval_max
    if not considered:
        return {"considered": False, "triggered": False, "reason": "s3_len_above_official_eval_max", "delta": None}
    if official_selected is None:
        return {"considered": True, "triggered": False, "reason": "official_not_run", "delta": None}

    delta = int(official_selected) - int(s3_selected)
    bounded_triggered = (
        int(s3_selected) <= int(settings.repair_max_s3_len)
        and int(settings.repair_min_official_len)
        <= int(official_selected)
        <= int(settings.repair_max_official_len)
        and int(settings.repair_min_delta)
        <= int(delta)
        <= int(settings.repair_max_delta)
    )
    suspicion_triggered = False
    if settings.suspicion_min_official_len is not None:
        suspicion_max_s3_len = (
            int(settings.suspicion_max_s3_len)
            if settings.suspicion_max_s3_len is not None
            else int(settings.repair_max_s3_len)
        )
        suspicion_triggered = (
            int(s3_selected) <= suspicion_max_s3_len
            and int(settings.suspicion_min_official_len)
            <= int(official_selected)
            <= int(settings.suspicion_max_official_len)
            and int(delta) >= int(settings.suspicion_min_delta)
        )
    mid_rescue_triggered = False
    if settings.mid_rescue_min_official_len is not None:
        mid_rescue_max_s3_len = (
            int(settings.mid_rescue_max_s3_len)
            if settings.mid_rescue_max_s3_len is not None
            else official_eval_max
        )
        mid_rescue_triggered = (
            int(s3_selected) <= mid_rescue_max_s3_len
            and int(settings.mid_rescue_min_official_len)
            <= int(official_selected)
            <= int(settings.mid_rescue_max_official_len)
            and int(settings.mid_rescue_min_delta)
            <= int(delta)
            <= int(settings.mid_rescue_max_delta)
            and (
                settings.mid_rescue_min_long_ratio is None
                or (long_ratio is not None and float(long_ratio) >= float(settings.mid_rescue_min_long_ratio))
            )
            and _source_matches(pre_repair_source, settings.mid_rescue_source)
        )

    triggered = bool(bounded_triggered or suspicion_triggered or mid_rescue_triggered)
    if bounded_triggered:
        reason = "official_bounded_repair"
    elif suspicion_triggered:
        reason = "official_long_suspicion"
    elif mid_rescue_triggered:
        reason = "official_mid_rescue"
    else:
        reason = "official_no_trigger"
    return {
        "considered": True,
        "triggered": triggered,
        "reason": reason,
        "delta": int(delta),
        "bounded_triggered": bool(bounded_triggered),
        "suspicion_triggered": bool(suspicion_triggered),
        "mid_rescue_triggered": bool(mid_rescue_triggered),
    }


def build_bounded_repair_settings(args: argparse.Namespace) -> DreamBoundedRepairSettings:
    return DreamBoundedRepairSettings(
        base_probe_lengths_csv=args.base_probe_lengths,
        base_alpha=args.base_alpha,
        weak_probe_lengths_csv=args.weak_probe_lengths,
        strong_probe_lengths_csv=args.strong_probe_lengths,
        long_alpha=args.long_alpha,
        strong_min_len=args.strong_min_len,
        weak_min_base_len=args.weak_min_base_len,
        weak_max_base_len=args.weak_max_base_len,
        ratio_trigger_threshold=args.ratio_trigger_threshold,
        long_score_floor=args.long_score_floor,
        raw_ratio_threshold=args.raw_ratio_threshold,
        support_count_threshold=args.support_count_threshold,
        cap_base_le8=args.cap_base_le8,
        cap_base_9_12=args.cap_base_9_12,
        short_safe_policy=args.short_safe_policy,
        correction_selection_rule=args.correction_selection_rule,
        shortest_supported_ratio=args.shortest_supported_ratio,
        official_initial_length=args.official_initial_length,
        official_span=args.official_span,
        official_max_length=args.official_max_length,
        official_dstep=args.official_dstep,
        official_use_bias=not args.official_no_bias,
        official_bias_params=parse_bias_params(args.official_bias_params),
        official_eval_max_s3_len=args.official_eval_max_s3_len,
        repair_max_s3_len=args.repair_max_s3_len,
        repair_min_official_len=args.repair_min_official_len,
        repair_max_official_len=args.repair_max_official_len,
        repair_min_delta=args.repair_min_delta,
        repair_max_delta=args.repair_max_delta,
        suspicion_max_s3_len=args.suspicion_max_s3_len,
        suspicion_min_official_len=args.suspicion_min_official_len,
        suspicion_max_official_len=args.suspicion_max_official_len,
        suspicion_min_delta=args.suspicion_min_delta,
        mid_rescue_max_s3_len=args.mid_rescue_max_s3_len,
        mid_rescue_min_official_len=args.mid_rescue_min_official_len,
        mid_rescue_max_official_len=args.mid_rescue_max_official_len,
        mid_rescue_min_delta=args.mid_rescue_min_delta,
        mid_rescue_max_delta=args.mid_rescue_max_delta,
        mid_rescue_min_long_ratio=args.mid_rescue_min_long_ratio,
        mid_rescue_source=args.mid_rescue_source,
    )


def resolve_mask_length(task: CodeTask, tokenizer, model, cfg: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
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
            score = probe_length_score(task, tokenizer, model, cfg, args, mask_length)
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
            "length_probe_sec": float(total_probe_sec),
            "probe_lengths": [int(item["mask_length"]) for item in candidates],
            "tie_break": cfg.decode.cal_lite_tie_break,
            "score_mode": cfg.decode.cal_lite_score_mode,
            "length_alpha": float(cfg.decode.cal_lite_length_alpha),
            **length_diff(selected, oracle),
        }

    if source == "lcal_official_bounded_repair":
        settings = build_bounded_repair_settings(args)
        base_selection = select_dream_cal_lite(
            task,
            tokenizer,
            model,
            cfg,
            args,
            probe_lengths_csv=settings.base_probe_lengths_csv,
            alpha=settings.base_alpha,
        )
        base_selected = int(base_selection["selected_mask_length"])
        best_info = _base_best_info(base_selection, settings, args.tie_break)
        trigger = _trigger_decision(base_selected, best_info, settings)

        long_selection: Optional[Dict[str, Any]] = None
        final_selection = base_selection
        final_source = "base"
        total_probe_sec = float(base_selection["length_probe_sec"])
        s3_selected = base_selected

        if bool(trigger["triggered"]):
            raw_long_selection = select_dream_cal_lite(
                task,
                tokenizer,
                model,
                cfg,
                args,
                probe_lengths_csv=str(trigger["correction_probe_lengths_csv"]),
                alpha=settings.long_alpha,
            )
            long_selection = _apply_correction_selection(raw_long_selection, base_selected, settings, args.tie_break)
            long_selected = int(long_selection["selected_mask_length"])
            if long_selected >= base_selected:
                final_selection = long_selection
                final_source = "strong_correction" if trigger["strong_triggered"] else "weak_correction"
            total_probe_sec += float(long_selection["length_probe_sec"])
            s3_selected = max(base_selected, long_selected)

        pre_repair_source = final_source
        official_selection: Optional[Dict[str, Any]] = None
        decision = _repair_decision(
            s3_selected,
            None,
            settings,
            pre_repair_source=pre_repair_source,
            long_ratio=best_info.get("long_ratio"),
        )
        if bool(decision["considered"]):
            official_selection = select_dream_official_cal(task, tokenizer, model, cfg, args, settings)
            official_selected = int(official_selection["selected_mask_length"])
            decision = _repair_decision(
                s3_selected,
                official_selected,
                settings,
                pre_repair_source=pre_repair_source,
                long_ratio=best_info.get("long_ratio"),
            )
            total_probe_sec += float(official_selection["length_probe_sec"])
            if bool(decision["triggered"]):
                final_selection = official_selection
                final_source = str(decision["reason"])

        selected = int(final_selection["selected_mask_length"])
        diff = length_diff(selected, oracle)
        base_diff = length_diff(base_selected, oracle)
        lcal_meta = {
            "version": "dream_lcal_official_bounded_repair",
            "short_safe_policy": settings.short_safe_policy,
            "pre_repair_source": pre_repair_source,
            "final_source": final_source,
            "long_triggered": bool(trigger["triggered"]),
            "strong_triggered": bool(trigger["strong_triggered"]),
            "weak_triggered": bool(trigger["weak_triggered"]),
            "correction_kind": trigger["correction_kind"],
            "trigger_reason": str(trigger["trigger_reason"]),
            "best_score": float(best_info["best_score"]),
            "best_len": int(best_info["best_len"]),
            "best_long_score": float(best_info["best_long_score"]),
            "best_long_len": int(best_info["best_long_len"]),
            "long_ratio": None if best_info["long_ratio"] is None else float(best_info["long_ratio"]),
            "best_raw_score": float(best_info["best_raw_score"]),
            "best_raw_len": int(best_info["best_raw_len"]),
            "best_long_raw_score": float(best_info["best_long_raw_score"]),
            "best_long_raw_len": int(best_info["best_long_raw_len"]),
            "raw_long_ratio": None if best_info["raw_long_ratio"] is None else float(best_info["raw_long_ratio"]),
            "long_score_floor_passed": bool(trigger["long_score_floor_passed"]),
            "ratio_passed": bool(trigger["ratio_passed"]),
            "raw_ratio_passed": bool(trigger["raw_ratio_passed"]),
            "support_count": int(best_info["support_count"]),
            "support_passed": bool(trigger["support_passed"]),
            "weak_window": bool(trigger["weak_window"]),
            "base_probe_lengths": parse_probe_lengths(settings.base_probe_lengths_csv),
            "base_alpha": float(settings.base_alpha),
            "base_selected_length": base_selected,
            "base_selected_mask_length": base_selected,
            "base_selected_score": float(base_selection["selected_score"]),
            "base_selected_raw_score": float(base_selection["selected_raw_score"]),
            "base_selected_adjusted_score": float(base_selection["selected_adjusted_score"]),
            "base_length_probe_sec": float(base_selection["length_probe_sec"]),
            "base_selected_minus_oracle_length": base_diff["selected_minus_oracle_length"],
            "base_abs_selected_minus_oracle_length": base_diff["abs_selected_minus_oracle_length"],
            "base_candidate_scores": base_selection["candidate_scores"],
            "weak_probe_lengths": parse_probe_lengths(settings.weak_probe_lengths_csv),
            "strong_probe_lengths": parse_probe_lengths(settings.strong_probe_lengths_csv),
            "long_probe_lengths": (
                None
                if trigger["correction_probe_lengths_csv"] is None
                else parse_probe_lengths(str(trigger["correction_probe_lengths_csv"]))
            ),
            "long_alpha": float(settings.long_alpha),
            "long_selected_length": None if long_selection is None else int(long_selection["selected_mask_length"]),
            "long_selected_mask_length": None if long_selection is None else int(long_selection["selected_mask_length"]),
            "long_selected_score": None if long_selection is None else float(long_selection["selected_score"]),
            "long_selected_raw_score": None if long_selection is None else float(long_selection["selected_raw_score"]),
            "long_selected_adjusted_score": None if long_selection is None else float(long_selection["selected_adjusted_score"]),
            "long_length_probe_sec": 0.0 if long_selection is None else float(long_selection["length_probe_sec"]),
            "long_candidate_scores": None if long_selection is None else long_selection["candidate_scores"],
            "long_argmax_selected_length": None
            if long_selection is None
            else int(long_selection["correction_argmax_selected_mask_length"]),
            "pre_repair_selected_length": s3_selected,
            "s3_selected_length": s3_selected,
            "official_repair_considered": bool(decision["considered"]),
            "official_repair_triggered": bool(decision["triggered"]),
            "official_repair_reason": str(decision["reason"]),
            "official_repair_delta": decision["delta"],
            "official_repair_bounded_triggered": bool(decision.get("bounded_triggered", False)),
            "official_repair_suspicion_triggered": bool(decision.get("suspicion_triggered", False)),
            "official_repair_mid_rescue_triggered": bool(decision.get("mid_rescue_triggered", False)),
            "official_selected_length": None if official_selection is None else int(official_selection["selected_mask_length"]),
            "official_selected_score": None if official_selection is None else float(official_selection["selected_score"]),
            "official_selected_raw_score": None if official_selection is None else float(official_selection["selected_raw_score"]),
            "official_selected_adjusted_score": None
            if official_selection is None
            else float(official_selection["selected_adjusted_score"]),
            "official_length_probe_sec": 0.0 if official_selection is None else float(official_selection["length_probe_sec"]),
            "official_search_steps": None if official_selection is None else len(official_selection["candidate_scores"]),
            "official_candidate_scores": None if official_selection is None else official_selection["candidate_scores"],
            "official_eval_max_s3_len": settings.official_eval_max_s3_len,
            "repair_max_s3_len": int(settings.repair_max_s3_len),
            "repair_min_official_len": int(settings.repair_min_official_len),
            "repair_max_official_len": int(settings.repair_max_official_len),
            "repair_min_delta": int(settings.repair_min_delta),
            "repair_max_delta": int(settings.repair_max_delta),
            "suspicion_max_s3_len": settings.suspicion_max_s3_len,
            "suspicion_min_official_len": settings.suspicion_min_official_len,
            "suspicion_max_official_len": int(settings.suspicion_max_official_len),
            "suspicion_min_delta": int(settings.suspicion_min_delta),
            "mid_rescue_max_s3_len": settings.mid_rescue_max_s3_len,
            "mid_rescue_min_official_len": settings.mid_rescue_min_official_len,
            "mid_rescue_max_official_len": int(settings.mid_rescue_max_official_len),
            "mid_rescue_min_delta": int(settings.mid_rescue_min_delta),
            "mid_rescue_max_delta": int(settings.mid_rescue_max_delta),
            "mid_rescue_min_long_ratio": settings.mid_rescue_min_long_ratio,
            "mid_rescue_source": settings.mid_rescue_source,
            "final_selected_length": selected,
            "final_minus_base_length": selected - base_selected,
            "final_minus_s3_length": selected - s3_selected,
            "settings": {**asdict(settings), "official_bias_params": list(settings.official_bias_params)},
        }
        return {
            **base,
            "mask_length_source": "lcal_official_bounded_repair",
            "mask_length": selected,
            "selected_mask_length": selected,
            "selected_score": float(final_selection["selected_score"]),
            "selected_raw_score": float(final_selection["selected_raw_score"]),
            "selected_adjusted_score": float(final_selection["selected_adjusted_score"]),
            "candidate_scores": final_selection["candidate_scores"],
            "length_probe_sec": float(total_probe_sec),
            "probe_lengths": final_selection["probe_lengths"],
            "tie_break": final_selection["tie_break"],
            "score_mode": final_selection["score_mode"],
            "length_alpha": final_selection["length_alpha"],
            "lcal_v3": lcal_meta,
            "official_cal": {} if official_selection is None else official_selection["official_cal"],
            **diff,
        }

    raise ValueError(f"Unsupported mask length source: {source}")


def clean_middle_text(text: str, tokenizer) -> str:
    cleaned = text
    stops = [
        tokenizer.eos_token,
        tokenizer.pad_token,
        "<|endoftext|>",
        "<|beginoftext|>",
        "<|mask|>",
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|fim_pad|>",
        "<|im_end|>",
    ]
    for stop in stops:
        if stop and stop in cleaned:
            cleaned = cleaned.split(stop, 1)[0]
    return cleaned


def build_right_pad_hook(input_length: int, eos_token_id: int, enabled: bool):
    def hook(step, x, logits):
        if enabled and x.shape[1] > input_length:
            x[:, input_length:] = eos_token_id
        return x

    return hook


def run_task(task: CodeTask, tokenizer, model, cfg: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    length_meta = resolve_mask_length(task, tokenizer, model, cfg, args)
    length_probe_sec = float(length_meta["length_probe_sec"])
    selected_len = int(length_meta["mask_length"])
    canvas = build_canvas(task, tokenizer, selected_len, args)

    device = resolve_model_device(model)
    input_ids = torch.tensor([canvas["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([canvas["attention_mask"]], dtype=torch.long, device=device)

    right_pad_new_tokens = max(1, int(args.right_pad_new_tokens))
    force_right_pad_eos = not args.no_force_right_pad_eos
    hook = build_right_pad_hook(canvas["input_length"], canvas["eos_token_id"], force_right_pad_eos)

    decode_start = time.perf_counter()
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=right_pad_new_tokens,
            steps=int(args.dream_steps),
            temperature=float(args.temperature),
            top_p=None if args.top_p is None else float(args.top_p),
            top_k=args.top_k,
            alg=args.alg,
            alg_temp=args.alg_temp,
            eos_penalty=float(args.eos_penalty),
            mask_token_id=canvas["mask_token_id"],
            pad_token_id=canvas["pad_token_id"],
            eos_token_id=canvas["eos_token_id"],
            bos_token_id=canvas["bos_token_id"],
            return_dict_in_generate=True,
            output_history=False,
            generation_tokens_hook_func=hook,
        )
    decode_sec = time.perf_counter() - decode_start

    sequence = output.sequences[0]
    middle_ids = sequence[canvas["middle_start"] : canvas["middle_end"]].tolist()
    raw_middle = tokenizer.decode(middle_ids, skip_special_tokens=False)
    skipped_middle = tokenizer.decode(middle_ids, skip_special_tokens=True)
    middle_text = clean_middle_text(skipped_middle, tokenizer)
    full_text = task.prefix + middle_text + task.suffix

    verification = run_verifier_stack(task, full_text, middle_text)
    verification_sec = sum(item.duration_sec for item in verification.values())
    total_sec = decode_sec + verification_sec
    total_sec_including_probe = total_sec + length_probe_sec
    tier3 = verification.get("tier3_unit_tests")
    passed = bool(tier3.passed) if tier3 else False

    reconstruction = {
        "prefix_text": task.prefix,
        "middle_text": middle_text,
        "suffix_text": task.suffix,
        "full_text": full_text,
    }
    diagnostics = build_reconstruction_diagnostics(task, reconstruction)
    diagnostics.update(
        {
            "raw_middle_text": raw_middle,
            "skip_special_middle_text": skipped_middle,
            "reference_middle_text": infer_reference_middle_text(task),
            "selected_middle_token_ids": middle_ids,
            "middle_start": int(canvas["middle_start"]),
            "middle_end": int(canvas["middle_end"]),
            "input_token_count": int(canvas["input_length"]),
            "sequence_token_count": int(sequence.shape[0]),
            "right_pad_new_tokens": int(right_pad_new_tokens),
            "force_right_pad_eos": bool(force_right_pad_eos),
            "decode_backend": "dreamcoder_native_diffusion_generate_fixed_canvas",
            "canvas_format": "bos_prefix_masks_suffix_eos",
            "use_bos": bool(canvas["use_bos"]),
            "use_eos": bool(canvas["use_eos"]),
        }
    )

    return {
        "task_id": task.task_id,
        "dataset_subset": cfg.data.dataset_subset,
        "metrics": {
            "passed": passed,
            "decode_sec": float(decode_sec),
            "verification_sec": float(verification_sec),
            "total_sec": float(total_sec),
            "total_sec_including_probe": float(total_sec_including_probe),
            "length_probe_sec": float(length_probe_sec),
            "total_steps": int(args.dream_steps),
            "effective_steps": int(args.dream_steps),
            "stopped": False,
            "stop_step": None,
            "stop_reason": "dreamcoder_native_diffusion_generate_fixed_canvas",
            "mask_length": selected_len,
            "oracle_mask_length": length_meta["oracle_mask_length"],
            "selected_mask_length": selected_len,
            "selected_score": length_meta["selected_score"],
            "selected_raw_score": length_meta["selected_raw_score"],
            "selected_adjusted_score": length_meta["selected_adjusted_score"],
            "selected_minus_oracle_length": length_meta["selected_minus_oracle_length"],
            "abs_selected_minus_oracle_length": length_meta["abs_selected_minus_oracle_length"],
            "mask_length_source": length_meta["mask_length_source"],
            "final_source": (length_meta.get("lcal_v3") or {}).get("final_source"),
            "pre_repair_source": (length_meta.get("lcal_v3") or {}).get("pre_repair_source"),
            "s3_selected_length": (length_meta.get("lcal_v3") or {}).get("s3_selected_length"),
            "base_selected_length": (length_meta.get("lcal_v3") or {}).get("base_selected_length"),
            "long_selected_length": (length_meta.get("lcal_v3") or {}).get("long_selected_length"),
            "official_selected_length": (length_meta.get("lcal_v3") or {}).get("official_selected_length"),
            "official_repair_considered": (length_meta.get("lcal_v3") or {}).get("official_repair_considered"),
            "official_repair_triggered": (length_meta.get("lcal_v3") or {}).get("official_repair_triggered"),
            "official_repair_reason": (length_meta.get("lcal_v3") or {}).get("official_repair_reason"),
            "official_repair_delta": (length_meta.get("lcal_v3") or {}).get("official_repair_delta"),
            "trigger_reason": (length_meta.get("lcal_v3") or {}).get("trigger_reason"),
            "long_ratio": (length_meta.get("lcal_v3") or {}).get("long_ratio"),
            "raw_long_ratio": (length_meta.get("lcal_v3") or {}).get("raw_long_ratio"),
            "decode_backend": "dreamcoder_native_diffusion_generate_fixed_canvas",
            "canvas_format": "bos_prefix_masks_suffix_eos",
            "dream_steps": int(args.dream_steps),
            "dream_alg": args.alg,
            "dream_temperature": float(args.temperature),
            "dream_top_p": args.top_p,
            "dream_top_k": args.top_k,
            "dream_alg_temp": args.alg_temp,
            "dream_eos_penalty": float(args.eos_penalty),
            "right_pad_new_tokens": int(right_pad_new_tokens),
            "force_right_pad_eos": bool(force_right_pad_eos),
        },
        "verification": {key: value.to_dict() for key, value in verification.items()},
        "length_probe": {
            "candidate_scores": length_meta["candidate_scores"],
            "length_probe_sec": float(length_probe_sec),
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
            "logit_alignment": "dream_shifted_logits_middle_canvas",
            "lcal_v3": length_meta.get("lcal_v3"),
            "official_cal": length_meta.get("official_cal"),
        },
        "stopping": {
            "enabled": False,
            "method": "dreamcoder_native_diffusion_generate",
            "policy": None,
            "stopped": False,
            "stop_step": None,
            "stop_reason": "dreamcoder_native_diffusion_generate_fixed_canvas",
            "effective_steps": int(args.dream_steps),
        },
        "code": full_text,
        "diagnostics": diagnostics,
        "lcal_v3": length_meta.get("lcal_v3"),
        "official_cal": length_meta.get("official_cal"),
    }


def _oracle_bucket(length: Optional[int]) -> str:
    if length is None:
        return "unknown"
    value = int(length)
    if value <= 8:
        return "<=8"
    if value <= 12:
        return "9-12"
    if value <= 16:
        return "13-16"
    if value <= 24:
        return "17-24"
    return "25+"


def load_jsonl(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_pairwise(
    results: List[Dict[str, Any]],
    baseline_rows: List[Dict[str, Any]],
    baseline_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not baseline_rows:
        return None
    baseline_by_task = {row.get("task_id"): row for row in baseline_rows}
    common = []
    for row in results:
        baseline = baseline_by_task.get(row.get("task_id"))
        if baseline is not None:
            common.append((row, baseline))
    if not common:
        return None
    wins = losses = tie_pass = tie_fail = 0
    for row, baseline in common:
        passed = bool(row.get("metrics", {}).get("passed", False))
        base_passed = bool(baseline.get("metrics", {}).get("passed", False))
        if passed and not base_passed:
            wins += 1
        elif base_passed and not passed:
            losses += 1
        elif passed and base_passed:
            tie_pass += 1
        else:
            tie_fail += 1
    return {
        "baseline_results": baseline_path,
        "common_samples": len(common),
        "overall_win": wins,
        "overall_loss": losses,
        "overall_tie_pass": tie_pass,
        "overall_tie_fail": tie_fail,
    }


def summarize(
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
    baseline_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not results:
        return {"num_samples": 0}
    metrics = [row["metrics"] for row in results]
    num_samples = len(results)
    pass_count = sum(1 for row in results if row["metrics"]["passed"])
    diffs = [m.get("selected_minus_oracle_length") for m in metrics if m.get("selected_minus_oracle_length") is not None]
    oracle_buckets = sorted({_oracle_bucket(m.get("oracle_mask_length")) for m in metrics})
    repair_considered = [m for m in metrics if bool(m.get("official_repair_considered", False))]
    repair_triggered = [m for m in metrics if bool(m.get("official_repair_triggered", False))]
    summary = {
        "num_samples": num_samples,
        "pass_count": pass_count,
        "pass_rate": pass_count / num_samples,
        "avg_decode_sec": _avg(m.get("decode_sec") for m in metrics),
        "avg_verification_sec": _avg(m.get("verification_sec") for m in metrics),
        "avg_total_sec": _avg(m.get("total_sec") for m in metrics),
        "avg_length_probe_sec": _avg(m.get("length_probe_sec") for m in metrics),
        "avg_total_sec_including_probe": _avg(m.get("total_sec_including_probe") for m in metrics),
        "avg_selected_mask_length": _avg(m.get("selected_mask_length") for m in metrics),
        "avg_oracle_mask_length": _avg(m.get("oracle_mask_length") for m in metrics),
        "avg_selected_minus_oracle_length": _avg(diffs),
        "avg_abs_selected_minus_oracle_length": _avg(m.get("abs_selected_minus_oracle_length") for m in metrics),
        "under_select_rate": _rate(diff < 0 for diff in diffs),
        "over_select_rate": _rate(diff > 0 for diff in diffs),
        "exact_length_match_rate": _rate(diff == 0 for diff in diffs),
        "selected_length_histogram": _hist(m.get("selected_mask_length") for m in metrics),
        "oracle_length_histogram": _hist(m.get("oracle_mask_length") for m in metrics),
        "oracle_bucket_histogram": _hist(_oracle_bucket(m.get("oracle_mask_length")) for m in metrics),
        "oracle_bucket_pass_rate": {
            bucket: _rate(
                bool(m.get("passed", False))
                for m in metrics
                if _oracle_bucket(m.get("oracle_mask_length")) == bucket
            )
            for bucket in oracle_buckets
        },
        "oracle_bucket_avg_selected_minus_oracle_length": {
            bucket: _avg(
                m.get("selected_minus_oracle_length")
                for m in metrics
                if _oracle_bucket(m.get("oracle_mask_length")) == bucket
            )
            for bucket in oracle_buckets
        },
        "decode_backends": sorted({str(m.get("decode_backend")) for m in metrics}),
        "canvas_formats": sorted({str(m.get("canvas_format")) for m in metrics}),
        "dream_algs": sorted({str(m.get("dream_alg")) for m in metrics}),
        "dream_temperatures": sorted({float(m.get("dream_temperature")) for m in metrics}),
        "dream_eos_penalties": sorted({float(m.get("dream_eos_penalty")) for m in metrics}),
        "mask_length_source": args.mask_length_source,
        "official_repair_considered_count": len(repair_considered),
        "official_repair_trigger_count": len(repair_triggered),
        "official_repair_trigger_rate": len(repair_triggered) / len(metrics) if metrics else None,
        "official_repair_reason_histogram": _hist(m.get("official_repair_reason") for m in metrics),
        "official_repair_source_histogram": _hist(m.get("final_source") for m in metrics),
        "official_repair_oracle_bucket_histogram": _hist(
            _oracle_bucket(m.get("oracle_mask_length")) for m in repair_triggered
        ),
        "official_repair_pass_rate": _rate(bool(m.get("passed", False)) for m in repair_triggered),
        "official_repair_true_long_precision": _rate(
            int(m.get("oracle_mask_length")) >= 17
            for m in repair_triggered
            if m.get("oracle_mask_length") is not None
        ),
        "baseline_comparison": summarize_pairwise(results, baseline_rows or [], args.baseline_results),
        "seed": int(args.seed),
        "seed_policy_note": (
            "temperature=0 with alg_temp=0 uses deterministic argmax/top-k transfer except for backend nondeterminism; "
            "one reproducibility smoke is enough. If temperature>0 or alg_temp>0 is reported, run at least 3 seeds."
        ),
    }
    summary["required_metrics"] = {
        "overall_pass_rate": summary.get("pass_rate"),
        "<=8_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("<=8"),
        "9-12_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("9-12"),
        "13-16_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("13-16"),
        "17-24_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("17-24"),
        "25+_bucket_pass_rate": summary["oracle_bucket_pass_rate"].get("25+"),
        "avg_selected_minus_oracle_length": summary.get("avg_selected_minus_oracle_length"),
        "official_repair_trigger_rate": summary.get("official_repair_trigger_rate"),
        "official_repair_true_long_precision": summary.get("official_repair_true_long_precision"),
    }
    return summary


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
    ensure_modeling_rope_utils_available()

    from transformers import AutoModel, AutoTokenizer

    logger = JsonlLogger(cfg.logging.output_dir, cfg.logging.experiment_name)
    config_payload = cfg.to_dict()
    config_payload["baseline_results"] = args.baseline_results
    config_payload["dreamcoder_official_infilling"] = {
        "decode_backend": "dreamcoder_native_diffusion_generate_fixed_canvas",
        "canvas_format": "bos_prefix_masks_suffix_eos",
        "dream_steps": args.dream_steps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "alg": args.alg,
        "alg_temp": args.alg_temp,
        "eos_penalty": args.eos_penalty,
        "right_pad_new_tokens": args.right_pad_new_tokens,
        "force_right_pad_eos": not args.no_force_right_pad_eos,
        "seed_policy_note": (
            "temperature=0 and alg_temp=0 are effectively deterministic; stochastic settings need >=3 seeds."
        ),
    }
    if args.mask_length_source == "lcal_official_bounded_repair":
        settings = build_bounded_repair_settings(args)
        config_payload["dream_lcal_official_bounded_repair"] = {
            **asdict(settings),
            "official_bias_params": list(settings.official_bias_params),
        }
    logger.save_config(config_payload)

    print("=" * 80, flush=True)
    print("Starting Dream-Coder official fixed-canvas infilling experiment", flush=True)
    print(f"experiment_name = {cfg.logging.experiment_name}", flush=True)
    print(f"output_dir       = {cfg.logging.output_dir}", flush=True)
    print(f"model_path       = {cfg.model.model_path}", flush=True)
    print(f"dataset_subset   = {cfg.data.dataset_subset}", flush=True)
    print(f"max_samples      = {cfg.data.max_samples}", flush=True)
    print(f"mask_length_src  = {cfg.decode.mask_length_source}", flush=True)
    print(f"probe_lengths    = {cfg.decode.cal_lite_probe_lengths_csv}", flush=True)
    print(f"length_alpha     = {cfg.decode.cal_lite_length_alpha}", flush=True)
    print(f"dream_steps      = {args.dream_steps}", flush=True)
    print(f"temperature      = {args.temperature}", flush=True)
    print(f"top_p            = {args.top_p}", flush=True)
    print(f"alg              = {args.alg}", flush=True)
    print(f"eos_penalty      = {args.eos_penalty}", flush=True)
    print(f"seed             = {args.seed}", flush=True)
    print(f"baseline_results = {args.baseline_results}", flush=True)
    if args.mask_length_source == "lcal_official_bounded_repair":
        print("experiment_design = DreamCoder official-canvas LCAL/S3 + official-CAL bounded repair", flush=True)
        print(f"base_grid        = {args.base_probe_lengths}", flush=True)
        print(f"base_alpha       = {args.base_alpha}", flush=True)
        print(f"strong_grid      = {args.strong_probe_lengths}", flush=True)
        print(f"weak_grid        = {args.weak_probe_lengths}", flush=True)
        print(f"long_alpha       = {args.long_alpha}", flush=True)
        print(f"repair_policy    = off_eval_s3<={args.official_eval_max_s3_len}, repair_off={args.repair_min_official_len}..{args.repair_max_official_len}, mid_rescue_off={args.mid_rescue_min_official_len}..{args.mid_rescue_max_official_len}", flush=True)
        print("metric           = HumanEval-SingleLineInfilling pass@1 via local verifier stack", flush=True)
        print("kill_criteria    = stop if smoke has malformed rows, verifier missing, near-zero pass collapse, OOM, or no summary.json", flush=True)
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

    baseline_rows: List[Dict[str, Any]] = []
    if args.baseline_results:
        if os.path.exists(args.baseline_results):
            baseline_rows = load_jsonl(args.baseline_results)
        else:
            print(f"WARNING: baseline results not found: {args.baseline_results}", flush=True)
    summary = summarize(results, args, baseline_rows)
    logger.save_json("summary.json", summary)

    print("-" * 80, flush=True)
    print("Dream-Coder official fixed-canvas infilling experiment finished", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)
    print(f"Run directory: {logger.run_dir}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
