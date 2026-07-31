from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .modeling import resolve_mask_token_id


PAPER_ID = "arXiv:2602.07546"
PAPER_VERSION = "v1"
IMPLEMENTATION_LABEL = (
    "local LR-DLLM Stage-I-only adaptation from arXiv:2602.07546v1 Algorithm 1"
)
STAGE = "stage1_only"
FIT_SCOPE = "all_exponential_probes_algorithm1"
TIE_BREAK = "shorter_on_exact_cl_tie_local_engineering_choice"


def build_exponential_probe_lengths(max_length: int) -> list[int]:
    """Return exactly the powers of two allowed by Algorithm 1."""
    limit = int(max_length)
    if limit < 1:
        raise ValueError("max_length must be positive")
    lengths: list[int] = []
    value = 1
    while value <= limit:
        lengths.append(value)
        value *= 2
    return lengths


@torch.no_grad()
def compute_span_average_negative_entropy(
    logits: torch.Tensor,
    middle_start: int,
    middle_end: int,
) -> float:
    """Compute mean sum_v p(v) log p(v) over only the masked middle span."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    start, end = int(middle_start), int(middle_end)
    if start < 0 or end <= start or end > int(logits.shape[1]):
        raise ValueError("middle span is empty or outside the logits sequence")
    mask_logits = logits[:, start:end, :].float()
    log_probs = torch.log_softmax(mask_logits, dim=-1)
    probs = log_probs.exp()
    score = (probs * log_probs).sum(dim=-1).mean()
    value = float(score.item())
    if not math.isfinite(value):
        raise RuntimeError("negative entropy is not finite")
    return value


def fit_log_length_bias(
    lengths: Sequence[int],
    average_negative_entropies: Sequence[float],
) -> dict[str, Any]:
    """Fit A(L) = intercept + k_hat * log(L) with unregularized OLS."""
    if len(lengths) != len(average_negative_entropies):
        raise ValueError("length and score counts differ")
    if len(lengths) < 2:
        raise ValueError("at least two probes are required for OLS")
    normalized_lengths = [int(length) for length in lengths]
    if any(length <= 0 for length in normalized_lengths):
        raise ValueError("all probe lengths must be positive")
    if len(set(normalized_lengths)) != len(normalized_lengths):
        raise ValueError("probe lengths must be distinct")

    x = [math.log(length) for length in normalized_lengths]
    y = [float(value) for value in average_negative_entropies]
    if not all(math.isfinite(value) for value in [*x, *y]):
        raise RuntimeError("OLS inputs are not finite")

    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("OLS log-length variance is not positive and finite")
    numerator = sum((x_value - x_mean) * (y_value - y_mean) for x_value, y_value in zip(x, y))
    k_hat = numerator / denominator
    intercept = y_mean - k_hat * x_mean
    predictions = [intercept + k_hat * value for value in x]
    residual_sum_squares = sum((observed - predicted) ** 2 for observed, predicted in zip(y, predictions))
    total_sum_squares = sum((observed - y_mean) ** 2 for observed in y)
    fit_r2 = 1.0 if total_sum_squares == 0.0 else 1.0 - residual_sum_squares / total_sum_squares

    if not all(math.isfinite(value) for value in (k_hat, intercept, fit_r2)):
        raise RuntimeError("OLS parameters are not finite")
    return {
        "k_hat": float(k_hat),
        "intercept": float(intercept),
        "fit_r2": float(fit_r2),
        "log_lengths": [float(value) for value in x],
        "fit_scope": FIT_SCOPE,
    }


def select_best_corrected_length(lengths: Sequence[int], cl_scores: Sequence[float]) -> int:
    if len(lengths) != len(cl_scores) or not lengths:
        raise ValueError("length and corrected-score counts must be equal and non-empty")
    candidates = [(int(length), float(score)) for length, score in zip(lengths, cl_scores)]
    if any(length <= 0 or not math.isfinite(score) for length, score in candidates):
        raise RuntimeError("corrected-score candidates are invalid")
    return max(candidates, key=lambda item: (item[1], -item[0]))[0]


def _task_text(task: Any, primary: str, fallback: str) -> str:
    if isinstance(task, Mapping):
        if primary in task:
            return str(task[primary])
        if fallback in task:
            return str(task[fallback])
        raise KeyError(f"task must contain {primary!r} or {fallback!r}")
    if hasattr(task, primary):
        return str(getattr(task, primary))
    if hasattr(task, fallback):
        return str(getattr(task, fallback))
    raise AttributeError(f"task must expose {primary!r} or {fallback!r}")


def _model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def tokenize_prefix_suffix(task: Any, tokenizer: Any, device: torch.device | str) -> dict[str, torch.Tensor]:
    """Mirror the pinned official CAL adapter tokenization exactly."""
    prefix = _task_text(task, "prompt", "prefix")
    suffix = _task_text(task, "suffix", "suffix")
    encoded_prefix = tokenizer(prefix, add_special_tokens=False, padding=True, return_tensors="pt")
    encoded_suffix = tokenizer(suffix, add_special_tokens=False, padding=True, return_tensors="pt")
    prefix_ids = encoded_prefix["input_ids"].to(device)
    suffix_ids = encoded_suffix["input_ids"].to(device)
    prefix_attention_mask = encoded_prefix["attention_mask"].to(device)
    suffix_attention_mask = encoded_suffix["attention_mask"].to(device)
    if prefix_ids.ndim != 2 or suffix_ids.ndim != 2 or prefix_ids.shape[0] != suffix_ids.shape[0]:
        raise ValueError("prefix and suffix tokenization must produce matching 2-D batches")
    return {
        "prefix_ids": prefix_ids,
        "suffix_ids": suffix_ids,
        "attention_mask": prefix_attention_mask,
        "suffix_attention_mask": suffix_attention_mask,
    }


def _config_max_length(config: Any) -> int:
    if isinstance(config, Mapping):
        if "max_probe_length" in config:
            return int(config["max_probe_length"])
        if "max_length" in config:
            return int(config["max_length"])
    for name in ("max_probe_length", "max_length"):
        if hasattr(config, name):
            return int(getattr(config, name))
    raise ValueError("config must define max_probe_length")


def _output_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, Mapping) and "logits" in outputs:
        return outputs["logits"]
    return outputs[0]


@torch.no_grad()
def select_length_lrdllm_stage1(
    task: Any,
    tokenizer: Any,
    model: Any,
    config: Any,
    *,
    prepared_inputs: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Run paper Algorithm 1 once, without evaluator or generation outcomes."""
    max_probe_length = _config_max_length(config)
    probe_lengths = build_exponential_probe_lengths(max_probe_length)
    if len(probe_lengths) < 2:
        raise ValueError("Stage I OLS requires max_probe_length >= 2")
    model.eval()
    device = _model_device(model)
    prepared = dict(prepared_inputs or tokenize_prefix_suffix(task, tokenizer, device))
    prefix_ids = prepared["prefix_ids"]
    suffix_ids = prepared["suffix_ids"]
    prefix_attention_mask = prepared["attention_mask"]
    suffix_attention_mask = prepared["suffix_attention_mask"]
    mask_token_id = resolve_mask_token_id(tokenizer)
    batch_size = int(prefix_ids.shape[0])
    prefix_length = int(prefix_ids.shape[1])
    suffix_length = int(suffix_ids.shape[1])

    average_negative_entropies: list[float] = []
    probe_input_token_counts: list[int] = []
    probe_token_forwards = 0
    for length in probe_lengths:
        input_ids = torch.full(
            (batch_size, prefix_length + length + suffix_length),
            mask_token_id,
            dtype=torch.long,
            device=device,
        )
        input_ids[:, :prefix_length] = prefix_ids
        input_ids[:, prefix_length + length :] = suffix_ids
        middle_attention = torch.ones(
            (batch_size, length), dtype=prefix_attention_mask.dtype, device=device
        )
        attention_mask = torch.cat(
            [prefix_attention_mask, middle_attention, suffix_attention_mask], dim=-1
        )
        outputs = model(input_ids, attention_mask=attention_mask)
        score = compute_span_average_negative_entropy(
            _output_logits(outputs), prefix_length, prefix_length + length
        )
        average_negative_entropies.append(score)
        probe_input_token_counts.append(int(input_ids.shape[-1]))
        probe_token_forwards += int(input_ids.numel())

    fit = fit_log_length_bias(probe_lengths, average_negative_entropies)
    log_lengths = list(fit["log_lengths"])
    cl_scores = [
        score - float(fit["k_hat"]) * log_length
        for score, log_length in zip(average_negative_entropies, log_lengths)
    ]
    if not all(math.isfinite(value) for value in cl_scores):
        raise RuntimeError("corrected length scores are not finite")
    selected_length = select_best_corrected_length(probe_lengths, cl_scores)
    raw_entropy_argmax_length = select_best_corrected_length(
        probe_lengths, average_negative_entropies
    )
    finite_values = [
        *average_negative_entropies,
        *log_lengths,
        float(fit["k_hat"]),
        float(fit["intercept"]),
        float(fit["fit_r2"]),
        *cl_scores,
    ]
    finite_check_passed = all(math.isfinite(value) for value in finite_values)
    if not finite_check_passed:
        raise RuntimeError("Stage I diagnostics failed the finite check")

    return {
        "paper_id": PAPER_ID,
        "paper_version": PAPER_VERSION,
        "implementation_label": IMPLEMENTATION_LABEL,
        "stage": STAGE,
        "max_probe_length": int(max_probe_length),
        "probe_lengths": probe_lengths,
        "probe_forward_count": len(probe_lengths),
        "probe_input_token_counts": probe_input_token_counts,
        "probe_token_forwards": int(probe_token_forwards),
        "average_negative_entropy_by_length": {
            str(length): float(score) for length, score in zip(probe_lengths, average_negative_entropies)
        },
        "log_length_by_length": {
            str(length): float(value) for length, value in zip(probe_lengths, log_lengths)
        },
        "k_hat": float(fit["k_hat"]),
        "intercept": float(fit["intercept"]),
        "fit_r2": float(fit["fit_r2"]),
        "cl_score_by_length": {
            str(length): float(score) for length, score in zip(probe_lengths, cl_scores)
        },
        "raw_entropy_argmax_length": int(raw_entropy_argmax_length),
        "selected_length": int(selected_length),
        "tie_break": TIE_BREAK,
        "fit_scope": FIT_SCOPE,
        "finite_check_passed": True,
    }
