from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CodeTask:
    task_id: str
    prefix: str
    suffix: str
    full_prompt: str
    test_code: str
    entry_point: str
    canonical_solution: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


def _extract_prefix_suffix(raw_task: Dict[str, Any]) -> tuple[str, str, str]:
    if "prefix" in raw_task and "suffix" in raw_task:
        prefix = raw_task["prefix"]
        suffix = raw_task["suffix"]
        return prefix, suffix, prefix + "<FILL_ME>" + suffix

    if "prompt" in raw_task and "suffix" in raw_task:
        prefix = raw_task["prompt"]
        suffix = raw_task["suffix"]
        return prefix, suffix, prefix + "<FILL_ME>" + suffix

    if "prompt" in raw_task and "<FILL_ME>" in raw_task["prompt"]:
        prefix, suffix = raw_task["prompt"].split("<FILL_ME>", 1)
        return prefix, suffix, raw_task["prompt"]

    if "prompt" in raw_task:
        prefix = raw_task["prompt"]
        return prefix, "", prefix + "<FILL_ME>"

    raise KeyError("Unable to infer prefix/suffix from dataset item.")


def normalize_humaneval_infilling_subset(dataset_subset: Optional[str]) -> str:
    if not dataset_subset:
        return "HumanEval-SingleLineInfilling"

    alias_map = {
        "singlelineinfilling": "HumanEval-SingleLineInfilling",
        "humanevalsinglelineinfilling": "HumanEval-SingleLineInfilling",
        "singleline": "HumanEval-SingleLineInfilling",
        "multilineinfilling": "HumanEval-MultiLineInfilling",
        "humanevalmultilineinfilling": "HumanEval-MultiLineInfilling",
        "multiline": "HumanEval-MultiLineInfilling",
        "randomspaninfilling": "HumanEval-RandomSpanInfilling",
        "humanevalrandomspaninfilling": "HumanEval-RandomSpanInfilling",
        "randomspan": "HumanEval-RandomSpanInfilling",
        "randomspanlight": "HumanEval-RandomSpanInfillingLight",
        "humanevalrandomspaninfillinglight": "HumanEval-RandomSpanInfillingLight",
    }
    key = dataset_subset.strip().replace("_", "").replace("-", "").lower()
    return alias_map.get(key, dataset_subset)


def load_humaneval_infilling(
    split: str = "test",
    max_samples: Optional[int] = None,
    dataset_subset: Optional[str] = "HumanEval-SingleLineInfilling",
) -> List[CodeTask]:
    from datasets import load_dataset

    normalized_subset = normalize_humaneval_infilling_subset(dataset_subset)
    dataset = load_dataset(
        "loubnabnl/humaneval_infilling",
        normalized_subset,
        trust_remote_code=True,
    )
    rows = dataset[split]
    upper = len(rows) if max_samples is None else min(max_samples, len(rows))

    tasks: List[CodeTask] = []
    for idx in range(upper):
        row = rows[idx]
        prefix, suffix, full_prompt = _extract_prefix_suffix(row)
        tasks.append(
            CodeTask(
                task_id=row.get("task_id", f"{split}_{idx}"),
                prefix=prefix,
                suffix=suffix,
                full_prompt=full_prompt,
                test_code=row.get("test", ""),
                entry_point=row.get("entry_point", ""),
                canonical_solution=row.get("canonical_solution"),
                raw=dict(row),
            )
        )
    return tasks


def _slice_middle_from_full_solution(full_solution: str, prefix: str, suffix: str) -> Optional[str]:
    if not isinstance(full_solution, str) or not full_solution:
        return None

    if not full_solution.startswith(prefix):
        return None
    if suffix and not full_solution.endswith(suffix):
        return None

    suffix_len = len(suffix)
    if suffix_len > 0:
        return full_solution[len(prefix) : len(full_solution) - suffix_len]
    return full_solution[len(prefix) :]


def infer_reference_middle_text(task: CodeTask) -> Optional[str]:
    raw = task.raw or {}

    # 1) Most direct case for HumanEval-Infilling:
    # canonical_solution is the target infill itself.
    canonical_solution = raw.get("canonical_solution", task.canonical_solution)
    if isinstance(canonical_solution, str) and canonical_solution:
        sliced = _slice_middle_from_full_solution(canonical_solution, task.prefix, task.suffix)
        if sliced is not None:
            return sliced
        return canonical_solution

    # 2) Other datasets may store the target infill directly under these keys.
    direct_middle_keys = [
        "completion",
        "target",
        "ground_truth",
        "middle",
        "masked_solution",
        "solution",
    ]
    for key in direct_middle_keys:
        value = raw.get(key)
        if isinstance(value, str) and value:
            sliced = _slice_middle_from_full_solution(value, task.prefix, task.suffix)
            if sliced is not None:
                return sliced
            return value

    # 3) Some datasets may store a full reconstructed code object.
    full_solution_keys = [
        "code",
        "full_solution",
    ]
    for key in full_solution_keys:
        value = raw.get(key)
        if isinstance(value, str) and value:
            sliced = _slice_middle_from_full_solution(value, task.prefix, task.suffix)
            if sliced is not None:
                return sliced

    return None


def compute_oracle_mask_length(task: CodeTask, tokenizer, add_special_tokens: bool = False) -> Optional[int]:
    reference_middle = infer_reference_middle_text(task)
    if reference_middle is None:
        return None
    return len(tokenizer.encode(reference_middle, add_special_tokens=add_special_tokens))