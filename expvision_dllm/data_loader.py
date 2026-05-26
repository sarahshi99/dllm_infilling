from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

from datasets import load_dataset


@dataclass
class CodeTask:
    task_id: str
    prefix: str
    suffix: str
    full_prompt: str
    test_code: str
    entry_point: str
    canonical_solution: Optional[str] = None
    raw: Optional[Dict] = None


def _extract_prefix_suffix(raw_task: Dict) -> tuple[str, str, str]:
    """
    Normalizes slightly different FIM dataset formats.
    Preference order:
    1. explicit prefix/suffix
    2. prompt + suffix
    3. prompt containing <FILL_ME>
    """
    if "prefix" in raw_task and "suffix" in raw_task:
        prefix = raw_task["prefix"]
        suffix = raw_task["suffix"]
        full_prompt = prefix + "<FILL_ME>" + suffix
        return prefix, suffix, full_prompt

    if "prompt" in raw_task and "suffix" in raw_task:
        prefix = raw_task["prompt"]
        suffix = raw_task["suffix"]
        full_prompt = prefix + "<FILL_ME>" + suffix
        return prefix, suffix, full_prompt

    if "prompt" in raw_task and "<FILL_ME>" in raw_task["prompt"]:
        prefix, suffix = raw_task["prompt"].split("<FILL_ME>", 1)
        return prefix, suffix, raw_task["prompt"]

    if "prompt" in raw_task:
        # Fallback: treat prompt as prefix-only generation.
        return raw_task["prompt"], "", raw_task["prompt"] + "<FILL_ME>"

    raise KeyError("Unable to infer prefix/suffix from dataset item.")


# 旧实现：
# def load_humaneval_infilling(split: str = "test", max_samples: Optional[int] = None) -> List[CodeTask]:
#     dataset = load_dataset("loubnabnl/humaneval_infilling", trust_remote_code=True)
#
# 删除原因：
# 1. 这会隐式走默认 subset，结果口径不透明；
# 2. G 系列实验需要显式指定 SingleLine / MultiLine / RandomSpan；
# 3. 不同 Hugging Face 文档和本地口头习惯的 subset 名称可能带别名，需要统一归一化。


def normalize_humaneval_infilling_subset(dataset_subset: Optional[str]) -> str:
    if not dataset_subset:
        return "HumanEval-SingleLineInfilling"

    alias_map = {
        "singlelineinfilling": "HumanEval-SingleLineInfilling",
        "humaneval-singlelineinfilling": "HumanEval-SingleLineInfilling",
        "singleline": "HumanEval-SingleLineInfilling",
        "multilineinfilling": "HumanEval-MultiLineInfilling",
        "humaneval-multilineinfilling": "HumanEval-MultiLineInfilling",
        "multiline": "HumanEval-MultiLineInfilling",
        "randomspaninfilling": "HumanEval-RandomSpanInfilling",
        "humaneval-randomspaninfilling": "HumanEval-RandomSpanInfilling",
        "randomspan": "HumanEval-RandomSpanInfilling",
        "randomspanlight": "HumanEval-RandomSpanInfillingLight",
        "humaneval-randomspaninfillinglight": "HumanEval-RandomSpanInfillingLight",
    }
    key = dataset_subset.strip().replace("_", "").replace("-", "").lower()
    return alias_map.get(key, dataset_subset)


def load_humaneval_infilling(
    split: str = "test",
    max_samples: Optional[int] = None,
    dataset_subset: Optional[str] = "HumanEval-SingleLineInfilling",
) -> List[CodeTask]:
    normalized_subset = normalize_humaneval_infilling_subset(dataset_subset)
    dataset = load_dataset(
        "loubnabnl/humaneval_infilling",
        normalized_subset,
        trust_remote_code=True,
    )
    rows = dataset[split]
    tasks: List[CodeTask] = []
    upper = len(rows) if max_samples is None else min(max_samples, len(rows))

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
                raw=row,
            )
        )
    return tasks
