from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Iterable


class JsonlLogger:
    def __init__(self, output_dir: str, experiment_name: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{timestamp}"
        self.run_dir = os.path.join(output_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.result_path = os.path.join(self.run_dir, "results.jsonl")
        self.trace_path = os.path.join(self.run_dir, "step_traces.jsonl")
        self.config_path = os.path.join(self.run_dir, "config.json")

    def save_config(self, config: Any) -> None:
        payload = asdict(config) if is_dataclass(config) else config
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def log_result(self, payload: Dict[str, Any]) -> None:
        with open(self.result_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_trace(self, payload: Dict[str, Any]) -> None:
        with open(self.trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save_json(self, filename: str, obj: Dict[str, Any]) -> None:
        path = os.path.join(self.run_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, ensure_ascii=False, indent=2)
