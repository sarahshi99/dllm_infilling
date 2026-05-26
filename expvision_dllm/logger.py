from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional


class JsonlLogger:
    def __init__(self, output_dir: str, experiment_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{timestamp}"
        self.run_dir = os.path.join(output_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.result_path = os.path.join(self.run_dir, "results.jsonl")
        self.trace_path = os.path.join(self.run_dir, "step_traces.jsonl")
        self.snapshot_record_path = os.path.join(self.run_dir, "snapshot_records.jsonl")
        self.selector_decision_path = os.path.join(self.run_dir, "selector_decisions.jsonl")
        self.trajectory_summary_path = os.path.join(self.run_dir, "trajectory_summaries.jsonl")
        self.config_path = os.path.join(self.run_dir, "config.json")

    def save_config(self, config: Any) -> None:
        payload = asdict(config) if is_dataclass(config) else config
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def log_result(self, payload: Dict[str, Any]) -> None:
        with open(self.result_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_trace(self, payload: Dict[str, Any]) -> None:
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_snapshot_records(self, payloads: Iterable[Dict[str, Any]]) -> None:
        with open(self.snapshot_record_path, "a", encoding="utf-8") as f:
            for payload in payloads:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_selector_decision(self, payload: Dict[str, Any]) -> None:
        with open(self.selector_decision_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_trajectory_summary(self, payload: Dict[str, Any]) -> None:
        with open(self.trajectory_summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save_text(self, filename: str, text: str) -> None:
        path = os.path.join(self.run_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def save_json(self, filename: str, obj: Dict[str, Any]) -> None:
        path = os.path.join(self.run_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)