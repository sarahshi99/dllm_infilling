from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    model_path: str = "GSAI-ML/LLaDA-8B-Base"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class DataConfig:
    dataset_name: str = "loubnabnl/humaneval_infilling"
    dataset_subset: str = "HumanEval-SingleLineInfilling"
    split: str = "test"
    max_samples: Optional[int] = None


@dataclass
class DecodeConfig:
    fixed_mask_length: int = 64
    mask_length_source: str = "fixed"  # fixed | oracle | cal_lite
    total_steps: int = 64
    seed: int = 42
    add_special_tokens_to_prefix: bool = True
    add_special_tokens_to_suffix: bool = False
    save_step_traces: bool = False
    save_full_text_per_step: bool = False

    # CAL-lite diagnostic / v2 settings
    cal_lite_probe_lengths_csv: str = "3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,32,48,64"
    cal_lite_tie_break: str = "shorter"  # shorter | longer
    cal_lite_score_mode: str = "raw"  # raw | length_power | length_power_proportional
    cal_lite_length_alpha: float = 0.0
    cal_lite_length_prop_beta: float = 0.0
    cal_lite_length_prop_ref_length: float = 12.0
    cal_lite_length_prop_cap_length: Optional[float] = None


@dataclass
class LoggingConfig:
    output_dir: str = "outputs_clean"
    experiment_name: str = "vanilla_clean"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
