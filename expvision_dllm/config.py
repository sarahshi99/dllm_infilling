from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    model_path: str = "GSAI-ML/LLaDA-8B-Base"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True


@dataclass
class DecodeConfig:
    num_mask_tokens: int = 64
    total_steps: int = 64
    seed: int = 42
    add_special_tokens_to_prefix: bool = True
    add_special_tokens_to_suffix: bool = False
    max_samples: Optional[int] = None
    shadow_mode: bool = False
    save_step_traces: bool = True
    use_fast_offsets_if_available: bool = True
    candidate_k: int = 1
    late_stage_candidate_k: int = 2
    ablation_mode: str = "A"

    # 旧实现没有显式数据子集字段，默认直接调用 load_dataset(...)
    # 删除原因：
    # 1. HumanEval-Infilling 有多个 subset；
    # 2. 如果不把 subset 写进配置和日志，后续结果口径会混淆；
    # 3. G 系列需要明确先在 SingleLineInfilling 上做收益归因。
    # dataset_subset: str = ""

    # 新增：显式记录数据子集，默认先对齐当前最常用的 SingleLine 口径。
    dataset_subset: str = "HumanEval-SingleLineInfilling"
    dataset_split: str = "test"

    # 新增：G 系列控制实验配置。
    # g_control_mode 用于区分 G0/G1/G2/G2-prime/G3。
    g_control_mode: str = "G0"

    # 新增：推断期 Tier 3 检查从后段开始触发，避免把 probe 过早塞进前期噪声阶段。
    inference_tier3_probe_start_ratio: float = 0.75
    inference_tier3_probe_stride: int = 4
    enable_inference_tier3_probe: bool = False
    early_stop_on_tier3_pass: bool = False

    # 新增：G3 多重重启配置。
    g3_num_restarts: int = 1
    g3_selection_rule: str = "proxy"
    g3_seed_stride: int = 1

    # 新增：G3 需要真正产生不同候选，因此单独暴露随机采样控制。
    # 设计原因：上一版 G3 只是换 seed，但 token-only argmax 解码本身是确定性的，
    # 不会真正产生 restart 多样性，因此 G3 没有实验意义。
    g3_do_sample: bool = True
    g3_top_k: int = 5
    g3_temperature: float = 1.0
    g3_sample_start_ratio: float = 0.0

    # 新增：R/M 系列——单次晚期修补框架配置。
    # 设计原因：
    # 1. 让 repair 最多只发生一次，避免多次 repair 使定位器/执行器效应缠绕；
    # 2. R 系列固定执行器，只测触发器/定位器；
    # 3. M 系列固定定位器，只测 merge/materialization。
    rm_control_mode: str = "R0"
    rm_probe_start_ratio: float = 0.75
    rm_probe_stride: int = 4
    rm_repair_max_token_span: int = 12
    rm_repair_enabled: bool = False
    rm_trigger_requires_tier12_pass: bool = True
    rm_executor_mode: str = "strict_span_only"
    rm_locator_mode: str = "subtree_locator"

    # 新增：S 系列——单轨迹 snapshot 选择配置。
    s_control_mode: str = "S0"

    # =========================
    # 旧实现（保留对照，不再直接使用）
    # =========================
    # s_snapshot_start_ratio: float = 0.75
    # s_snapshot_stride: int = 4
    # s_consistency_window: int = 2
    #
    # 删除原因：
    # 1. 当前采样窗口已经冻结成显式 sampling policy，而不是 start_ratio/stride 的隐式组合；
    # 2. 若继续只保留 ratio/stride，后续 total_steps 改动时很容易再次发生分析线与方法线漂移。

    # 新增：S 系列显式采样策略与特征/版本字段。
    s_sampling_policy_version: str = "late_uniform_5points_v1"
    s_consistency_window: int = 2
    s_recent_window: int = 3
    s_low_conf_threshold: float = 0.70
    s_feature_schema_version: str = "snapshot_schema_v2"
    s_selector_version: str = "selector_v2"

    # 新增：为后续 oracle gap / trajectory summary 预留版本字段。
    verifier_version: str = "verifier_v1"
    baseline_chain_version: str = "s_chain_v2"

    # 新增：是否在方法链中直接为每个 snapshot 运行 Tier3。
    # 默认关闭，避免无官方测试方法线被额外的中间 Tier3 成本拖慢；
    # analysis 线如果需要 oracle gap，可单独基于保存的 snapshot 离线重算。
    s_capture_snapshot_tier3_summary: bool = False


@dataclass
class TierScheduleConfig:
    blind_ratio: float = 0.4
    structure_ratio: float = 0.4
    semantic_ratio: float = 0.2

    def validate(self) -> None:
        total = self.blind_ratio + self.structure_ratio + self.semantic_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"TierScheduleConfig ratios must sum to 1.0, got {total}")


@dataclass
class VerifierConfig:
    timeout: float = 3.0
    enable_tier1_parse: bool = True
    enable_tier2_exec: bool = True
    enable_tier3_tests: bool = True
    compile_mode: str = "exec"
    parser_backend: str = "python_ast"
    schedule: TierScheduleConfig = field(default_factory=TierScheduleConfig)


@dataclass
class PolicyConfig:
    base_policy: str = "confidence"
    structure_policy: str = "limited_hierarchy"
    default_unit_for_parse_error: str = "statement"
    default_unit_for_test_error: str = "subtree"
    escalate_after_failed_attempts: int = 2
    cap_structural_units_per_step: int = 2
    min_confidence_for_lock: float = 0.80


@dataclass
class LoggingConfig:
    output_dir: str = "outputs"
    experiment_name: str = "default_exp"
    write_jsonl: bool = True
    write_predictions: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)