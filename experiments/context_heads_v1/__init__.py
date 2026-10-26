"""CPU-verified context heads; no backbone, data collection, or GPU runner."""
from .contract import HeadBatch, HeadConfig, HeadOutput, split_chains, history_for
from .heads import ContextHead, MODEL_NAMES

__all__ = ["HeadBatch", "HeadConfig", "HeadOutput", "ContextHead", "MODEL_NAMES",
           "split_chains", "history_for"]
