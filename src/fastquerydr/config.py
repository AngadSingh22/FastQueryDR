from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ExperimentConfig:
    name: str
    output_dir: str
    seed: int


@dataclass
class ModelConfig:
    encoder_name: str
    architecture: str = "symmetric"
    pooling: str = "mean"
    normalize: bool = True
    query_num_hidden_layers: Optional[int] = None
    freeze_document_encoder: bool = False


@dataclass
class DataConfig:
    train_path: str
    max_train_examples: int
    val_examples: int
    text_max_length: int
    query_prefix: str = ""
    passage_prefix: str = ""


@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    grad_accumulation_steps: int
    log_every_steps: int
    eval_every_steps: int
    max_grad_norm: float
    num_workers: int
    mixed_precision: bool
    device: str = "auto"
    max_steps: Optional[int] = None
    best_checkpoint_metric: str = "val_loss"


@dataclass
class LatencyConfig:
    enabled: bool = False
    warmup_queries: int = 20
    measured_queries: int = 100
    query_batch_size: int = 1
    search_top_k: int = 100


@dataclass
class RetrievalConfig:
    enabled: bool = False
    corpus_path: str = ""
    query_path: str = ""
    qrels_path: str = ""
    batch_size: int = 64
    top_k: int = 100
    save_embeddings: bool = False
    latency: Optional[LatencyConfig] = None
    selection: Optional["RetrievalSelectionConfig"] = None


@dataclass
class RetrievalSelectionConfig:
    enabled: bool = False
    query_limit: int = 100
    corpus_size: int = 5000
    top_k: int = 100
    patience: int = 2


@dataclass
class AppConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    retrieval: Optional[RetrievalConfig] = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> AppConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    return AppConfig(
        experiment=ExperimentConfig(**raw["experiment"]),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        training=TrainingConfig(**raw["training"]),
        retrieval=(
            RetrievalConfig(
                **{
                    **raw["retrieval"],
                    "latency": LatencyConfig(**raw["retrieval"]["latency"])
                    if "latency" in raw["retrieval"]
                    else None,
                    "selection": RetrievalSelectionConfig(**raw["retrieval"]["selection"])
                    if "selection" in raw["retrieval"]
                    else None,
                }
            )
            if "retrieval" in raw
            else None
        ),
    )
