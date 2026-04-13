from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    name: str
    output_dir: str
    seed: int


@dataclass
class ModelConfig:
    encoder_name: str
    pooling: str = "mean"
    normalize: bool = True


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


@dataclass
class AppConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig

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
    )
