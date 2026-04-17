from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from fastquerydr.config import ModelConfig


def _pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
    normalize: bool,
) -> torch.Tensor:
    if pooling == "cls":
        embeddings = hidden_states[:, 0]
    elif pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    else:
        raise ValueError(f"Unsupported pooling: {pooling}")

    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings


def _truncate_encoder_layers(encoder: nn.Module, num_hidden_layers: int) -> None:
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")
    if not hasattr(encoder, "encoder") or not hasattr(encoder.encoder, "layer"):
        raise ValueError("Query-side truncation currently expects a transformer with encoder.layer")
    total_layers = len(encoder.encoder.layer)
    if num_hidden_layers > total_layers:
        raise ValueError(f"Requested {num_hidden_layers} layers, but encoder has only {total_layers}")
    encoder.encoder.layer = nn.ModuleList(list(encoder.encoder.layer[:num_hidden_layers]))
    if hasattr(encoder.config, "num_hidden_layers"):
        encoder.config.num_hidden_layers = num_hidden_layers


def _activation_module(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported projection activation: {name}")


def _build_projection_head(
    hidden_size: int,
    output_dim: int | None,
    activation: str,
) -> nn.Module | None:
    if output_dim is None:
        return None
    return nn.Sequential(
        nn.Linear(hidden_size, output_dim),
        _activation_module(activation),
        nn.Linear(output_dim, hidden_size),
    )


class SymmetricBiEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        pooling: str = "mean",
        normalize: bool = True,
        query_pooling: str | None = None,
        passage_pooling: str | None = None,
        query_projection_dim: int | None = None,
        query_projection_activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.pooling = pooling
        self.query_pooling = query_pooling or pooling
        self.passage_pooling = passage_pooling or pooling
        self.normalize = normalize
        self.query_projection = _build_projection_head(
            hidden_size=self.encoder.config.hidden_size,
            output_dim=query_projection_dim,
            activation=query_projection_activation,
        )
        self.temperature = nn.Parameter(torch.tensor(0.0))

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**inputs)
        return _pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"], self.query_pooling, self.normalize)

    def encode_query(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**inputs)
        embeddings = _pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"], self.query_pooling, normalize=False)
        if self.query_projection is not None:
            embeddings = self.query_projection(embeddings)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode_passage(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**inputs)
        return _pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"], self.passage_pooling, self.normalize)

    def similarity(self, query_embeddings: torch.Tensor, passage_embeddings: torch.Tensor) -> torch.Tensor:
        scale = self.temperature.exp().clamp(max=100.0)
        return query_embeddings @ passage_embeddings.transpose(0, 1) * scale

    def forward(self, query_inputs: dict[str, torch.Tensor], passage_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        queries = self.encode_query(query_inputs)
        passages = self.encode_passage(passage_inputs)
        return self.similarity(queries, passages)


class AsymmetricBiEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        pooling: str = "mean",
        normalize: bool = True,
        query_pooling: str | None = None,
        passage_pooling: str | None = None,
        query_num_hidden_layers: int | None = None,
        freeze_document_encoder: bool = True,
        query_projection_dim: int | None = None,
        query_projection_activation: str = "gelu",
    ) -> None:
        super().__init__()
        passage_encoder = AutoModel.from_pretrained(encoder_name)
        self.query_encoder = copy.deepcopy(passage_encoder)
        self.passage_encoder = passage_encoder
        if query_num_hidden_layers is not None:
            _truncate_encoder_layers(self.query_encoder, query_num_hidden_layers)
        if freeze_document_encoder:
            for parameter in self.passage_encoder.parameters():
                parameter.requires_grad = False
        self.pooling = pooling
        self.query_pooling = query_pooling or pooling
        self.passage_pooling = passage_pooling or pooling
        self.normalize = normalize
        self.freeze_document_encoder = freeze_document_encoder
        self.query_projection = _build_projection_head(
            hidden_size=self.query_encoder.config.hidden_size,
            output_dim=query_projection_dim,
            activation=query_projection_activation,
        )
        self.temperature = nn.Parameter(torch.tensor(0.0))

    def _encode_with_encoder(self, encoder: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = encoder(**inputs)
        return _pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"], self.pooling, self.normalize)

    def encode_query(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.query_encoder(**inputs)
        embeddings = _pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"], self.query_pooling, normalize=False)
        if self.query_projection is not None:
            embeddings = self.query_projection(embeddings)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode_passage(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.passage_encoder(**inputs)
        return _pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"], self.passage_pooling, self.normalize)

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode_query(inputs)

    def similarity(self, query_embeddings: torch.Tensor, passage_embeddings: torch.Tensor) -> torch.Tensor:
        scale = self.temperature.exp().clamp(max=100.0)
        return query_embeddings @ passage_embeddings.transpose(0, 1) * scale

    def forward(self, query_inputs: dict[str, torch.Tensor], passage_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        queries = self.encode_query(query_inputs)
        passages = self.encode_passage(passage_inputs)
        return self.similarity(queries, passages)


def build_bi_encoder(model_config: ModelConfig) -> nn.Module:
    if model_config.architecture == "symmetric":
        return SymmetricBiEncoder(
            encoder_name=model_config.encoder_name,
            pooling=model_config.pooling,
            query_pooling=model_config.query_pooling,
            passage_pooling=model_config.passage_pooling,
            normalize=model_config.normalize,
            query_projection_dim=model_config.query_projection_dim,
            query_projection_activation=model_config.query_projection_activation,
        )
    if model_config.architecture == "asymmetric":
        return AsymmetricBiEncoder(
            encoder_name=model_config.encoder_name,
            pooling=model_config.pooling,
            query_pooling=model_config.query_pooling,
            passage_pooling=model_config.passage_pooling,
            normalize=model_config.normalize,
            query_num_hidden_layers=model_config.query_num_hidden_layers,
            freeze_document_encoder=model_config.freeze_document_encoder,
            query_projection_dim=model_config.query_projection_dim,
            query_projection_activation=model_config.query_projection_activation,
        )
    raise ValueError(f"Unsupported model architecture: {model_config.architecture}")
