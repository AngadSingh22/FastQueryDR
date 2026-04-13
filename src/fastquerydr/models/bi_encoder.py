from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


class SymmetricBiEncoder(nn.Module):
    def __init__(self, encoder_name: str, pooling: str = "mean", normalize: bool = True) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**inputs)
        hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        if self.pooling == "cls":
            embeddings = hidden_states[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def forward(self, query_inputs: dict[str, torch.Tensor], passage_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        queries = self.encode(query_inputs)
        passages = self.encode(passage_inputs)
        scale = self.temperature.exp().clamp(max=100.0)
        return queries @ passages.transpose(0, 1) * scale
