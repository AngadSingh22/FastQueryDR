from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass
class TripleExample:
    query: str
    positive: str
    negative: str


class TriplesDataset(Dataset):
    def __init__(self, examples: list[TripleExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TripleExample:
        return self.examples[index]


def _read_triples(path: str | Path, limit: int) -> list[TripleExample]:
    examples: list[TripleExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"Expected 3 tab-separated fields, got {len(parts)} in line: {line[:120]}")
            query, positive, negative = parts[:3]
            examples.append(TripleExample(query=query, positive=positive, negative=negative))
            if len(examples) >= limit:
                break

    if not examples:
        raise ValueError(f"No training triples were loaded from {path}")
    return examples


def build_train_val_datasets(path: str | Path, max_examples: int, val_examples: int) -> tuple[TriplesDataset, TriplesDataset]:
    if val_examples >= max_examples:
        raise ValueError("val_examples must be smaller than max_train_examples")

    examples = _read_triples(path=path, limit=max_examples)
    if len(examples) <= val_examples:
        raise ValueError("Loaded examples are insufficient to create the requested validation split")

    val_set = TriplesDataset(examples[:val_examples])
    train_set = TriplesDataset(examples[val_examples:])
    return train_set, val_set


class TriplesCollator:
    def __init__(
        self,
        tokenizer,
        text_max_length: int,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )

    def __call__(self, batch: list[TripleExample]) -> dict[str, dict[str, torch.Tensor]]:
        queries = [f"{self.query_prefix}{item.query}" for item in batch]
        positives = [f"{self.passage_prefix}{item.positive}" for item in batch]
        negatives = [f"{self.passage_prefix}{item.negative}" for item in batch]

        return {
            "query_inputs": self._tokenize(queries),
            "positive_inputs": self._tokenize(positives),
            "negative_inputs": self._tokenize(negatives),
        }
