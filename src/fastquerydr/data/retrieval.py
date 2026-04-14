from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextRecord:
    record_id: str
    text: str


def _iter_tsv_rows(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line:
                yield line.split("\t")


def load_id_text_tsv(path: str | Path) -> list[TextRecord]:
    records: list[TextRecord] = []
    for row in _iter_tsv_rows(path):
        if row[0].lower() in {"passage_id", "query_id", "doc_id", "id"}:
            continue
        if len(row) < 2:
            raise ValueError(f"Expected at least 2 tab-separated columns in {path}, got {len(row)}")
        records.append(TextRecord(record_id=row[0], text="\t".join(row[1:])))

    if not records:
        raise ValueError(f"No records loaded from {path}")
    return records


def load_qrels(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for row in _iter_tsv_rows(path):
        if row[0].lower() == "query_id":
            continue
        if len(row) >= 4:
            query_id, _, doc_id, relevance = row[:4]
            if int(relevance) <= 0:
                continue
        elif len(row) >= 2:
            query_id, doc_id = row[:2]
        else:
            raise ValueError(f"Expected at least 2 columns in qrels file {path}, got {len(row)}")
        qrels.setdefault(query_id, set()).add(doc_id)

    if not qrels:
        raise ValueError(f"No qrels loaded from {path}")
    return qrels
