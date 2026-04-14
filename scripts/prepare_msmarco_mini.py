from __future__ import annotations

import argparse
import gzip
import json
import random
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a manageable MS MARCO passage-ranking slice")
    parser.add_argument("--raw-dir", default="data/msmarco/raw")
    parser.add_argument("--output-dir", default="data/msmarco")
    parser.add_argument("--max-triples", type=int, default=100000)
    parser.add_argument("--random-docs", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def extract_member(archive_path: Path, member_name: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember(member_name)
        with archive.extractfile(member) as source, destination.open("wb") as target:
            if source is None:
                raise ValueError(f"Failed to extract {member_name} from {archive_path}")
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
    return destination


def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                queries[parts[0]] = "\t".join(parts[1:])
    return queries


def load_dev_qrels(path: Path) -> tuple[dict[str, set[str]], set[str], set[str]]:
    qrels: dict[str, set[str]] = {}
    query_ids: set[str] = set()
    passage_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            qid, _, pid, rel = parts[:4]
            if int(rel) <= 0:
                continue
            qrels.setdefault(qid, set()).add(pid)
            query_ids.add(qid)
            passage_ids.add(pid)
    return qrels, query_ids, passage_ids


def load_train_triples(
    path: Path,
    query_lookup: dict[str, str],
    needed_passage_ids: set[str],
    max_triples: int,
) -> list[tuple[str, str, str]]:
    triples: list[tuple[str, str, str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, pos_pid, neg_pid = parts[:3]
            query_text = query_lookup.get(qid)
            if query_text is None:
                continue
            triples.append((qid, pos_pid, neg_pid))
            needed_passage_ids.add(pos_pid)
            needed_passage_ids.add(neg_pid)
            if len(triples) >= max_triples:
                break
    if len(triples) < max_triples:
        raise ValueError(f"Requested {max_triples} triples but found only {len(triples)}")
    return triples


def sample_collection(
    path: Path,
    required_ids: set[str],
    random_docs: int,
    seed: int,
) -> tuple[dict[str, str], int]:
    rng = random.Random(seed)
    corpus: dict[str, str] = {}
    reservoir: list[tuple[str, str]] = []
    seen_optional = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            pid = parts[0]
            passage = "\t".join(parts[1:])
            if pid in required_ids:
                corpus[pid] = passage
                continue

            seen_optional += 1
            if len(reservoir) < random_docs:
                reservoir.append((pid, passage))
            else:
                replacement_index = rng.randint(0, seen_optional - 1)
                if replacement_index < random_docs:
                    reservoir[replacement_index] = (pid, passage)

    for pid, passage in reservoir:
        corpus.setdefault(pid, passage)

    missing = len(required_ids - set(corpus))
    return corpus, missing


def write_train_triples(
    output_path: Path,
    triples: list[tuple[str, str, str]],
    query_lookup: dict[str, str],
    corpus_lookup: dict[str, str],
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for qid, pos_pid, neg_pid in triples:
            query_text = query_lookup[qid]
            positive = corpus_lookup.get(pos_pid)
            negative = corpus_lookup.get(neg_pid)
            if positive is None or negative is None:
                continue
            handle.write(f"{query_text}\t{positive}\t{negative}\n")


def write_corpus(output_path: Path, corpus_lookup: dict[str, str]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for pid in sorted(corpus_lookup, key=int):
            handle.write(f"{pid}\t{corpus_lookup[pid]}\n")


def write_dev_queries(output_path: Path, query_lookup: dict[str, str], query_ids: set[str]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for qid in sorted(query_ids, key=int):
            query_text = query_lookup.get(qid)
            if query_text is not None:
                handle.write(f"{qid}\t{query_text}\n")


def write_dev_qrels(output_path: Path, qrels: dict[str, set[str]], corpus_lookup: dict[str, str]) -> int:
    kept = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for qid in sorted(qrels, key=int):
            for pid in sorted(qrels[qid], key=int):
                if pid in corpus_lookup:
                    handle.write(f"{qid}\t0\t{pid}\t1\n")
                    kept += 1
    return kept


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collection_archive = raw_dir / "collection.tar.gz"
    queries_archive = raw_dir / "queries.tar.gz"
    train_triples_path = raw_dir / "qidpidtriples.train.full.2.tsv.gz"
    dev_qrels_path = raw_dir / "qrels.dev.tsv"

    collection_tsv = raw_dir / "collection.tsv"
    queries_train_tsv = raw_dir / "queries.train.tsv"
    queries_dev_tsv = raw_dir / "queries.dev.tsv"

    if not collection_tsv.exists():
        extract_member(collection_archive, "collection.tsv", collection_tsv)
    if not queries_train_tsv.exists():
        extract_member(queries_archive, "queries.train.tsv", queries_train_tsv)
    if not queries_dev_tsv.exists():
        extract_member(queries_archive, "queries.dev.tsv", queries_dev_tsv)

    train_queries = load_queries(queries_train_tsv)
    dev_queries = load_queries(queries_dev_tsv)
    qrels, dev_query_ids, required_passage_ids = load_dev_qrels(dev_qrels_path)

    triples = load_train_triples(
        path=train_triples_path,
        query_lookup=train_queries,
        needed_passage_ids=required_passage_ids,
        max_triples=args.max_triples,
    )

    corpus_lookup, missing_required = sample_collection(
        path=collection_tsv,
        required_ids=required_passage_ids,
        random_docs=args.random_docs,
        seed=args.seed,
    )
    if missing_required:
        raise ValueError(f"Missing {missing_required} required passages from collection scan")

    train_output = output_dir / "train_triples.tsv"
    corpus_output = output_dir / "corpus.tsv"
    dev_queries_output = output_dir / "dev_queries.tsv"
    dev_qrels_output = output_dir / "dev_qrels.tsv"

    write_train_triples(train_output, triples, train_queries, corpus_lookup)
    write_corpus(corpus_output, corpus_lookup)
    write_dev_queries(dev_queries_output, dev_queries, dev_query_ids)
    kept_qrels = write_dev_qrels(dev_qrels_output, qrels, corpus_lookup)

    manifest = {
        "max_triples_requested": args.max_triples,
        "random_docs_requested": args.random_docs,
        "train_triples_written": sum(1 for _ in train_output.open("r", encoding="utf-8")),
        "corpus_docs_written": len(corpus_lookup),
        "dev_queries_written": sum(1 for _ in dev_queries_output.open("r", encoding="utf-8")),
        "dev_qrels_written": kept_qrels,
        "seed": args.seed,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
