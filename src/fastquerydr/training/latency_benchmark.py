from __future__ import annotations

import argparse
import json

from fastquerydr.config import load_config
from fastquerydr.retrieval.pipeline import prepare_retrieval_artifacts
from fastquerydr.retrieval import benchmark_latency
from fastquerydr.utils.repro import prepare_run_dir, seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 3 latency benchmark")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path to load before benchmarking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config.experiment.seed)
    run_dir = prepare_run_dir(config.experiment.output_dir, f"{config.experiment.name}_latency")
    device = select_device(config.training.device)
    artifacts = prepare_retrieval_artifacts(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
    )
    metrics = benchmark_latency(
        config=config,
        model=artifacts.model,
        tokenizer=artifacts.tokenizer,
        query_ids=artifacts.query_ids,
        query_texts=artifacts.query_texts,
        index=artifacts.index,
        run_dir=run_dir,
        device=device,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
