from __future__ import annotations

import argparse

from fastquerydr.config import load_config
from fastquerydr.retrieval import run_retrieval_pipeline
from fastquerydr.utils.repro import prepare_run_dir, seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation for a trained bi-encoder")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path to load before retrieval evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config.experiment.seed)
    run_dir = prepare_run_dir(config.experiment.output_dir, f"{config.experiment.name}_retrieval")
    device = select_device(config.training.device)
    metrics = run_retrieval_pipeline(
        config=config,
        checkpoint_path=args.checkpoint,
        run_dir=run_dir,
        device=device,
    )
    print(metrics)


if __name__ == "__main__":
    main()
