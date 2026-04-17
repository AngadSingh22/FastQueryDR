from __future__ import annotations

import argparse
import json

from fastquerydr.config import load_config
from fastquerydr.retrieval.ann_eval import run_ann_comparison
from fastquerydr.utils.repro import prepare_run_dir, seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exact-vs-ANN comparison for a FastQueryDR checkpoint")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path to load")
    parser.add_argument("--query-limit", type=int, default=None, help="Optional limit on evaluated queries")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.experiment.seed)
    run_dir = prepare_run_dir(config.experiment.output_dir, f"{config.experiment.name}_ann")
    device = select_device(config.training.device)
    metrics = run_ann_comparison(
        config=config,
        checkpoint_path=args.checkpoint,
        run_dir=run_dir,
        device=device,
        query_limit=args.query_limit,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
