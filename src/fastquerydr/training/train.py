from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from fastquerydr.config import AppConfig, load_config
from fastquerydr.data import TriplesCollator, build_train_val_datasets
from fastquerydr.models import build_bi_encoder
from fastquerydr.retrieval import run_retrieval_pipeline
from fastquerydr.retrieval.index import build_flat_ip_index
from fastquerydr.retrieval.metrics import mean_reciprocal_rank_at_k, recall_at_k
from fastquerydr.retrieval.pipeline import encode_texts, rank_documents
from fastquerydr.retrieval.probe import RetrievalProbe, build_retrieval_probe
from fastquerydr.utils.repro import prepare_run_dir, seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a configured bi-encoder")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, dict[str, torch.Tensor]], device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    moved: dict[str, dict[str, torch.Tensor]] = {}
    for key, inputs in batch.items():
        moved[key] = {name: tensor.to(device) for name, tensor in inputs.items()}
    return moved


def build_dataloaders(config: AppConfig, tokenizer):
    train_dataset, val_dataset = build_train_val_datasets(
        path=config.data.train_path,
        max_examples=config.data.max_train_examples,
        val_examples=config.data.val_examples,
        seed=config.experiment.seed,
    )
    collator = TriplesCollator(
        tokenizer=tokenizer,
        text_max_length=config.data.text_max_length,
        query_prefix=config.data.query_prefix,
        passage_prefix=config.data.passage_prefix,
        include_negatives=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=collator,
    )
    return train_loader, val_loader


def compute_loss(model: nn.Module, batch: dict[str, dict[str, torch.Tensor]], criterion: nn.Module) -> torch.Tensor:
    query_embeddings = model.encode_query(batch["query_inputs"])
    positive_embeddings = model.encode_passage(batch["positive_inputs"])
    logits = model.similarity(query_embeddings, positive_embeddings)
    labels = torch.arange(logits.size(0), device=logits.device)
    return criterion(logits, labels)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        loss = compute_loss(model, batch, criterion)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_metadata(run_dir: Path, config: AppConfig) -> None:
    with (run_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)


def build_training_probe(config: AppConfig) -> Optional[RetrievalProbe]:
    selection = config.retrieval.selection if config.retrieval is not None else None
    if config.retrieval is None or selection is None or not selection.enabled:
        return None
    return build_retrieval_probe(
        corpus_path=config.retrieval.corpus_path,
        query_path=config.retrieval.query_path,
        qrels_path=config.retrieval.qrels_path,
        query_limit=selection.query_limit,
        corpus_size=selection.corpus_size,
        top_k=selection.top_k,
        seed=config.experiment.seed,
    )


@torch.no_grad()
def evaluate_retrieval_probe(
    *,
    model: nn.Module,
    tokenizer,
    config: AppConfig,
    probe: RetrievalProbe,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    corpus_embeddings = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=probe.corpus_texts,
        prefix=config.data.passage_prefix,
        encoder_role="passage",
        max_length=config.data.text_max_length,
        batch_size=min(config.retrieval.batch_size, 64),
        device=device,
    )
    index = build_flat_ip_index(corpus_embeddings)
    query_embeddings = encode_texts(
        model=model,
        tokenizer=tokenizer,
        texts=probe.query_texts,
        prefix=config.data.query_prefix,
        encoder_role="query",
        max_length=config.data.text_max_length,
        batch_size=min(config.retrieval.batch_size, 64),
        device=device,
    )
    ranked_doc_ids = rank_documents(index, query_embeddings, probe.corpus_ids, probe.top_k)
    metrics = {
        "retrieval_mrr_at_10": mean_reciprocal_rank_at_k(ranked_doc_ids, probe.qrels, probe.query_ids, k=min(10, probe.top_k)),
        "retrieval_recall_at_100": recall_at_k(ranked_doc_ids, probe.qrels, probe.query_ids, k=min(100, probe.top_k)),
    }
    model.train()
    return metrics


def metric_improved(metric_name: str, best_value: Optional[float], current_value: float) -> bool:
    if best_value is None:
        return True
    if metric_name == "val_loss":
        return current_value < best_value
    return current_value > best_value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config.experiment.seed)
    run_dir = prepare_run_dir(config.experiment.output_dir, config.experiment.name)
    save_metadata(run_dir, config)

    device = select_device(config.training.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)
    train_loader, val_loader = build_dataloaders(config, tokenizer)
    retrieval_probe = build_training_probe(config)

    model = build_bi_encoder(config.model).to(device)

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable_parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    total_train_steps = len(train_loader) * config.training.num_epochs
    if config.training.max_steps is not None:
        total_train_steps = min(total_train_steps, config.training.max_steps)
    total_updates = math.ceil(total_train_steps / config.training.grad_accumulation_steps)
    warmup_steps = int(total_updates * config.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_updates, 1),
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=config.training.mixed_precision and device.type == "cuda")

    lowest_val_loss = float("inf")
    best_checkpoint_val_loss: Optional[float] = None
    best_metric_value: Optional[float] = None
    best_retrieval_metrics: Optional[dict[str, float]] = None
    checkpoint_metric = config.training.best_checkpoint_metric
    if checkpoint_metric != "val_loss" and retrieval_probe is None:
        raise ValueError("Non-loss checkpoint selection requires retrieval.selection.enabled")
    early_stopping_patience = config.retrieval.selection.patience if retrieval_probe is not None else None
    non_improving_evals = 0
    stopped_early = False
    stopped_step: Optional[int] = None
    global_step = 0
    progress = tqdm(total=total_train_steps, desc="training", leave=False)
    stop_training = False
    checkpoint_path = run_dir / "best_model.pt"

    for epoch in range(config.training.num_epochs):
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            batch = move_batch_to_device(batch, device)

            with torch.amp.autocast(device_type=device.type, enabled=config.training.mixed_precision and device.type == "cuda"):
                loss = compute_loss(model, batch, criterion)
                scaled_loss = loss / config.training.grad_accumulation_steps

            scaler.scale(scaled_loss).backward()

            if step % config.training.grad_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % config.training.log_every_steps == 0:
                progress.set_postfix(epoch=epoch + 1, loss=f"{loss.item():.4f}")

            if global_step % config.training.eval_every_steps == 0:
                val_loss = evaluate(model, val_loader, criterion, device)
                lowest_val_loss = min(lowest_val_loss, val_loss)
                retrieval_metrics = (
                    evaluate_retrieval_probe(model=model, tokenizer=tokenizer, config=config, probe=retrieval_probe, device=device)
                    if retrieval_probe is not None
                    else None
                )
                metric_values = {"val_loss": val_loss}
                if retrieval_metrics is not None:
                    metric_values.update(retrieval_metrics)
                current_metric_value = metric_values[checkpoint_metric]
                if metric_improved(checkpoint_metric, best_metric_value, current_metric_value):
                    best_metric_value = current_metric_value
                    best_checkpoint_val_loss = val_loss
                    best_retrieval_metrics = retrieval_metrics
                    torch.save(model.state_dict(), checkpoint_path)
                    non_improving_evals = 0
                else:
                    non_improving_evals += 1

                message = f"step={global_step} train_loss={loss.item():.4f} val_loss={val_loss:.4f}"
                if retrieval_metrics is not None:
                    message += (
                        f" retrieval_mrr_at_10={retrieval_metrics['retrieval_mrr_at_10']:.4f}"
                        f" retrieval_recall_at_100={retrieval_metrics['retrieval_recall_at_100']:.4f}"
                    )
                tqdm.write(message)

                if early_stopping_patience is not None and non_improving_evals >= early_stopping_patience:
                    stopped_early = True
                    stopped_step = global_step
                    stop_training = True
                    break

            if config.training.max_steps is not None and global_step >= config.training.max_steps:
                stop_training = True
                stopped_step = global_step
                break

            progress.update(1)

        if len(train_loader) % config.training.grad_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if stop_training:
            break

    final_val_loss = evaluate(model, val_loader, criterion, device)
    torch.save(model.state_dict(), run_dir / "last_model.pt")
    final_retrieval_metrics = (
        evaluate_retrieval_probe(model=model, tokenizer=tokenizer, config=config, probe=retrieval_probe, device=device)
        if retrieval_probe is not None
        else None
    )
    final_metric_values = {"val_loss": final_val_loss}
    if final_retrieval_metrics is not None:
        final_metric_values.update(final_retrieval_metrics)
    if metric_improved(checkpoint_metric, best_metric_value, final_metric_values[checkpoint_metric]):
        best_metric_value = final_metric_values[checkpoint_metric]
        best_checkpoint_val_loss = final_val_loss
        best_retrieval_metrics = final_retrieval_metrics
        torch.save(model.state_dict(), checkpoint_path)

    metrics = {
        "final_val_loss": final_val_loss,
        "best_val_loss": lowest_val_loss,
        "best_checkpoint_val_loss": best_checkpoint_val_loss,
        "best_checkpoint_metric": checkpoint_metric,
        "best_checkpoint_metric_value": best_metric_value,
        "best_retrieval_metrics": best_retrieval_metrics,
        "final_retrieval_metrics": final_retrieval_metrics,
        "stopped_early": stopped_early,
        "stopped_step": stopped_step,
        "run_dir": str(run_dir),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if config.retrieval is not None and config.retrieval.enabled:
        retrieval_metrics = run_retrieval_pipeline(
            config=config,
            checkpoint_path=str(run_dir / "best_model.pt"),
            run_dir=run_dir,
            device=device,
        )
        tqdm.write(json.dumps(retrieval_metrics, indent=2))


if __name__ == "__main__":
    main()
