from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from fastquerydr.config import AppConfig, load_config
from fastquerydr.data import TriplesCollator, build_train_val_datasets
from fastquerydr.models import SymmetricBiEncoder
from fastquerydr.retrieval import run_retrieval_pipeline
from fastquerydr.utils.repro import prepare_run_dir, seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the symmetric teacher bi-encoder")
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


def compute_loss(model: SymmetricBiEncoder, batch: dict[str, dict[str, torch.Tensor]], criterion: nn.Module) -> torch.Tensor:
    query_embeddings = model.encode(batch["query_inputs"])
    positive_embeddings = model.encode(batch["positive_inputs"])
    logits = model.similarity(query_embeddings, positive_embeddings)
    labels = torch.arange(logits.size(0), device=logits.device)
    return criterion(logits, labels)


@torch.no_grad()
def evaluate(model: SymmetricBiEncoder, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(config.experiment.seed)
    run_dir = prepare_run_dir(config.experiment.output_dir, config.experiment.name)
    save_metadata(run_dir, config)

    device = select_device(config.training.device)
    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder_name)
    train_loader, val_loader = build_dataloaders(config, tokenizer)

    model = SymmetricBiEncoder(
        encoder_name=config.model.encoder_name,
        pooling=config.model.pooling,
        normalize=config.model.normalize,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / config.training.grad_accumulation_steps)
    total_updates = updates_per_epoch * config.training.num_epochs
    warmup_steps = int(total_updates * config.training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_updates, 1),
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=config.training.mixed_precision and device.type == "cuda")

    best_val_loss = float("inf")
    global_step = 0
    progress = tqdm(total=len(train_loader) * config.training.num_epochs, desc="training", leave=False)

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
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), run_dir / "best_model.pt")
                tqdm.write(f"step={global_step} train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

            progress.update(1)

        if len(train_loader) % config.training.grad_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    final_val_loss = evaluate(model, val_loader, criterion, device)
    torch.save(model.state_dict(), run_dir / "last_model.pt")
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        torch.save(model.state_dict(), run_dir / "best_model.pt")

    metrics = {
        "final_val_loss": final_val_loss,
        "best_val_loss": min(best_val_loss, final_val_loss),
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
