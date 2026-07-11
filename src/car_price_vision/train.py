"""Two-stage training loop for MultiTaskCarNet.

Stage 1: backbone frozen, only the two regression heads are trained (fast,
stabilizes the heads before touching pretrained weights).
Stage 2: the last `unfreeze_last_n_blocks` backbone blocks are unfrozen and
fine-tuned end-to-end at a lower learning rate, together with the heads.

Designed to run detached (e.g. `nohup python -m car_price_vision.train
--config configs/default.yaml &`, or inside `tmux`/`screen` on `rtx`):
  - all progress is written to a CSV log file and a rotating text log under
    `paths.log_dir`, not just stdout;
  - checkpoints are written every `train.checkpoint_every_epochs` epochs to
    `paths.checkpoint_dir`, and training can resume from the latest one.

Usage:
    python -m car_price_vision.train --config configs/default.yaml
    python -m car_price_vision.train --config configs/default.yaml --resume
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from car_price_vision.data.dataset import DVMCarDataset
from car_price_vision.data.splits import make_splits
from car_price_vision.data.transforms import eval_transforms, train_transforms
from car_price_vision.losses import multitask_loss
from car_price_vision.models.multitask import MultiTaskCarNet
from car_price_vision.utils import AverageMeter, Config, get_device, load_config, seed_everything, setup_logging

import pandas as pd


class CsvLogger:
    """Appends one row per (epoch, split) to a CSV file, creating the header
    on first write. Kept separate from the text logger so downstream
    plotting (see notebooks/03_finetune.ipynb) can just `pd.read_csv` it.
    """

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict) -> None:
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


def build_dataloaders(cfg: Config) -> dict[str, DataLoader]:
    """Build train/val/test/holdout DataLoaders from the manifest + config."""
    manifest_path = cfg.paths["manifest_csv"]
    df = pd.read_csv(manifest_path)

    # TODO(phase 2): once subset_size approaches the full DVM-CAR size,
    # consider removing this cap or sampling stratified by brand/year
    # instead of a plain head() truncation.
    subset_size = cfg.data.get("subset_size")
    if subset_size is not None and subset_size < len(df):
        df = df.sample(n=subset_size, random_state=cfg.seed).reset_index(drop=True)

    splits = make_splits(
        df,
        mode=cfg.data.get("split_mode", "by_advert"),
        holdout_models_frac=cfg.data.get("holdout_models_frac", 0.1),
        val_frac=cfg.data.get("val_frac", 0.1),
        test_frac=cfg.data.get("test_frac", 0.1),
        seed=cfg.seed,
    )

    img_size = cfg.data.get("img_size", 224)
    data_root = cfg.paths.get("data_root")
    # Top-level `target_norm` block (see configs/default.yaml); Config has no
    # dedicated property for it, so go through the generic dict accessor.
    target_norm = cfg.get("target_norm", None)

    train_ds = DVMCarDataset(
        df, data_root=data_root, transform=train_transforms(img_size), indices=splits["train"], target_norm=target_norm
    )
    val_ds = DVMCarDataset(
        df, data_root=data_root, transform=eval_transforms(img_size), indices=splits["val"], target_norm=target_norm
    )
    test_ds = DVMCarDataset(
        df, data_root=data_root, transform=eval_transforms(img_size), indices=splits["test"], target_norm=target_norm
    )
    holdout_ds = DVMCarDataset(
        df, data_root=data_root, transform=eval_transforms(img_size), indices=splits["holdout"], target_norm=target_norm
    )

    num_workers = cfg.data.get("num_workers", 8)
    batch_size = cfg.train.get("batch_size", 64)
    eval_batch_size = cfg.eval.get("batch_size", 128)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
        "val": DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers),
        "holdout": DataLoader(holdout_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers),
    }


def run_epoch(
    model: MultiTaskCarNet,
    loader: DataLoader,
    device: torch.device,
    cfg: Config,
    optimizer: torch.optim.Optimizer | None = None,
    logger=None,
    log_every_steps: int = 50,
    target_norm: dict | None = None,
) -> dict[str, float]:
    """Run one epoch over `loader`. Trains if `optimizer` is given, otherwise
    evaluates in no-grad mode. Returns averaged loss/year-MAE/price-log-MAE.

    Loss is always computed in whatever space the dataset returns targets in
    (z-space when `target_norm` was passed to the dataset, see
    data/dataset.py). `year_mae`/`price_log_mae` are de-standardized back to
    real units (years / natural-log-GBP) for human-readable logging using
    `target_norm`'s stds, so they read the same as before targets were
    standardized. If `target_norm` is None, no de-standardization is applied
    (stds default to 1.0, i.e. raw-space MAE, matching the old behavior).

    AMP: when `cfg.train.amp` is true and `device` is CUDA, the forward pass
    and loss computation run under `torch.autocast(..., dtype=torch.bfloat16)`.
    bf16 needs no GradScaler (unlike fp16), so backward/optimizer step run
    at full precision outside the autocast context. Falls back to full
    precision on CPU/MPS or when `train.amp` is false.
    """
    is_train = optimizer is not None
    model.train(is_train)

    loss_meter = AverageMeter("loss")
    year_mae_meter = AverageMeter("year_mae")
    price_mae_meter = AverageMeter("price_log_mae")

    weight_year = cfg.loss.get("weight_year", 1.0)
    weight_price = cfg.loss.get("weight_price", 1.0)
    huber_delta = cfg.loss.get("huber_delta", 1.0)
    grad_clip_norm = cfg.train.get("grad_clip_norm", 1.0)

    year_std = target_norm["year_std"] if target_norm is not None else 1.0
    log_price_std = target_norm["log_price_std"] if target_norm is not None else 1.0

    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"
    # Only ever construct a CUDA autocast context when we've confirmed the
    # device is CUDA; otherwise stay a no-op so this is safe on CPU/MPS too.
    amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for step, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with amp_context:
                preds = model(images)
                losses = multitask_loss(
                    preds, targets, weight_year=weight_year, weight_price=weight_price, huber_delta=huber_delta
                )

            if is_train:
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip_norm)
                optimizer.step()

            batch_size = images.size(0)
            loss_meter.update(losses["loss"].item(), batch_size)
            year_mae_meter.update(
                year_std * torch.mean(torch.abs(preds["year"].detach() - targets["year"])).item(), batch_size
            )
            price_mae_meter.update(
                log_price_std * torch.mean(torch.abs(preds["log_price"].detach() - targets["log_price"])).item(),
                batch_size,
            )

            if logger is not None and is_train and step % log_every_steps == 0:
                logger.info(
                    "  step %d/%d | loss=%.4f year_mae=%.3f price_log_mae=%.4f",
                    step,
                    len(loader),
                    loss_meter.avg,
                    year_mae_meter.avg,
                    price_mae_meter.avg,
                )

    return {"loss": loss_meter.avg, "year_mae": year_mae_meter.avg, "price_log_mae": price_mae_meter.avg}


def save_checkpoint(
    path: Path,
    model: MultiTaskCarNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stage: int,
    target_norm: dict | None = None,
    backbone_name: str | None = None,
    img_size: int | None = None,
) -> None:
    """Persist model/optimizer state plus enough metadata (target
    standardization constants, backbone architecture, input resolution) for
    eval.py / serving to reconstruct the model and invert predictions back
    to real units without needing the training config on hand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "target_norm": target_norm,
            "backbone_name": backbone_name,
            "img_size": img_size,
        },
        path,
    )


def load_checkpoint(path: Path, model: MultiTaskCarNet, optimizer: torch.optim.Optimizer | None = None) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train(cfg: Config, resume: bool = False) -> None:
    seed_everything(cfg.seed)
    device = get_device()

    log_dir = Path(cfg.paths.get("log_dir", "outputs/default_run/logs"))
    checkpoint_dir = Path(cfg.paths.get("checkpoint_dir", "checkpoints/default_run"))
    logger = setup_logging(log_dir)
    csv_logger = CsvLogger(
        log_dir / "metrics.csv",
        fieldnames=["stage", "epoch", "split", "loss", "year_mae", "price_log_mae", "elapsed_s"],
    )

    logger.info("Device: %s", device)
    backbone_name = cfg.model.get("backbone", "convnext_tiny")
    img_size = cfg.data.get("img_size", 224)
    target_norm = cfg.get("target_norm", None)
    logger.info("Backbone: %s", backbone_name)
    logger.info("Target norm: %s", target_norm)

    loaders = build_dataloaders(cfg)

    model = MultiTaskCarNet(
        backbone_name=backbone_name,
        pretrained=cfg.model.get("pretrained", True),
        head_hidden_dim=cfg.model.get("head_hidden_dim", 256),
        head_dropout=cfg.model.get("head_dropout", 0.2),
    ).to(device)

    start_stage = 1
    start_epoch = 0
    latest_ckpt = checkpoint_dir / "latest.pt"
    if resume and latest_ckpt.exists():
        logger.info("Resuming from %s", latest_ckpt)
        checkpoint = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_stage = checkpoint["stage"]
        start_epoch = checkpoint["epoch"] + 1

    stage_configs = [
        (1, cfg.train.get("stage1", {"epochs": 5, "lr": 1e-3})),
        (2, cfg.train.get("stage2", {"epochs": 15, "lr": 1e-4})),
    ]
    weight_decay = cfg.train.get("weight_decay", 1e-4)
    log_every_steps = cfg.train.get("log_every_steps", 50)
    checkpoint_every = cfg.train.get("checkpoint_every_epochs", 1)

    for stage, stage_cfg in stage_configs:
        if stage < start_stage:
            continue

        if stage == 1:
            model.freeze_backbone()
            logger.info("Stage 1: backbone frozen, training heads only.")
        elif stage == 2:
            n_blocks = cfg.train.get("stage2", {}).get("unfreeze_last_n_blocks", 2)
            model.unfreeze_backbone_last_blocks(n_blocks)
            logger.info("Stage 2: unfroze last %d backbone block(s) for fine-tuning.", n_blocks)

        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=stage_cfg["lr"], weight_decay=weight_decay)

        epoch_start = start_epoch if stage == start_stage else 0
        for epoch in range(epoch_start, stage_cfg["epochs"]):
            t0 = time.time()
            train_metrics = run_epoch(
                model,
                loaders["train"],
                device,
                cfg,
                optimizer=optimizer,
                logger=logger,
                log_every_steps=log_every_steps,
                target_norm=target_norm,
            )
            val_metrics = run_epoch(model, loaders["val"], device, cfg, optimizer=None, target_norm=target_norm)
            elapsed = time.time() - t0

            logger.info(
                "[stage %d][epoch %d/%d] train_loss=%.4f val_loss=%.4f val_year_mae=%.3f val_price_log_mae=%.4f (%.1fs)",
                stage,
                epoch,
                stage_cfg["epochs"] - 1,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["year_mae"],
                val_metrics["price_log_mae"],
                elapsed,
            )
            csv_logger.log({"stage": stage, "epoch": epoch, "split": "train", "elapsed_s": elapsed, **train_metrics})
            csv_logger.log({"stage": stage, "epoch": epoch, "split": "val", "elapsed_s": elapsed, **val_metrics})

            if epoch % checkpoint_every == 0:
                save_checkpoint(
                    latest_ckpt,
                    model,
                    optimizer,
                    epoch,
                    stage,
                    target_norm=target_norm,
                    backbone_name=backbone_name,
                    img_size=img_size,
                )
                save_checkpoint(
                    checkpoint_dir / f"stage{stage}_epoch{epoch}.pt",
                    model,
                    optimizer,
                    epoch,
                    stage,
                    target_norm=target_norm,
                    backbone_name=backbone_name,
                    img_size=img_size,
                )

    logger.info("Training complete. Final checkpoint: %s", latest_ckpt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MultiTaskCarNet (year + price heads).")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to a YAML config file.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint_dir/latest.pt if present.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
