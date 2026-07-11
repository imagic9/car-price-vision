"""Evaluate a trained checkpoint on val/test/unseen-models-holdout splits.

Model outputs (and the targets `train.py` computed loss against) live in
z-space (see `data/dataset.py`'s `target_norm` standardization and
`train.py:save_checkpoint`). This script de-standardizes predictions back to
real units before computing any metric:

    real_year      = pred_year_z * year_std + year_mean
    real_log_price = pred_logprice_z * log_price_std + log_price_mean
    real_price_gbp = exp(real_log_price)

`target_norm` (plus the backbone name and img_size used to train the
checkpoint) is read from the checkpoint itself, so this script does not
depend on the config's `target_norm`/`model.backbone`/`data.img_size`
matching what the checkpoint was actually trained with -- the config is only
used as a fallback if the checkpoint predates these fields, and for
everything unrelated to model reconstruction (paths, split fractions, seed,
batch size).

Computes MAE-years, MAE-log, MAPE, R^2 (year and price), and within-brand
price correlation (see metrics.py for the rationale of each). Prints a
summary table and dumps the full results as JSON.

Usage:
    python -m car_price_vision.eval --config configs/default.yaml \\
        --checkpoint checkpoints/default_run/latest.pt \\
        --splits val test holdout \\
        --out-dir outputs/default_run
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from car_price_vision.data.dataset import DVMCarDataset
from car_price_vision.data.splits import make_splits
from car_price_vision.data.transforms import eval_transforms
from car_price_vision.metrics import mae_log, mae_years, mape, r2, within_brand_corr
from car_price_vision.models.multitask import MultiTaskCarNet
from car_price_vision.utils import Config, get_device, load_config, seed_everything

# Keys rounded to this many decimal places when printed/dumped. MAPE is a
# percentage so gets fewer decimals; correlations/R^2 are unitless in [-1,1]
# (ish) so a bit more precision is useful there.
ROUND_DIGITS = {
    "mae_years": 3,
    "mae_log": 4,
    "mape": 2,
    "r2_year": 4,
    "r2_price_log": 4,
    "within_brand_corr_mean": 4,
}


def resolve_checkpoint_meta(checkpoint: dict, cfg: Config) -> tuple[dict, str, int]:
    """Recover `target_norm`/backbone/img_size from the checkpoint, falling
    back to the config only if the checkpoint predates these fields.
    """
    target_norm = checkpoint.get("target_norm") or cfg.get("target_norm")
    if target_norm is None:
        raise ValueError(
            "No target_norm found in the checkpoint or the config. Predictions are in "
            "z-space and cannot be de-standardized without it -- see configs/default.yaml."
        )
    backbone_name = checkpoint.get("backbone_name") or cfg.model.get("backbone", "convnext_tiny")
    img_size = checkpoint.get("img_size") or cfg.data.get("img_size", 224)
    return target_norm, backbone_name, img_size


def build_model(checkpoint: dict, backbone_name: str, cfg: Config, device: torch.device) -> MultiTaskCarNet:
    model = MultiTaskCarNet(
        backbone_name=backbone_name,
        pretrained=False,  # weights come from the checkpoint, not ImageNet, at eval time
        head_hidden_dim=cfg.model.get("head_hidden_dim", 256),
        head_dropout=cfg.model.get("head_dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_manifest_and_splits(cfg: Config) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Rebuild the manifest + splits exactly as `train.py:build_dataloaders`
    does (same subset_size sampling before splitting, same split params and
    seed), so val/test/holdout here line up with what the checkpoint was
    trained/validated on.
    """
    df = pd.read_csv(cfg.paths["manifest_csv"])

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
    return df, splits


@torch.no_grad()
def predict(
    model: MultiTaskCarNet, loader: DataLoader, device: torch.device, target_norm: dict
) -> dict[str, np.ndarray]:
    """Run the model over a DataLoader and collect de-standardized
    predictions (real years / real natural-log GBP) alongside the raw
    ground-truth targets already returned by the dataset (the dataset here
    is built with `target_norm=None`, so `targets` are already real units).
    """
    model.eval()
    year_mean, year_std = target_norm["year_mean"], target_norm["year_std"]
    log_price_mean, log_price_std = target_norm["log_price_mean"], target_norm["log_price_std"]

    use_amp = device.type == "cuda"
    amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

    pred_year, true_year = [], []
    pred_log_price, true_log_price = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with amp_context:
            preds = model(images)

        pred_year_z = preds["year"].float().cpu().numpy()
        pred_log_price_z = preds["log_price"].float().cpu().numpy()

        pred_year.append(pred_year_z * year_std + year_mean)
        pred_log_price.append(pred_log_price_z * log_price_std + log_price_mean)
        true_year.append(targets["year"].numpy())
        true_log_price.append(targets["log_price"].numpy())

    return {
        "pred_year": np.concatenate(pred_year),
        "true_year": np.concatenate(true_year),
        "pred_log_price": np.concatenate(pred_log_price),
        "true_log_price": np.concatenate(true_log_price),
    }


def evaluate_split(
    model: MultiTaskCarNet,
    dataset: DVMCarDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    target_norm: dict,
) -> dict:
    """Compute the full metric suite (in real units) for a single split."""
    if len(dataset) == 0:
        return {"n_samples": 0}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = predict(model, loader, device, target_norm)

    metrics = {
        "n_samples": len(dataset),
        "mae_years": mae_years(out["pred_year"], out["true_year"]),
        "mae_log": mae_log(out["pred_log_price"], out["true_log_price"]),
        "mape": mape(out["pred_log_price"], out["true_log_price"]),
        "r2_year": r2(out["pred_year"], out["true_year"]),
        "r2_price_log": r2(out["pred_log_price"], out["true_log_price"]),
    }

    # Brand comes straight from the manifest rows behind this split (dataset
    # was constructed with indices=splits[split_name], so row order matches
    # out["pred_*"]/out["true_*"] order).
    brand_df = pd.DataFrame(
        {
            "brand": [dataset.row_metadata(i)["brand"] for i in range(len(dataset))],
            "pred_price": np.exp(out["pred_log_price"]),
            "price_gbp": np.exp(out["true_log_price"]),
        }
    )
    brand_corr = within_brand_corr(brand_df)
    metrics["within_brand_corr_mean"] = brand_corr["mean"]
    metrics["within_brand_corr_n_brands"] = brand_corr["n_brands_used"]
    metrics["within_brand_corr_per_brand"] = brand_corr["per_brand"]

    for key, digits in ROUND_DIGITS.items():
        if key in metrics and metrics[key] is not None and not np.isnan(metrics[key]):
            metrics[key] = round(metrics[key], digits)

    return metrics


def run_eval(cfg: Config, checkpoint_path: str, splits_to_run: list[str]) -> dict:
    seed_everything(cfg.seed)
    device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    target_norm, backbone_name, img_size = resolve_checkpoint_meta(checkpoint, cfg)

    model = build_model(checkpoint, backbone_name, cfg, device)

    df, splits = load_manifest_and_splits(cfg)
    data_root = cfg.paths.get("data_root")
    transform = eval_transforms(img_size)

    batch_size = cfg.eval.get("batch_size", 128)
    num_workers = cfg.data.get("num_workers", 8)

    results: dict[str, dict] = {}
    for split_name in splits_to_run:
        if split_name not in splits:
            raise ValueError(f"Unknown split: {split_name!r}. Available: {list(splits.keys())}")
        # target_norm=None -> dataset returns raw year / raw log_price, i.e.
        # real-unit ground truth to compare de-standardized predictions against.
        dataset = DVMCarDataset(df, data_root=data_root, transform=transform, indices=splits[split_name], target_norm=None)
        results[split_name] = evaluate_split(model, dataset, device, batch_size, num_workers, target_norm)

    return results


def print_summary(results: dict) -> None:
    print("\n=== Evaluation summary (real units: years / GBP) ===")
    for split_name, metrics in results.items():
        print(f"\n[{split_name}] n={metrics.get('n_samples', 0)}")
        for key in ("mae_years", "mae_log", "mape", "r2_year", "r2_price_log", "within_brand_corr_mean"):
            if key in metrics and metrics[key] is not None:
                value = metrics[key]
                suffix = "%" if key == "mape" else ""
                print(f"  {key:28s} {value:.4f}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a MultiTaskCarNet checkpoint (metrics in real units).")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint from train.py.")
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["val", "test", "holdout"], help="Which splits to evaluate."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to write eval_metrics.json into. Defaults to the config's paths.out_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results = run_eval(cfg, args.checkpoint, args.splits)
    print_summary(results)

    out_dir = args.out_dir or cfg.paths.get("out_dir", "outputs/default_run")
    out_path = Path(out_dir) / "eval_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
