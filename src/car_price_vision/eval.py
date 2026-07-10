"""Evaluate a trained checkpoint on val/test/unseen-models-holdout splits.

Computes MAE-years, MAE-log, MAPE, R^2 (year and price), and within-brand
price correlation (see metrics.py for the rationale of each). Prints a
summary table and dumps the full results as JSON.

Usage:
    python -m car_price_vision.eval --config configs/default.yaml \\
        --checkpoint checkpoints/default_run/latest.pt \\
        --splits val test holdout \\
        --out outputs/default_run/eval_results.json
"""

from __future__ import annotations

import argparse
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


@torch.no_grad()
def predict(model: MultiTaskCarNet, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    """Run the model over a DataLoader and collect predictions/targets."""
    model.eval()
    pred_year, true_year = [], []
    pred_log_price, true_log_price = [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        preds = model(images)
        pred_year.append(preds["year"].cpu().numpy())
        true_year.append(targets["year"].numpy())
        pred_log_price.append(preds["log_price"].cpu().numpy())
        true_log_price.append(targets["log_price"].numpy())

    return {
        "pred_year": np.concatenate(pred_year),
        "true_year": np.concatenate(true_year),
        "pred_log_price": np.concatenate(pred_log_price),
        "true_log_price": np.concatenate(true_log_price),
    }


def evaluate_split(
    model: MultiTaskCarNet, dataset: DVMCarDataset, device: torch.device, batch_size: int, num_workers: int
) -> dict:
    """Compute the full metric suite for a single dataset/split."""
    if len(dataset) == 0:
        return {"n_samples": 0}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = predict(model, loader, device)

    metrics = {
        "n_samples": len(dataset),
        "mae_years": mae_years(out["pred_year"], out["true_year"]),
        "mae_log": mae_log(out["pred_log_price"], out["true_log_price"]),
        "mape": mape(out["pred_log_price"], out["true_log_price"]),
        "r2_year": r2(out["pred_year"], out["true_year"]),
        "r2_price_log": r2(out["pred_log_price"], out["true_log_price"]),
    }

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

    return metrics


def run_eval(cfg: Config, checkpoint_path: str, splits_to_run: list[str]) -> dict:
    seed_everything(cfg.seed)
    device = get_device()

    df = pd.read_csv(cfg.paths["manifest_csv"])
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
    transform = eval_transforms(img_size)

    model = MultiTaskCarNet(
        backbone_name=cfg.model.get("backbone", "convnext_tiny"),
        pretrained=False,  # weights come from the checkpoint, not ImageNet, at eval time
        head_hidden_dim=cfg.model.get("head_hidden_dim", 256),
        head_dropout=cfg.model.get("head_dropout", 0.2),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    batch_size = cfg.eval.get("batch_size", 128)
    num_workers = cfg.data.get("num_workers", 8)

    results: dict[str, dict] = {}
    for split_name in splits_to_run:
        if split_name not in splits:
            raise ValueError(f"Unknown split: {split_name!r}. Available: {list(splits.keys())}")
        dataset = DVMCarDataset(df, data_root=data_root, transform=transform, indices=splits[split_name])
        results[split_name] = evaluate_split(model, dataset, device, batch_size, num_workers)

    return results


def print_summary(results: dict) -> None:
    print("\n=== Evaluation summary ===")
    for split_name, metrics in results.items():
        print(f"\n[{split_name}] n={metrics.get('n_samples', 0)}")
        for key in ("mae_years", "mae_log", "mape", "r2_year", "r2_price_log", "within_brand_corr_mean"):
            if key in metrics:
                print(f"  {key:28s} {metrics[key]:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a MultiTaskCarNet checkpoint.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint from train.py.")
    parser.add_argument(
        "--splits", type=str, nargs="+", default=["val", "test", "holdout"], help="Which splits to evaluate."
    )
    parser.add_argument("--out", type=str, default=None, help="Optional path to dump results as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results = run_eval(cfg, args.checkpoint, args.splits)
    print_summary(results)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
