"""Logo-mask ablation: does the model rely on the badge/grille (shortcut) or
on overall design/condition cues?

Experimental design
--------------------
For each front-view photo we run the same checkpoint under three conditions
that differ only in what part of the *input image* is occluded before
`eval_transforms` (the model itself, weights, and preprocessing are
untouched):

  - `none`    unmodified image (baseline).
  - `badge`   a fixed relative-coordinate box roughly covering the front
              badge/grille is filled with the ImageNet mean color.
  - `control` a same-area box in the top-left corner (mostly background/sky,
              not the car) is filled the same way -- this is the causal
              control: it removes the *same amount* of pixel information but
              from a region that should carry no brand signal.

If occluding the badge shifts predictions and degrades accuracy much more
than occluding the control region, that is evidence the model leans on the
badge/grille as a brand-lookup shortcut rather than on genuine visual design
cues spread across the car body. Because `badge` and `control` remove
comparable amounts of information, the (badge - none) vs (control - none)
deltas isolate the *location* of the occlusion as the explanatory variable.

Masking is applied to the front-view (`is_front=True`) subset only, since the
badge-box heuristic assumes a roughly centered front-on shot; on side/rear
photos the box would land on an arbitrary, meaningless region.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from car_price_vision.data.dataset import DVMCarDataset
from car_price_vision.data.transforms import eval_transforms
from car_price_vision.eval import build_model, load_manifest_and_splits, resolve_checkpoint_meta
from car_price_vision.metrics import mae_years, mape, r2
from car_price_vision.utils import Config, get_device, load_config, seed_everything

logger = logging.getLogger("ablation_logo_mask")

# Fill color for occluded regions: the ImageNet mean (0.485, 0.456, 0.406),
# scaled to 0-255 -- a neutral color the backbone's normalization was fit
# against, rather than e.g. black, which would be a much stronger deviation
# from typical eval_transforms inputs and a confound in its own right.
MASK_FILL_COLOR = (124, 116, 104)

CONDITIONS = ("none", "badge", "control")


class MaskedDVMCarDataset(DVMCarDataset):
    """`DVMCarDataset` variant that occludes a fixed relative-coordinate box
    on every loaded PIL image before the eval transform runs.

    Reuses `DVMCarDataset`'s manifest handling, path resolution, and blank-
    image fallback for corrupt files by only overriding `_load_image`: it
    calls the parent implementation to get the (already RGB, already
    fallback-handled) PIL image, then paints the box on top with
    `MASK_FILL_COLOR`. `box=None` reproduces the parent's behavior exactly
    (the `none`/baseline condition), so all three conditions share one class.
    """

    def __init__(self, *args, box: tuple[float, float, float, float] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.box = box

    def _load_image(self, image_path: str) -> Image.Image:
        image = super()._load_image(image_path)
        if self.box is None:
            return image
        return _apply_mask(image, self.box)


def _apply_mask(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """Fill the relative-coordinate rectangle `box` (x0, y0, x1, y1, each in
    [0, 1]) on a copy of `image` with `MASK_FILL_COLOR`."""
    w, h = image.size
    x0, y0, x1, y1 = box
    px0, py0, px1, py1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    masked = image.copy()
    ImageDraw.Draw(masked).rectangle([px0, py0, px1, py1], fill=MASK_FILL_COLOR)
    return masked


def filter_front_views(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Restrict `indices` (row positions into `df`) to front-view photos.

    The manifest's `is_front` flag (confirmed_fronts membership) is False for
    every row in the current manifest (the build_manifest.py join never
    matched), so it is only trusted when it actually marks something; the
    fallback is the DVM-CAR `viewpoint` angle, where 0 degrees = frontal.
    """
    if "is_front" in df.columns and bool(df["is_front"].sum()):
        is_front = df["is_front"].astype(bool).to_numpy()
        return indices[is_front[indices]]
    if "viewpoint" not in df.columns:
        raise ValueError(
            "Manifest has neither a usable 'is_front' column nor a 'viewpoint' column -- "
            "cannot restrict to front-view photos. Regenerate with scripts/build_manifest.py."
        )
    is_front = (df["viewpoint"].astype(str).str.strip() == "0").to_numpy()
    return indices[is_front[indices]]


def subsample(indices: np.ndarray, max_images: int, seed: int) -> np.ndarray:
    """Deterministic subsample of `indices` to at most `max_images` rows."""
    if max_images is None or len(indices) <= max_images:
        return indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(indices), size=max_images, replace=False)
    chosen.sort()
    return indices[chosen]


@torch.no_grad()
def predict(model, loader: DataLoader, device: torch.device, target_norm: dict) -> dict[str, np.ndarray]:
    """Run the model over a DataLoader, returning de-standardized real-unit
    predictions and ground truth (mirrors `eval.py:predict`)."""
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


def condition_metrics(out: dict[str, np.ndarray]) -> dict[str, float]:
    """The requested per-condition metric suite, reusing metrics.py."""
    return {
        "n_samples": int(len(out["pred_year"])),
        "mae_years": mae_years(out["pred_year"], out["true_year"]),
        "mape": mape(out["pred_log_price"], out["true_log_price"]),
        "r2_price_log": r2(out["pred_log_price"], out["true_log_price"]),
    }


def shift_stats(out_masked: dict[str, np.ndarray], out_none: dict[str, np.ndarray]) -> dict[str, float]:
    """How much predictions shift under a masked condition vs. the `none`
    baseline, on a *per-image* basis (independent of the ground truth)."""
    delta_year = np.abs(out_masked["pred_year"] - out_none["pred_year"])
    delta_log_price = np.abs(out_masked["pred_log_price"] - out_none["pred_log_price"])

    price_none = np.exp(out_none["pred_log_price"])
    price_masked = np.exp(out_masked["pred_log_price"])
    rel_price_shift = np.abs(price_masked - price_none) / np.maximum(np.abs(price_none), 1e-6)

    return {
        "mean_abs_delta_year": float(np.mean(delta_year)),
        "median_abs_delta_year": float(np.median(delta_year)),
        "mean_abs_delta_log_price": float(np.mean(delta_log_price)),
        "median_abs_delta_log_price": float(np.median(delta_log_price)),
        "frac_price_shift_gt10pct": float(np.mean(rel_price_shift > 0.10)),
        "frac_price_shift_gt25pct": float(np.mean(rel_price_shift > 0.25)),
    }


def ablation_deltas(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """(badge - none) vs (control - none) for each accuracy metric, plus
    their difference (badge_vs_control) -- the key ablation comparison: if
    badge-masking hurts accuracy much more than control-masking, the model
    was leaning on the badge/grille region."""
    keys = ("mae_years", "mape", "r2_price_log")
    badge_minus_none = {k: metrics["badge"][k] - metrics["none"][k] for k in keys}
    control_minus_none = {k: metrics["control"][k] - metrics["none"][k] for k in keys}
    badge_vs_control = {k: badge_minus_none[k] - control_minus_none[k] for k in keys}
    return {
        "badge_minus_none": badge_minus_none,
        "control_minus_none": control_minus_none,
        "badge_vs_control": badge_vs_control,
    }


def save_examples(
    df: pd.DataFrame,
    indices: np.ndarray,
    data_root: str | None,
    badge_box: tuple[float, float, float, float],
    control_box: tuple[float, float, float, float],
    n: int,
    out_dir: Path,
) -> list[str]:
    """Write up to `n` side-by-side (original | badge-masked | control-masked)
    PNGs so a human can verify the boxes actually cover grille/badge vs.
    background. Returns the list of written file names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plain_ds = DVMCarDataset(df, data_root=data_root, transform=None, indices=indices, target_norm=None)

    written = []
    n = min(n, len(plain_ds))
    for i in range(n):
        image = plain_ds._load_image(plain_ds.df.iloc[i]["image_path"])
        badge_img = _apply_mask(image, badge_box)
        control_img = _apply_mask(image, control_box)

        w, h = image.size
        combo = Image.new("RGB", (w * 3 + 20, h), color=(255, 255, 255))
        combo.paste(image, (0, 0))
        combo.paste(badge_img, (w + 10, 0))
        combo.paste(control_img, (2 * (w + 10), 0))

        fname = f"example_{i:02d}.png"
        combo.save(out_dir / fname)
        written.append(fname)

    return written


def run_split(
    split_name: str,
    df: pd.DataFrame,
    split_indices: np.ndarray,
    model,
    device: torch.device,
    target_norm: dict,
    img_size: int,
    badge_box: tuple[float, float, float, float],
    control_box: tuple[float, float, float, float],
    max_images: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    data_root: str | None,
) -> dict:
    front_indices = filter_front_views(df, split_indices)
    n_front = len(front_indices)
    used_indices = subsample(front_indices, max_images, seed)
    logger.info(
        "[%s] %d rows in split, %d front-view, %d used after subsample (max_images=%s)",
        split_name,
        len(split_indices),
        n_front,
        len(used_indices),
        max_images,
    )

    transform = eval_transforms(img_size)
    boxes = {"none": None, "badge": badge_box, "control": control_box}

    outputs: dict[str, dict[str, np.ndarray]] = {}
    for condition, box in boxes.items():
        dataset = MaskedDVMCarDataset(
            df, data_root=data_root, transform=transform, indices=used_indices, target_norm=None, box=box
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        outputs[condition] = predict(model, loader, device, target_norm)
        logger.info("[%s] condition=%-7s n=%d", split_name, condition, len(outputs[condition]["pred_year"]))

    metrics = {cond: condition_metrics(out) for cond, out in outputs.items()}
    shifts = {
        cond: shift_stats(outputs[cond], outputs["none"]) for cond in ("badge", "control")
    }
    deltas = ablation_deltas(metrics)

    return {
        "split": split_name,
        "n_front_view": int(n_front),
        "n_used": int(len(used_indices)),
        "conditions": metrics,
        "shift_vs_none": shifts,
        "ablation_deltas": deltas,
    }


def write_summary_md(all_results: dict[str, dict], out_path: Path) -> None:
    lines = ["# Logo-mask ablation summary", ""]
    for split_name, result in all_results.items():
        lines.append(f"## {split_name} (n_used={result['n_used']}, n_front_view={result['n_front_view']})")
        lines.append("")
        lines.append("| condition | n | mae_years | mape (%) | r2_price_log |")
        lines.append("|---|---|---|---|---|")
        for cond in CONDITIONS:
            m = result["conditions"][cond]
            lines.append(
                f"| {cond} | {m['n_samples']} | {m['mae_years']:.3f} | {m['mape']:.2f} | {m['r2_price_log']:.4f} |"
            )
        lines.append("")
        lines.append("| shift vs none | mean \\|Δyear\\| | median \\|Δyear\\| | mean \\|Δlog_price\\| "
                     "| median \\|Δlog_price\\| | frac >10% | frac >25% |")
        lines.append("|---|---|---|---|---|---|---|")
        for cond in ("badge", "control"):
            s = result["shift_vs_none"][cond]
            lines.append(
                f"| {cond} | {s['mean_abs_delta_year']:.3f} | {s['median_abs_delta_year']:.3f} | "
                f"{s['mean_abs_delta_log_price']:.4f} | {s['median_abs_delta_log_price']:.4f} | "
                f"{s['frac_price_shift_gt10pct']:.3f} | {s['frac_price_shift_gt25pct']:.3f} |"
            )
        lines.append("")
        lines.append("| ablation delta | mae_years | mape (%) | r2_price_log |")
        lines.append("|---|---|---|---|")
        for key in ("badge_minus_none", "control_minus_none", "badge_vs_control"):
            d = result["ablation_deltas"][key]
            lines.append(f"| {key} | {d['mae_years']:+.3f} | {d['mape']:+.2f} | {d['r2_price_log']:+.4f} |")
        lines.append("")

    out_path.write_text("\n".join(lines))


def print_summary(all_results: dict[str, dict]) -> None:
    print("\n=== Logo-mask ablation summary ===")
    for split_name, result in all_results.items():
        print(f"\n[{split_name}] n_used={result['n_used']} (n_front_view={result['n_front_view']})")
        print(f"  {'condition':10s} {'n':>6s} {'mae_years':>10s} {'mape%':>8s} {'r2_price_log':>13s}")
        for cond in CONDITIONS:
            m = result["conditions"][cond]
            print(
                f"  {cond:10s} {m['n_samples']:>6d} {m['mae_years']:>10.3f} {m['mape']:>8.2f} {m['r2_price_log']:>13.4f}"
            )
        print("  --- shift vs none (badge vs control) ---")
        for cond in ("badge", "control"):
            s = result["shift_vs_none"][cond]
            print(
                f"  {cond:10s} mean|Δyear|={s['mean_abs_delta_year']:.3f} "
                f"median|Δyear|={s['median_abs_delta_year']:.3f} "
                f"mean|Δlogp|={s['mean_abs_delta_log_price']:.4f} "
                f"frac>10%={s['frac_price_shift_gt10pct']:.3f} frac>25%={s['frac_price_shift_gt25pct']:.3f}"
            )
        print("  --- ablation deltas (badge_minus_none vs control_minus_none) ---")
        d = result["ablation_deltas"]
        for key in ("badge_minus_none", "control_minus_none", "badge_vs_control"):
            v = d[key]
            print(f"  {key:20s} mae_years={v['mae_years']:+.3f} mape={v['mape']:+.2f} r2_price_log={v['r2_price_log']:+.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Logo-mask ablation: occlude the badge/grille (vs. a background control box) and "
        "measure how much MultiTaskCarNet predictions shift, to quantify brand-badge shortcut reliance."
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint from train.py.")
    parser.add_argument("--splits", type=str, nargs="+", default=["test"], help="Which splits to evaluate.")
    parser.add_argument(
        "--max-images", type=int, default=5000, help="Max front-view images per split, after filtering (seeded subsample)."
    )
    parser.add_argument("--out-dir", type=str, default="results/ablation_logo_mask")
    parser.add_argument(
        "--badge-box",
        type=float,
        nargs=4,
        default=[0.35, 0.45, 0.65, 0.72],
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Relative (x0 y0 x1 y1) box over the badge/grille region.",
    )
    parser.add_argument(
        "--control-box",
        type=float,
        nargs=4,
        default=[0.02, 0.02, 0.32, 0.29],
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Relative (x0 y0 x1 y1) box over a same-area background control region (top-left corner).",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to cfg.eval.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Defaults to cfg.data.num_workers.")
    parser.add_argument("--device", type=str, default=None, help="Defaults to auto (cuda > mps > cpu).")
    parser.add_argument("--save-examples", type=int, default=8, help="Number of side-by-side example PNGs per split.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    device = get_device(args.device)

    badge_box = tuple(args.badge_box)
    control_box = tuple(args.control_box)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    target_norm, backbone_name, img_size = resolve_checkpoint_meta(checkpoint, cfg)
    model = build_model(checkpoint, backbone_name, cfg, device)

    df, splits = load_manifest_and_splits(cfg)
    data_root = cfg.paths.get("data_root")
    batch_size = args.batch_size or cfg.eval.get("batch_size", 128)
    num_workers = args.num_workers if args.num_workers is not None else cfg.data.get("num_workers", 8)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_params = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "backbone_name": backbone_name,
        "img_size": img_size,
        "max_images": args.max_images,
        "badge_box": list(badge_box),
        "control_box": list(control_box),
        "mask_fill_color": list(MASK_FILL_COLOR),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(device),
        "seed": cfg.seed,
    }

    all_results: dict[str, dict] = {}
    for split_name in args.splits:
        if split_name not in splits:
            raise ValueError(f"Unknown split: {split_name!r}. Available: {list(splits.keys())}")

        result = run_split(
            split_name=split_name,
            df=df,
            split_indices=splits[split_name],
            model=model,
            device=device,
            target_norm=target_norm,
            img_size=img_size,
            badge_box=badge_box,
            control_box=control_box,
            max_images=args.max_images,
            seed=cfg.seed,
            batch_size=batch_size,
            num_workers=num_workers,
            data_root=data_root,
        )
        result["run_params"] = run_params
        all_results[split_name] = result

        split_json_path = out_dir / f"ablation_{split_name}.json"
        with split_json_path.open("w") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote %s", split_json_path)

        if args.save_examples > 0:
            front_indices = filter_front_views(df, splits[split_name])
            example_indices = subsample(front_indices, args.save_examples, cfg.seed)
            examples_dir = out_dir / "examples" / split_name
            written = save_examples(
                df, example_indices, data_root, badge_box, control_box, args.save_examples, examples_dir
            )
            logger.info("Wrote %d example image(s) to %s", len(written), examples_dir)

    write_summary_md(all_results, out_dir / "summary.md")
    logger.info("Wrote %s", out_dir / "summary.md")

    print_summary(all_results)


if __name__ == "__main__":
    main()
