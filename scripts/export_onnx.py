"""Export a trained MultiTaskCarNet checkpoint (see train.py:save_checkpoint)
to ONNX for CPU-friendly serving, alongside a plain torch checkpoint and a
`model_meta.json` sidecar.

IMPORTANT -- ONNX outputs are in z-space, NOT real years / GBP. The model
was trained against standardized targets (see data/dataset.py's
`target_norm` standardization), so both ONNX heads (`year_z`, `log_price_z`)
must be de-standardized by the caller (see serving/app.py) using the exact
`target_norm` constants written to `model_meta.json`:

    real_year      = year_z * target_norm["year_std"] + target_norm["year_mean"]
    real_log_price = log_price_z * target_norm["log_price_std"] + target_norm["log_price_mean"]
    real_price_gbp = exp(real_log_price)

Outputs written to --out-dir:
    model.onnx       ONNX graph, input "pixel_values" (N,3,H,W), dynamic
                      batch axis, outputs "year_z" and "log_price_z".
    model.pt          Same checkpoint dict train.py saved (model_state_dict +
                      target_norm + backbone_name + img_size + ...), so the
                      serving torch backend / Grad-CAM can load it unchanged.
    model_meta.json   Single source of truth for de-standardization + preprocessing:
                      target_norm, backbone, img_size, outputs, and the
                      ImageNet mean/std used by data/transforms.py:eval_transforms.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/default_run/latest.pt \\
        --out-dir outputs/default_run/export [--img-size 224]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from car_price_vision.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from car_price_vision.models.multitask import MultiTaskCarNet

OUTPUT_NAMES = ["year_z", "log_price_z"]


class _ONNXExportWrapper(nn.Module):
    """Wraps MultiTaskCarNet so the traced graph returns a fixed-order tuple
    of tensors (year_z, log_price_z) instead of a dict -- `torch.onnx.export`
    needs tensor/tuple outputs, not dicts, for reliable tracing.
    """

    def __init__(self, model: MultiTaskCarNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        preds = self.model(pixel_values)
        return preds["year"], preds["log_price"]


def load_checkpoint_and_model(checkpoint_path: Path) -> tuple[dict, MultiTaskCarNet, str, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    backbone_name = checkpoint.get("backbone_name")
    if backbone_name is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'backbone_name' field -- cannot reconstruct the "
            "model architecture. Re-save it with a train.py that writes backbone_name (see save_checkpoint)."
        )
    target_norm = checkpoint.get("target_norm")
    if target_norm is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'target_norm' field -- ONNX outputs would be "
            "impossible to de-standardize downstream. Re-save it with a train.py that writes target_norm."
        )

    # head_hidden_dim/head_dropout aren't persisted in the checkpoint (they
    # only affect the trained weights' shapes, which state_dict loading will
    # itself validate); MultiTaskCarNet's defaults match configs/default.yaml.
    model = MultiTaskCarNet(backbone_name=backbone_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return checkpoint, model, backbone_name, target_norm


def export_onnx(model: MultiTaskCarNet, img_size: int, onnx_path: Path) -> None:
    wrapper = _ONNXExportWrapper(model).eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "year_z": {0: "batch"},
            "log_price_z": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"[export_onnx] Wrote ONNX graph to {onnx_path}")


def verify_onnx(onnx_path: Path, img_size: int) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(2, 3, img_size, img_size).astype(np.float32)  # batch=2 also exercises the dynamic axis
    outputs = session.run(None, {input_name: dummy})

    if len(outputs) != len(OUTPUT_NAMES):
        raise RuntimeError(f"Expected {len(OUTPUT_NAMES)} ONNX outputs, got {len(outputs)}.")

    for name, arr in zip(OUTPUT_NAMES, outputs):
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"ONNX output '{name}' contains non-finite values: {arr}")
        print(f"[export_onnx] Verified output '{name}': shape={arr.shape}, dtype={arr.dtype}, finite=True")

    print("[export_onnx] ONNX model loads with onnxruntime and produces finite outputs. OK.")


def write_meta(meta_path: Path, target_norm: dict, backbone_name: str, img_size: int) -> None:
    meta = {
        "target_norm": target_norm,
        "backbone": backbone_name,
        "img_size": img_size,
        "outputs": OUTPUT_NAMES,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export_onnx] Wrote {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a MultiTaskCarNet checkpoint to ONNX for serving.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a .pt checkpoint from train.py.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write model.onnx/model.pt/model_meta.json.")
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Input resolution for the ONNX export. Defaults to the checkpoint's own img_size, then 224.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, model, backbone_name, target_norm = load_checkpoint_and_model(checkpoint_path)
    img_size = args.img_size or checkpoint.get("img_size") or 224
    print(f"[export_onnx] backbone={backbone_name} img_size={img_size} target_norm={target_norm}")

    onnx_path = out_dir / "model.onnx"
    export_onnx(model, img_size, onnx_path)
    verify_onnx(onnx_path, img_size)

    torch_path = out_dir / "model.pt"
    if checkpoint_path.resolve() == torch_path.resolve():
        print(f"[export_onnx] {torch_path} is already the source checkpoint, skipping copy.")
    else:
        try:
            shutil.copy2(checkpoint_path, torch_path)
        except OSError:
            # Cross-filesystem or otherwise uncopyable source: fall back to
            # re-saving the same in-memory checkpoint dict we loaded above.
            torch.save(checkpoint, torch_path)
        print(f"[export_onnx] Wrote torch checkpoint to {torch_path}")

    write_meta(out_dir / "model_meta.json", target_norm, backbone_name, img_size)


if __name__ == "__main__":
    main()
