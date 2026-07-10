"""Self-contained Grad-CAM implementation (no dependency on the external
`grad-cam` pip package).

Grad-CAM (Selvaraju et al., 2017) explains a scalar model output by
weighting the activation maps of a chosen convolutional layer by the
gradient of that output with respect to those activations, then averaging
over channels. Here the "output" being explained is one of our regression
heads' scalar prediction (year or log_price) rather than a classification
logit — the mechanics are identical, we just skip the softmax/class-index
step.

This is central to the brand-shortcut investigation described in the
README: running Grad-CAM on the `log_price` head and checking whether the
heatmap concentrates on the badge/grille (shortcut) vs. spread across
body panels/proportions (genuine visual reasoning) is one of the main
diagnostics used in notebooks/04_interpretability.ipynb.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import matplotlib
except ImportError:  # pragma: no cover - matplotlib is a listed dependency
    matplotlib = None


class GradCAM:
    """Grad-CAM over a single target layer of a CNN backbone.

    Typical usage::

        model = MultiTaskCarNet(backbone_name="convnext_tiny").eval()
        target_layer = model.backbone.default_target_layer()
        cam_extractor = GradCAM(model, target_layer)
        cam = cam_extractor(image_tensor.unsqueeze(0), output_key="log_price")
        cam_extractor.remove_hooks()

    Note on batching: gradients are computed from the *sum* of the target
    output over the batch, which is equivalent to computing per-sample
    gradients independently (cross-sample gradient terms are zero for a
    feedforward CNN), so this is safe to call with batch size > 1.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, inputs, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(
        self, x: torch.Tensor, output_key: Literal["year", "log_price"] = "log_price"
    ) -> np.ndarray:
        """Compute the Grad-CAM heatmap for a batch of images.

        Args:
            x: input batch, shape (B, 3, H, W), already normalized as the
                model expects (see data/transforms.py).
            output_key: which regression head's scalar output to explain,
                "year" or "log_price".

        Returns:
            cam: numpy array of shape (B, H', W') in [0, 1], where H'/W'
                are the target layer's spatial resolution (typically much
                smaller than the input; upsample with overlay_heatmap).
        """
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        x = x.clone().requires_grad_(True)
        preds = self.model(x)
        score = preds[output_key].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire. Make sure `target_layer` is actually "
                "part of the forward graph for this model/output_key."
            )

        # Global-average-pool the gradients over spatial dims -> per-channel weight.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1)  # (B, H', W')
        cam = torch.relu(cam)

        # Per-sample min-max normalize to [0, 1].
        b = cam.shape[0]
        flat = cam.view(b, -1)
        cam_min = flat.min(dim=1).values.view(b, 1, 1)
        cam_max = flat.max(dim=1).values.view(b, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        self.model.train(was_training)
        return cam.cpu().numpy()

    def remove_hooks(self) -> None:
        """Detach the forward/backward hooks. Call when done to avoid
        leaking hooks if the GradCAM object goes out of scope but the
        model stays alive (e.g. in the serving app).
        """
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()


def overlay_heatmap(img: Image.Image, cam: np.ndarray, alpha: float = 0.45, colormap: str = "jet") -> Image.Image:
    """Blend a Grad-CAM heatmap onto the original image for visualization.

    Args:
        img: original PIL image (any size); the heatmap is resized to match.
        cam: 2-D array in [0, 1] (a single sample's output from GradCAM.__call__,
            e.g. `cam_batch[0]`).
        alpha: blend strength of the heatmap over the original image.
        colormap: a matplotlib colormap name.

    Returns:
        A new RGB PIL Image the same size as `img`.
    """
    if matplotlib is None:  # pragma: no cover
        raise ImportError("matplotlib is required for overlay_heatmap (see requirements.txt).")

    img_rgb = img.convert("RGB")
    cam_img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8))
    cam_resized = cam_img.resize(img_rgb.size, resample=Image.BILINEAR)
    cam_arr = np.asarray(cam_resized).astype(np.float32) / 255.0

    cmap = matplotlib.colormaps[colormap]
    heatmap_rgba = cmap(cam_arr)  # (H, W, 4) in [0, 1]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap_rgb)

    return Image.blend(img_rgb, heatmap_img, alpha=alpha)
