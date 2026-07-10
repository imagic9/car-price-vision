"""Pretrained CNN/ViT backbones used as the shared feature extractor.

Supported backbones (torchvision, ImageNet-1k pretrained weights) and their
pooled feature dimensionality:

    convnext_tiny      -> 768
    efficientnet_v2_s   -> 1280
    vit_b_16            -> 768

Each backbone's classification head is stripped; `forward` returns a single
pooled feature vector per image, ready to feed into the regression heads in
heads.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_V2_S_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    efficientnet_v2_s,
    vit_b_16,
)

FEATURE_DIMS: dict[str, int] = {
    "convnext_tiny": 768,
    "efficientnet_v2_s": 1280,
    "vit_b_16": 768,
}

SUPPORTED_BACKBONES = tuple(FEATURE_DIMS.keys())


class BackboneWrapper(nn.Module):
    """Wraps a torchvision backbone to expose a uniform pooled-feature API,
    plus freeze/unfreeze helpers for two-stage transfer learning
    (see train.py).
    """

    def __init__(self, name: str, model: nn.Module) -> None:
        super().__init__()
        if name not in FEATURE_DIMS:
            raise ValueError(f"Unsupported backbone: {name!r}. Choose from {SUPPORTED_BACKBONES}.")
        self.name = name
        self.model = model
        self.feature_dim = FEATURE_DIMS[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "vit_b_16":
            # `self.model.heads` has been replaced with nn.Identity() in
            # build_backbone, so the forward pass already returns the
            # pooled [CLS]-token feature vector of shape (B, 768).
            return self.model(x)

        # CNN backbones (convnext_tiny, efficientnet_v2_s): re-implement the
        # tail of torchvision's forward manually so we stop *before* the
        # classifier, since we removed the classifier head.
        features = self.model.features(x)
        pooled = self.model.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        return pooled

    # -- last-conv/last-block accessor, used by Grad-CAM by default --------
    def default_target_layer(self) -> nn.Module:
        """Return the layer Grad-CAM should hook into by default: the last
        spatial block before pooling. Only meaningful for CNN backbones;
        raises for ViT (Grad-CAM needs a different, attention-based
        treatment there, out of scope for the MVP — see interpret/gradcam.py).
        """
        if self.name == "convnext_tiny":
            return self.model.features[-1]
        if self.name == "efficientnet_v2_s":
            return self.model.features[-1]
        raise NotImplementedError(
            f"No default conv target layer for backbone '{self.name}'. "
            "Grad-CAM in this repo targets CNN feature maps; use a "
            "convnext_tiny/efficientnet_v2_s backbone, or extend "
            "interpret/gradcam.py with an attention-rollout method for ViT."
        )

    # -- two-stage transfer learning helpers --------------------------------
    def freeze(self) -> None:
        """Freeze all backbone parameters (stage 1: train heads only)."""
        for p in self.model.parameters():
            p.requires_grad_(False)

    def unfreeze_last_blocks(self, n: int) -> None:
        """Unfreeze the last `n` top-level blocks for fine-tuning (stage 2).

        The rest of the backbone stays frozen. "Block" means:
          - convnext_tiny / efficientnet_v2_s: last `n` entries of
            `model.features` (each entry is itself a stage/sequential group).
          - vit_b_16: last `n` transformer encoder layers
            (`model.encoder.layers`).
        """
        if n <= 0:
            return

        if self.name in ("convnext_tiny", "efficientnet_v2_s"):
            blocks = list(self.model.features.children())
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad_(True)
        elif self.name == "vit_b_16":
            layers = list(self.model.encoder.layers.children())
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            # Also unfreeze the final encoder LayerNorm, which directly
            # feeds the pooled feature used by the heads.
            for p in self.model.encoder.ln.parameters():
                p.requires_grad_(True)
        else:  # pragma: no cover - guarded by FEATURE_DIMS check in __init__
            raise ValueError(f"Unsupported backbone: {self.name!r}")


def build_backbone(name: str, pretrained: bool = True) -> tuple[BackboneWrapper, int]:
    """Build a pretrained backbone with its classifier head removed.

    Args:
        name: one of SUPPORTED_BACKBONES.
        pretrained: if True, load ImageNet-1k pretrained weights.

    Returns:
        (feature_extractor, feature_dim) where feature_extractor is a
        BackboneWrapper whose forward(x) -> Tensor of shape (B, feature_dim).
    """
    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        model.classifier = nn.Identity()
    elif name == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_v2_s(weights=weights)
        model.classifier = nn.Identity()
    elif name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = vit_b_16(weights=weights)
        model.heads = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone: {name!r}. Choose from {SUPPORTED_BACKBONES}.")

    wrapper = BackboneWrapper(name=name, model=model)
    return wrapper, wrapper.feature_dim
