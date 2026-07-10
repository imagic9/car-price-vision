"""The combined multi-task model: shared backbone + year head + price head."""

from __future__ import annotations

import torch
import torch.nn as nn

from car_price_vision.models.backbone import BackboneWrapper, build_backbone
from car_price_vision.models.heads import RegressionHead


class MultiTaskCarNet(nn.Module):
    """Shared visual backbone with two independent regression heads.

    forward(x) -> {"year": Tensor(B,), "log_price": Tensor(B,)}

    Predicting `log_price` rather than raw GBP price keeps the target
    roughly symmetric/Gaussian-ish and lets a single Huber loss scale work
    for both cars worth a few hundred pounds and luxury cars worth six
    figures; see metrics.py for how this is converted back to GBP for MAPE.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone: BackboneWrapper
        self.backbone, feature_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.year_head = RegressionHead(feature_dim, hidden_dim=head_hidden_dim, dropout=head_dropout)
        self.price_head = RegressionHead(feature_dim, hidden_dim=head_hidden_dim, dropout=head_dropout)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            "year": self.year_head(features),
            "log_price": self.price_head(features),
        }

    def freeze_backbone(self) -> None:
        """Stage 1 of two-stage training: freeze backbone, train heads only."""
        self.backbone.freeze()

    def unfreeze_backbone_last_blocks(self, n: int) -> None:
        """Stage 2 of two-stage training: unfreeze the last `n` backbone blocks."""
        self.backbone.unfreeze_last_blocks(n)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Convenience accessor for building an optimizer over only the
        currently-trainable parameters (respects freeze/unfreeze state).
        """
        return [p for p in self.parameters() if p.requires_grad]
