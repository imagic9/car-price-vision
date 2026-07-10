"""Small regression heads sitting on top of the shared backbone features."""

from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """A small MLP regression head: Linear -> ReLU -> Dropout -> Linear(1).

    Used for both the year head and the log-price head (see
    models/multitask.py); they share this implementation but have
    independent weights.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (B, feature_dim). Returns: (B,) scalar predictions."""
        return self.net(x).squeeze(-1)
