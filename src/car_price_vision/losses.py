"""Multi-task training loss: weighted Huber on year + Huber on log-price."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multitask_loss(
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weight_year: float = 1.0,
    weight_price: float = 1.0,
    huber_delta: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the combined training loss for the two regression heads.

    Args:
        preds: model output, {"year": Tensor(B,), "log_price": Tensor(B,)}.
        targets: same shape/keys, ground truth (see data/dataset.py).
        weight_year: loss weight for the year term (config: loss.weight_year).
        weight_price: loss weight for the log_price term (config: loss.weight_price).
        huber_delta: delta parameter shared by both Huber terms (config: loss.huber_delta).

    Returns:
        dict with keys "loss" (the weighted sum used for backprop),
        "loss_year", and "loss_price" (unweighted, for logging).
    """
    loss_year = F.huber_loss(preds["year"], targets["year"], delta=huber_delta)
    loss_price = F.huber_loss(preds["log_price"], targets["log_price"], delta=huber_delta)
    total = weight_year * loss_year + weight_price * loss_price

    return {
        "loss": total,
        "loss_year": loss_year.detach(),
        "loss_price": loss_price.detach(),
    }
