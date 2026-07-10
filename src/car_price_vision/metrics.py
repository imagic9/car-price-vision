"""Evaluation metrics for the year and price regression heads.

All functions accept numpy arrays or anything array-like (torch tensors
are fine too, they get coerced via np.asarray after a .detach().cpu() if
needed by the caller). Shapes are all 1-D, one value per sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def mae_years(pred_year: np.ndarray, true_year: np.ndarray) -> float:
    """Mean absolute error of the year head, in years."""
    pred_year = _to_numpy(pred_year)
    true_year = _to_numpy(true_year)
    return float(np.mean(np.abs(pred_year - true_year)))


def mae_log(pred_log_price: np.ndarray, true_log_price: np.ndarray) -> float:
    """Mean absolute error in log-price space (unitless, log-GBP)."""
    pred_log_price = _to_numpy(pred_log_price)
    true_log_price = _to_numpy(true_log_price)
    return float(np.mean(np.abs(pred_log_price - true_log_price)))


def mape(pred_log_price: np.ndarray, true_log_price: np.ndarray, eps: float = 1e-6) -> float:
    """Mean absolute percentage error, computed on GBP prices recovered via
    exp() of the log-price predictions/targets (matches how log_price is
    constructed in data/dataset.py).
    """
    pred_price = np.exp(_to_numpy(pred_log_price))
    true_price = np.exp(_to_numpy(true_log_price))
    return float(np.mean(np.abs(pred_price - true_price) / np.maximum(np.abs(true_price), eps)) * 100.0)


def r2(pred: np.ndarray, true: np.ndarray) -> float:
    """R^2 (coefficient of determination). Works for either head; pass
    log-price arrays for the price head's R^2 in log space, or year arrays
    for the year head.
    """
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    return float(r2_score(true, pred))


def within_brand_corr(df: pd.DataFrame, pred_col: str = "pred_price", true_col: str = "price_gbp") -> dict:
    """Pearson correlation between predicted and true price, computed
    separately within each brand.

    This is one of the key diagnostics for the "brand shortcut" risk (see
    README Limitations): if the model has learned genuine within-brand
    price-relevant visual cues (condition, trim level, body style), the
    within-brand correlation should be meaningfully positive. If the model
    is mostly reading the badge/grille and mapping brand -> average price,
    within-brand correlation collapses toward zero even though *overall*
    (across-brand) correlation looks good.

    Args:
        df: DataFrame with at least columns "brand", pred_col, true_col.
        pred_col: column name holding predicted GBP price (exp of log pred).
        true_col: column name holding true GBP price.

    Returns:
        {
          "per_brand": {brand: pearson_r or None (if <3 samples / zero variance)},
          "mean": mean of the valid per-brand correlations,
          "n_brands_used": number of brands with a valid correlation,
        }
    """
    required = {"brand", pred_col, true_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    per_brand: dict[str, float | None] = {}
    for brand, group in df.groupby("brand"):
        if len(group) < 3:
            per_brand[brand] = None
            continue
        pred = group[pred_col].to_numpy(dtype=np.float64)
        true = group[true_col].to_numpy(dtype=np.float64)
        if np.std(pred) == 0 or np.std(true) == 0:
            per_brand[brand] = None
            continue
        corr = float(np.corrcoef(pred, true)[0, 1])
        per_brand[brand] = corr

    valid = [v for v in per_brand.values() if v is not None]
    mean_corr = float(np.mean(valid)) if valid else float("nan")

    return {
        "per_brand": per_brand,
        "mean": mean_corr,
        "n_brands_used": len(valid),
    }
