"""Tests for the evaluation metrics in metrics.py.

Values are hand-computed on small arrays so failures are easy to reason
about; within_brand_corr is exercised on synthetic DataFrames engineered to
have known (near) +1 / -1 within-brand correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from car_price_vision.metrics import mae_log, mae_years, mape, r2, within_brand_corr


def test_mae_years_hand_computed():
    pred = np.array([2010.0, 2015.0, 2020.0])
    true = np.array([2012.0, 2015.0, 2018.0])
    # |2010-2012|=2, |2015-2015|=0, |2020-2018|=2 -> mean = 4/3
    assert mae_years(pred, true) == pytest.approx(4.0 / 3.0)


def test_mae_years_zero_for_perfect_predictions():
    values = np.array([2001.0, 1999.0, 2020.0])
    assert mae_years(values, values) == pytest.approx(0.0)


def test_mae_log_hand_computed():
    pred = np.array([9.0, 9.5, 8.0])
    true = np.array([9.2, 9.5, 8.3])
    # |9.0-9.2|=0.2, |9.5-9.5|=0, |8.0-8.3|=0.3 -> mean = 0.5/3
    assert mae_log(pred, true) == pytest.approx(0.5 / 3.0)


def test_mae_log_zero_for_perfect_predictions():
    values = np.array([9.0, 9.5, 8.0])
    assert mae_log(values, values) == pytest.approx(0.0)


def test_mape_hand_computed():
    # true prices [100, 200] GBP; pred prices [110, 180] GBP
    true_log = np.log(np.array([100.0, 200.0]))
    pred_log = np.log(np.array([110.0, 180.0]))
    # ape = [10/100, 20/200] = [0.1, 0.1] -> mean * 100 = 10.0%
    assert mape(pred_log, true_log) == pytest.approx(10.0)


def test_mape_zero_for_perfect_predictions():
    true_log = np.log(np.array([100.0, 200.0, 5000.0]))
    assert mape(true_log, true_log) == pytest.approx(0.0)


def test_r2_perfect_predictions():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(true, true) == pytest.approx(1.0)


def test_r2_mean_predictor_is_zero():
    true = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    pred = np.full_like(true, true.mean())
    assert r2(pred, true) == pytest.approx(0.0, abs=1e-9)


def test_metrics_accept_torch_tensor_and_numpy_array():
    pred_np = np.array([2010.0, 2015.0, 2020.0])
    true_np = np.array([2012.0, 2015.0, 2018.0])
    pred_t = torch.tensor(pred_np)
    true_t = torch.tensor(true_np)

    result_np = mae_years(pred_np, true_np)
    result_torch = mae_years(pred_t, true_t)
    assert result_np == pytest.approx(result_torch)


def _brand_df(brand_prices: dict[str, list[float]], pred_fn) -> pd.DataFrame:
    """Build a brand/pred_price/price_gbp DataFrame; pred_fn maps a true-price
    array to a predicted-price array for that brand.
    """
    rows = []
    for brand, prices in brand_prices.items():
        true_arr = np.array(prices, dtype=np.float64)
        pred_arr = pred_fn(true_arr)
        for t, p in zip(true_arr, pred_arr):
            rows.append({"brand": brand, "price_gbp": t, "pred_price": p})
    return pd.DataFrame(rows)


def test_within_brand_corr_perfect_positive():
    brand_prices = {
        "Ford": [8000, 9000, 10000, 12000, 15000],
        "BMW": [20000, 25000, 30000, 35000, 40000],
    }
    # pred = linear function of true with positive slope -> correlation +1
    df = _brand_df(brand_prices, lambda true_arr: true_arr * 1.1 + 500)

    result = within_brand_corr(df)
    assert result["n_brands_used"] == 2
    for brand, corr in result["per_brand"].items():
        assert corr == pytest.approx(1.0, abs=1e-8)
    assert result["mean"] == pytest.approx(1.0, abs=1e-8)


def test_within_brand_corr_perfect_negative():
    brand_prices = {
        "Ford": [8000, 9000, 10000, 12000, 15000],
        "BMW": [20000, 25000, 30000, 35000, 40000],
    }
    # pred = negative linear function of true -> correlation -1
    df = _brand_df(brand_prices, lambda true_arr: -true_arr + 100000)

    result = within_brand_corr(df)
    assert result["n_brands_used"] == 2
    for brand, corr in result["per_brand"].items():
        assert corr == pytest.approx(-1.0, abs=1e-8)
    assert result["mean"] == pytest.approx(-1.0, abs=1e-8)


def test_within_brand_corr_respects_min_samples_threshold():
    """Brands with fewer than 3 samples are excluded (per_brand -> None) and
    do not count toward n_brands_used.
    """
    df = _brand_df(
        {
            "Ford": [8000, 9000, 10000, 12000],  # 4 samples: used
            "Tesla": [50000, 60000],  # 2 samples: below threshold, excluded
        },
        lambda true_arr: true_arr * 1.05,
    )

    result = within_brand_corr(df)
    assert result["per_brand"]["Tesla"] is None
    assert result["per_brand"]["Ford"] == pytest.approx(1.0, abs=1e-8)
    assert result["n_brands_used"] == 1


def test_within_brand_corr_missing_columns_raises():
    df = pd.DataFrame({"brand": ["Ford"], "price_gbp": [1000.0]})
    with pytest.raises(ValueError):
        within_brand_corr(df)
