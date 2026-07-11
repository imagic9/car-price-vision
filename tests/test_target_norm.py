"""Tests for the z-score standardization round-trip used for the two
regression targets (year, log_price).

The formulas here are copied from data/dataset.py (__getitem__, forward
direction) and eval.py:predict (inverse direction) rather than re-derived,
so a regression in either file's arithmetic would be caught by a mismatch
against these hand-written formulas. Constants come from configs/default.yaml
`target_norm`.
"""

from __future__ import annotations

import numpy as np
import pytest

# configs/default.yaml -> target_norm
YEAR_MEAN = 2012.3723
YEAR_STD = 4.4677
LOG_PRICE_MEAN = 9.0403
LOG_PRICE_STD = 0.9975


def standardize_year(year: float) -> float:
    """data/dataset.py __getitem__: year_t = (year - year_mean) / year_std"""
    return (year - YEAR_MEAN) / YEAR_STD


def destandardize_year(year_z: float) -> float:
    """eval.py predict: real_year = pred_year_z * year_std + year_mean"""
    return year_z * YEAR_STD + YEAR_MEAN


def standardize_log_price(log_price: float) -> float:
    """data/dataset.py __getitem__: logp_t = (log_price - log_price_mean) / log_price_std"""
    return (log_price - LOG_PRICE_MEAN) / LOG_PRICE_STD


def destandardize_log_price(log_price_z: float) -> float:
    """eval.py predict: real_log_price = pred_logprice_z * log_price_std + log_price_mean"""
    return log_price_z * LOG_PRICE_STD + LOG_PRICE_MEAN


@pytest.mark.parametrize("year", [1990, 2005, 2012, 2018, 2023])
def test_year_standardize_destandardize_round_trip(year):
    year_z = standardize_year(year)
    recovered = destandardize_year(year_z)
    assert recovered == pytest.approx(year, abs=1e-9)


@pytest.mark.parametrize("price_gbp", [500.0, 3000.0, 9500.0, 25000.0, 120000.0])
def test_price_standardize_destandardize_round_trip(price_gbp):
    log_price = float(np.log(price_gbp))
    log_price_z = standardize_log_price(log_price)

    recovered_log_price = destandardize_log_price(log_price_z)
    assert recovered_log_price == pytest.approx(log_price, abs=1e-9)

    recovered_price = float(np.exp(recovered_log_price))
    assert recovered_price == pytest.approx(price_gbp, rel=1e-6)


def test_standardized_targets_are_roughly_unit_scale():
    """Sanity check on the motivation stated in dataset.py's docstring: values
    near the mean should land close to 0 in z-space.
    """
    assert standardize_year(YEAR_MEAN) == pytest.approx(0.0, abs=1e-9)
    assert standardize_log_price(LOG_PRICE_MEAN) == pytest.approx(0.0, abs=1e-9)
