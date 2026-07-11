"""Tests for the leakage-safe splitting logic in data/splits.py.

These are the most important tests in the suite: splits.py exists
specifically to prevent advert-level and model-level leakage between
train/val/test/holdout, so we assert those invariants directly rather than
just checking that the function runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from car_price_vision.data.splits import _split_groups, make_splits

N_MODELS = 10
ADVERTS_PER_MODEL = 8
PHOTOS_PER_ADVERT = 3  # 10 * 8 * 3 = 240 rows


def build_manifest(
    n_models: int = N_MODELS,
    adverts_per_model: int = ADVERTS_PER_MODEL,
    photos_per_advert: int = PHOTOS_PER_ADVERT,
) -> pd.DataFrame:
    """Build a synthetic manifest with several photos per advert and several
    adverts per genmodel_id, mimicking the DVM-CAR manifest shape.
    """
    rows = []
    adv_counter = 0
    for model_idx in range(n_models):
        genmodel_id = f"gm_{model_idx}"
        model_name = f"Model_{model_idx}"
        brand = f"Brand_{model_idx % 3}"  # a few adverts models share a brand
        for _ in range(adverts_per_model):
            adv_id = f"adv_{adv_counter}"
            adv_counter += 1
            year = 2005 + (model_idx % 15)
            price = 3000 + model_idx * 500
            for _ in range(photos_per_advert):
                rows.append(
                    {
                        "adv_id": adv_id,
                        "genmodel_id": genmodel_id,
                        "model": model_name,
                        "brand": brand,
                        "year": year,
                        "price_gbp": price,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def manifest() -> pd.DataFrame:
    return build_manifest()


def _all_disjoint(*sets: set) -> bool:
    union_size = sum(len(s) for s in sets)
    merged = set().union(*sets)
    return union_size == len(merged)


def test_by_model_splits_disjoint_and_cover_all_rows(manifest):
    splits = make_splits(manifest, mode="by_model", seed=0)
    all_idx = np.concatenate([splits["train"], splits["val"], splits["test"], splits["holdout"]])

    # Every row assigned to exactly one split.
    assert sorted(all_idx.tolist()) == list(range(len(manifest)))

    idx_sets = [set(splits[k].tolist()) for k in ("train", "val", "test", "holdout")]
    assert _all_disjoint(*idx_sets)


def test_by_model_holdout_models_are_unseen(manifest):
    """No genmodel_id present in holdout may appear in train/val/test."""
    splits = make_splits(manifest, mode="by_model", holdout_models_frac=0.3, seed=1)

    holdout_models = set(manifest.iloc[splits["holdout"]]["genmodel_id"])
    other_models = set(manifest.iloc[np.concatenate([splits["train"], splits["val"], splits["test"]])]["genmodel_id"])

    assert holdout_models, "expected a non-empty holdout set of models"
    assert holdout_models.isdisjoint(other_models)


@pytest.mark.parametrize("mode", ["by_advert", "by_model"])
def test_no_advert_leakage_across_train_val_test(manifest, mode):
    """An adv_id (i.e. all photos of one advert) must land entirely within a
    single split -- never straddling train/val/test.
    """
    splits = make_splits(manifest, mode=mode, seed=2)

    adv_by_split = {
        name: set(manifest.iloc[splits[name]]["adv_id"]) for name in ("train", "val", "test")
    }
    assert _all_disjoint(*adv_by_split.values())


def test_same_seed_is_reproducible(manifest):
    splits_a = make_splits(manifest, mode="by_model", seed=7)
    splits_b = make_splits(manifest, mode="by_model", seed=7)

    for key in ("train", "val", "test", "holdout"):
        np.testing.assert_array_equal(splits_a[key], splits_b[key])


def test_different_seed_gives_different_split(manifest):
    splits_a = make_splits(manifest, mode="by_model", seed=1)
    splits_b = make_splits(manifest, mode="by_model", seed=999)

    same_everywhere = all(
        np.array_equal(splits_a[key], splits_b[key]) for key in ("train", "val", "test", "holdout")
    )
    assert not same_everywhere


def test_fractions_roughly_respected(manifest):
    """Group granularity (adv_id / genmodel_id) means exact fractions aren't
    achievable, so use a loose tolerance.
    """
    val_frac, test_frac = 0.2, 0.2
    splits = make_splits(manifest, mode="by_advert", val_frac=val_frac, test_frac=test_frac, seed=3)

    n_total = len(manifest)
    n_val = len(splits["val"])
    n_test = len(splits["test"])

    assert abs(n_val / n_total - val_frac) < 0.1
    assert abs(n_test / n_total - test_frac) < 0.1


def test_bad_mode_raises_value_error(manifest):
    with pytest.raises(ValueError):
        make_splits(manifest, mode="not_a_real_mode")


@pytest.mark.parametrize(
    "val_frac,test_frac",
    [(1.0, 0.0), (0.0, 1.0), (0.6, 0.6), (-0.1, 0.1)],
)
def test_bad_fractions_raise_value_error(manifest, val_frac, test_frac):
    with pytest.raises(ValueError):
        make_splits(manifest, mode="by_advert", val_frac=val_frac, test_frac=test_frac)


def test_split_groups_sizes_and_disjointness():
    """Unit test of the lower-level _split_groups helper in isolation."""
    group_ids = np.repeat(np.arange(50), 4)  # 50 groups, 4 rows each
    rng = np.random.default_rng(0)

    train_groups, val_groups, test_groups = _split_groups(group_ids, val_frac=0.2, test_frac=0.2, rng=rng)

    assert _all_disjoint(train_groups, val_groups, test_groups)
    assert len(val_groups) == 10
    assert len(test_groups) == 10
    assert len(train_groups) == 30
    assert train_groups | val_groups | test_groups == set(range(50))
