"""Leakage-safe train/val/test splitting.

Why this file exists (see README "Limitations" for the full discussion):
DVM-CAR has many photos per advert and many adverts per car model. A naive
random split over *images* leaks near-duplicate photos of the same physical
car (or the same model/trim) across train and val/test, which inflates
apparent accuracy without the model having learned anything generalizable.

Two split modes are supported:

- "by_advert": split at the *advert* level, grouping by `adv_id` so that
  every image belonging to one advert ends up in the same split. Adverts
  are then randomly assigned to train/val/test. This still allows the
  *same car model* to appear in both train and test (e.g. two different
  Ford Fiesta adverts), which is realistic for the deployed use case
  (predicting price for an arbitrary photo) but optimistic about
  generalization to genuinely new models.

- "by_model": additionally carves out a fraction of *car models*
  (identified by `genmodel_id`, DVM-CAR's stable per-model id -- falls back
  to the `model` name column if `genmodel_id` isn't present, e.g. for a
  non-DVM-CAR manifest) entirely into a held-out set that never appears in
  train/val/test. The remaining rows are then split by `adv_id` exactly as
  in "by_advert". This is the strict, leakage-safe test of whether the
  model has learned transferable visual design cues rather than memorizing
  per-model price/year priors -- directly relevant to the "brand/model
  shortcut" risk called out in the README.

No-leakage guarantee
---------------------
1. Grouping by `adv_id`: every `adv_id` value present in the manifest is
   assigned to exactly ONE of {train, val, test, holdout}. Consequently no
   two images of the same advert can ever straddle a split boundary.
2. Grouping by model for holdout: in "by_model" mode, the holdout set is
   built by selecting a subset of `genmodel_id` (or `model`) values BEFORE
   the train/val/test split is computed, and every row whose model is in
   that holdout set is removed from the train/val/test pool. Therefore the
   set of model ids in `holdout_idx` is, by construction, disjoint from the
   set of model ids in `train_idx` / `val_idx` / `test_idx`.
If the manifest does not contain an `adv_id` column (e.g. a minimal
fallback manifest), splitting degrades gracefully to per-row grouping
(each row is its own group), which is a known limitation -- callers should
prefer manifests produced by scripts/build_manifest.py, which always
include `adv_id`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _split_groups(
    group_ids: np.ndarray,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> tuple[set, set, set]:
    """Randomly partition an array of (possibly repeated) group ids into
    disjoint train/val/test *sets of unique group ids*, sized by fraction
    of unique groups.
    """
    unique_groups = np.unique(group_ids)
    rng.shuffle(unique_groups)
    n_groups = len(unique_groups)
    n_val = int(n_groups * val_frac)
    n_test = int(n_groups * test_frac)

    val_groups = set(unique_groups[:n_val])
    test_groups = set(unique_groups[n_val : n_val + n_test])
    train_groups = set(unique_groups[n_val + n_test :])
    return train_groups, val_groups, test_groups


def make_splits(
    df: pd.DataFrame,
    mode: str = "by_advert",
    holdout_models_frac: float = 0.1,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Compute leakage-safe split indices over a manifest DataFrame.

    Args:
        df: manifest DataFrame (see dataset.REQUIRED_COLUMNS). Must contain
            at least a "model" column. Leakage-safety at the advert level
            requires an "adv_id" column (present in manifests produced by
            scripts/build_manifest.py); "by_model" mode prefers a
            "genmodel_id" column over "model" for the holdout grouping,
            since it is a stable id rather than a free-text name.
        mode: "by_advert" or "by_model" (see module docstring).
        holdout_models_frac: fraction of distinct model ids (genmodel_id if
            present, else model) reserved for the unseen-models holdout
            set. Only used when mode == "by_model"; ignored (holdout_idx
            will be empty) for "by_advert".
        val_frac: fraction of the *remaining* (non-holdout) advert groups
            used for val.
        test_frac: fraction of the *remaining* (non-holdout) advert groups
            used for test.
        seed: RNG seed for reproducibility.

    Returns:
        dict with keys "train", "val", "test", "holdout" mapping to
        integer numpy arrays of row-positions into `df` (i.e. suitable for
        `df.iloc[idx]` or `DVMCarDataset(df, indices=idx)`).

    Raises:
        ValueError: if `mode` is not recognized, or if fractions are
            invalid, or if the required columns are missing.
    """
    if mode not in ("by_advert", "by_model"):
        raise ValueError(f"Unknown split mode: {mode!r}. Expected 'by_advert' or 'by_model'.")
    if "model" not in df.columns:
        raise ValueError("Manifest must have a 'model' column for leakage-safe splitting.")
    if not (0.0 <= val_frac < 1.0 and 0.0 <= test_frac < 1.0 and val_frac + test_frac < 1.0):
        raise ValueError("val_frac/test_frac must be in [0, 1) and sum to < 1.")

    rng = np.random.default_rng(seed)
    n = len(df)
    all_idx = np.arange(n)

    holdout_idx = np.array([], dtype=int)
    remaining_mask = np.ones(n, dtype=bool)

    if mode == "by_model":
        model_col = "genmodel_id" if "genmodel_id" in df.columns else "model"
        shuffled_models = np.asarray(df[model_col].unique(), dtype=object)
        rng.shuffle(shuffled_models)
        n_holdout_models = max(1, int(len(shuffled_models) * holdout_models_frac)) if len(shuffled_models) > 0 else 0
        holdout_models = set(shuffled_models[:n_holdout_models])

        is_holdout = df[model_col].isin(holdout_models).to_numpy()
        holdout_idx = all_idx[is_holdout]
        remaining_mask = ~is_holdout

    remaining_idx = all_idx[remaining_mask]

    if "adv_id" in df.columns:
        group_ids = df["adv_id"].to_numpy()[remaining_idx]
    else:
        # No advert id available: fall back to per-row grouping (each row
        # is its own group). This is a known limitation -- see docstring.
        group_ids = remaining_idx

    train_groups, val_groups, test_groups = _split_groups(group_ids, val_frac, test_frac, rng)

    if "adv_id" in df.columns:
        remaining_group_series = df["adv_id"].to_numpy()[remaining_idx]
    else:
        remaining_group_series = remaining_idx

    is_val = np.isin(remaining_group_series, list(val_groups)) if len(val_groups) else np.zeros(
        len(remaining_idx), dtype=bool
    )
    is_test = np.isin(remaining_group_series, list(test_groups)) if len(test_groups) else np.zeros(
        len(remaining_idx), dtype=bool
    )
    is_train = ~(is_val | is_test)

    val_idx = remaining_idx[is_val]
    test_idx = remaining_idx[is_test]
    train_idx = remaining_idx[is_train]

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "holdout": holdout_idx,
    }
