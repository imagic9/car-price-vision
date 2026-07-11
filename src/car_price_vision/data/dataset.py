"""Manifest-driven PyTorch Dataset for DVM-CAR (or any dataset with the same
unified manifest schema, e.g. a Stanford Cars fallback prepared by
scripts/build_manifest.py).

Expected manifest CSV columns
------------------------------
image_path   : str   path to the image, absolute or relative to `data_root`
year         : int   manufacture year of the vehicle (regression target 1)
price_gbp    : float advertised price in GBP (regression target 2, via log)
model        : str   car model name (used for leakage-safe splits)
brand        : str   car brand/make (used for within-brand analysis)
advert_year  : int   year the advert was posted (used to control for
                      GBP price inflation over time, see README Limitations)

TODO(phase 1): confirm final column names once scripts/build_manifest.py
is filled in against the real DVM-CAR metadata tables, and update this
docstring + REQUIRED_COLUMNS together.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["image_path", "year", "price_gbp", "model", "brand", "advert_year"]


class DVMCarDataset(Dataset):
    """Loads (image, targets) pairs from a unified manifest CSV.

    Targets returned per sample:
        {"year": float32 scalar tensor, "log_price": float32 scalar tensor}

    `log_price` is `log(price_gbp)` (natural log); training/eval code is
    expected to `exp()` predictions back to GBP when reporting MAPE, etc.
    (see metrics.py).

    Missing/corrupt images are handled gracefully: __getitem__ falls back to
    a blank (zero) image rather than raising, so a few bad files on disk
    don't crash a long-running training job. Counts are logged.
    """

    def __init__(
        self,
        manifest: pd.DataFrame | str | Path,
        data_root: str | Path | None = None,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        indices: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            manifest: either a pre-loaded DataFrame or a path to the
                manifest CSV produced by scripts/build_manifest.py.
            data_root: base directory that relative `image_path` values are
                resolved against. If None, `image_path` is used as-is.
            transform: callable applied to the loaded PIL image, typically
                one of transforms.train_transforms() / eval_transforms().
            indices: optional row indices (e.g. from splits.make_splits) to
                restrict this dataset to a subset of the manifest without
                copying the underlying DataFrame.
        """
        if isinstance(manifest, (str, Path)):
            df = pd.read_csv(manifest)
        else:
            df = manifest

        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Manifest is missing required columns: {sorted(missing_cols)}. "
                f"Expected schema: {REQUIRED_COLUMNS}. "
                "See scripts/build_manifest.py to (re)generate the manifest."
            )

        self.df = df.reset_index(drop=True) if indices is None else df.iloc[indices].reset_index(drop=True)
        self.data_root = Path(data_root) if data_root is not None else None
        self.transform = transform
        self._n_load_failures = 0

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_path(self, image_path: str) -> Path:
        p = Path(image_path)
        if self.data_root is not None and not p.is_absolute():
            return self.data_root / p
        return p

    def _load_image(self, image_path: str) -> Image.Image:
        path = self._resolve_path(image_path)
        try:
            with Image.open(path) as img:
                return img.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            self._n_load_failures += 1
            logger.warning("Failed to load image %s (%s). Using blank fallback.", path, exc)
            # 224x224 is the default img_size; the transform pipeline resizes
            # anyway, so any placeholder size works here.
            return Image.new("RGB", (224, 224), color=(0, 0, 0))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]
        image = self._load_image(row["image_path"])

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            # Fallback: minimal to-tensor without resizing/normalizing.
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        price_gbp = float(row["price_gbp"])
        log_price = float(np.log(max(price_gbp, 1e-6)))

        targets: dict[str, torch.Tensor] = {
            "year": torch.tensor(float(row["year"]), dtype=torch.float32),
            "log_price": torch.tensor(log_price, dtype=torch.float32),
        }
        return image_tensor, targets

    def row_metadata(self, idx: int) -> dict[str, Any]:
        """Return the raw manifest row (brand/model/advert_year/etc.) for
        analyses that need more than the training targets, e.g.
        metrics.within_brand_corr or the interpretability notebook.
        """
        return self.df.iloc[idx].to_dict()
