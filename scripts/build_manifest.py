"""Build the unified manifest CSV that DVMCarDataset expects, from raw
DVM-CAR metadata tables.

DVM-CAR ships metadata as several separate CSVs (basic table info, ad
table, sales table, image table, etc. -- see the dataset's own
documentation for exact filenames, which vary by release). This script is
a STUB: it defines the target schema and a skeleton pipeline, but the
actual column mapping from raw DVM-CAR files must be filled in once the
raw files are staged on `rtx` (phase 1).

Target manifest schema (consumed by car_price_vision.data.dataset.DVMCarDataset)
----------------------------------------------------------------------------------
image_path   : str   path to the image, relative to --data-root (or absolute)
year         : int   manufacture year of the vehicle
price_gbp    : float advertised price in GBP
model        : str   car model name
brand        : str   car brand/make
advert_year  : int   year the advert was posted (for inflation control)

Usage (once the TODOs below are filled in):
    python scripts/build_manifest.py \\
        --raw-dir /path/to/dvm-car/raw \\
        --data-root /path/to/dvm-car/images \\
        --out data/manifest.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = ["image_path", "year", "price_gbp", "model", "brand", "advert_year"]


def load_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the raw DVM-CAR metadata CSVs.

    TODO(phase 1): replace with the actual DVM-CAR filenames, e.g.:
        basic_df = pd.read_csv(raw_dir / "Basic_table.csv")
        ad_df = pd.read_csv(raw_dir / "Ad_table.csv")
        sales_df = pd.read_csv(raw_dir / "Sales_table.csv")
        image_df = pd.read_csv(raw_dir / "Image_table.csv")
    Return whatever subset is needed to build the manifest below.
    """
    raise NotImplementedError(
        "load_raw_tables is a stub. TODO(phase 1): implement loading of the real "
        "DVM-CAR metadata CSVs from `raw_dir` once they are staged on `rtx`."
    )


def build_manifest(raw_tables: dict[str, pd.DataFrame], data_root: Path) -> pd.DataFrame:
    """Join/derive the raw DVM-CAR tables into the unified manifest schema.

    TODO(phase 1): implement the real join. Rough sketch, once table names
    and key columns are known:
        1. Join Ad_table (price, advert date, model/brand) with Image_table
           (image file path per advert / vehicle) on a shared advert or
           vehicle id.
        2. Derive `year` from the basic/ad table's registration/manufacture
           year field.
        3. Derive `advert_year` from the advert's posting date field.
        4. Normalize `brand`/`model` strings (casing, whitespace).
        5. Drop rows with missing image files, non-positive price, or
           implausible year (e.g. < 1980 or > current year + 1).
    """
    raise NotImplementedError(
        "build_manifest is a stub. TODO(phase 1): implement the real column "
        f"mapping from DVM-CAR raw tables to the target schema: {MANIFEST_COLUMNS}."
    )


def validate_manifest(df: pd.DataFrame) -> None:
    """Sanity checks before writing the manifest to disk."""
    missing_cols = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_cols)}")
    if df[MANIFEST_COLUMNS].isnull().any().any():
        n_bad = df[MANIFEST_COLUMNS].isnull().any(axis=1).sum()
        raise ValueError(f"Manifest has {n_bad} rows with null values in required columns.")
    if (df["price_gbp"] <= 0).any():
        raise ValueError("Manifest has rows with non-positive price_gbp.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the unified DVM-CAR manifest CSV.")
    parser.add_argument("--raw-dir", type=str, required=True, help="Directory with raw DVM-CAR metadata CSVs.")
    parser.add_argument("--data-root", type=str, required=True, help="Directory containing the actual image files.")
    parser.add_argument("--out", type=str, default="data/manifest.csv", help="Output path for the manifest CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    data_root = Path(args.data_root)
    out_path = Path(args.out)

    raw_tables = load_raw_tables(raw_dir)
    manifest = build_manifest(raw_tables, data_root)
    validate_manifest(manifest)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest[MANIFEST_COLUMNS].to_csv(out_path, index=False)
    print(f"Wrote {len(manifest)} rows to {out_path}")


if __name__ == "__main__":
    main()
