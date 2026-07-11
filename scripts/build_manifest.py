"""Build the unified manifest CSV that DVMCarDataset expects, from the real
DVM-CAR metadata tables and the on-disk image tree.

DVM-CAR layout on the data box (see README.md for the full description):

    <data-root>/tables/Ad_table.csv        ~268k rows, one row per advert
    <data-root>/tables/Image_table.csv     ~1.45M rows, one row per image
    <data-root>/images/resized_DVM/{Maker}/{Genmodel}/{Year}/{Color}/{Image_name}
    <data-root>/images/confirmed_fronts/{Maker}/{Year}/{Image_name}

We do NOT trust reconstructed paths from the CSVs -- we walk the actual
image directories on disk (so every path in the manifest is guaranteed to
be openable) and parse the vehicle/advert identifiers out of each image's
own filename, which encodes them as:

    "{Maker}$${Genmodel}$${Year}$${Color}$${Genmodel_ID}$${adv_num}$$image_{N}.jpg"

so `adv_id = f"{Genmodel_ID}$${adv_num}"`, which matches `Ad_table.Adv_ID`.

Target manifest schema (consumed by car_price_vision.data.dataset.DVMCarDataset)
----------------------------------------------------------------------------------
image_path   : str    path to the image, RELATIVE to --data-root, forward slashes
adv_id       : str     advert id, "{Genmodel_ID}$${adv_num}"
genmodel_id  : str     DVM-CAR model id, e.g. "10_1"
brand        : str    car brand/make (Ad_table.Maker)
model        : str    car model name (Ad_table.Genmodel)
year         : int    manufacture year (Ad_table.Reg_year)
price_gbp    : float  advertised price in GBP (Ad_table.Price)
advert_year  : int    year the advert was posted (Ad_table.Adv_year)
color        : str    Ad_table.Color
bodytype     : str    Ad_table.Bodytype
mileage      : float  Ad_table.Runned_Miles (NaN allowed)
viewpoint    : int    Image_table.Predicted_viewpoint, or -1 if not merged
is_front     : bool   True if the image is a confirmed front view (either it
                      came from images/confirmed_fronts, or its basename
                      appears in that curated set)

Usage:
    python scripts/build_manifest.py \\
        --data-root /data/car-price-vision \\
        --out /data/car-price-vision/manifest.csv \\
        [--source resized|fronts|both] \\
        [--limit N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST_COLUMNS = [
    "image_path",
    "adv_id",
    "genmodel_id",
    "brand",
    "model",
    "year",
    "price_gbp",
    "advert_year",
    "color",
    "bodytype",
    "mileage",
    "viewpoint",
    "is_front",
]

MIN_YEAR = 1950
MAX_YEAR = 2021
RANDOM_SEED = 42


def load_ad_table(data_root: Path) -> pd.DataFrame:
    """Load Ad_table.csv, tolerant of leading-space column headers."""
    path = data_root / "tables" / "Ad_table.csv"
    df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def load_image_table(data_root: Path) -> pd.DataFrame | None:
    """Load Image_table.csv for the optional viewpoint merge.

    Returns None (with a printed warning) if the file is missing or fails
    to load, in which case the caller should fall back to viewpoint=-1.
    """
    path = data_root / "tables" / "Image_table.csv"
    if not path.exists():
        print(f"[build_manifest] Image_table.csv not found at {path}; viewpoint will be -1 for all rows.")
        return None
    try:
        df = pd.read_csv(path, skipinitialspace=True, low_memory=False)
        df.columns = df.columns.str.strip()
        return df
    except Exception as exc:  # noqa: BLE001 - best-effort optional merge
        print(f"[build_manifest] Failed to load Image_table.csv ({exc}); viewpoint will be -1 for all rows.")
        return None


def discover_images(data_root: Path, source: str) -> list[tuple[Path, bool]]:
    """Walk the filesystem for image files, returning (path, is_front) pairs.

    `path` is relative to `data_root`. We walk the disk (rather than trust
    any reconstructed path from the CSVs) so every returned path is real
    and openable.
    """
    results: list[tuple[Path, bool]] = []

    # Basenames of the curated confirmed_fronts set: a resized_DVM image is a
    # confirmed front view iff its basename appears here. Without this
    # cross-mark, a resized-only manifest would have is_front=False on every
    # row even though ~61k of its images ARE confirmed fronts.
    fronts_dir = data_root / "images" / "confirmed_fronts"
    front_basenames: set[str] = (
        {p.name for p in fronts_dir.rglob("*.jpg")} if fronts_dir.exists() else set()
    )
    if front_basenames:
        print(f"[build_manifest] {len(front_basenames)} confirmed_fronts basenames loaded for is_front marking.")

    if source in ("resized", "both"):
        resized_dir = data_root / "images" / "resized_DVM"
        if resized_dir.exists():
            print(f"[build_manifest] Walking {resized_dir} ...")
            for p in resized_dir.rglob("*.jpg"):
                results.append((p.relative_to(data_root), p.name in front_basenames))
        else:
            print(f"[build_manifest] WARNING: {resized_dir} does not exist; skipping.")

    if source in ("fronts", "both"):
        if fronts_dir.exists():
            print(f"[build_manifest] Walking {fronts_dir} ...")
            for p in fronts_dir.rglob("*.jpg"):
                results.append((p.relative_to(data_root), True))
        else:
            print(f"[build_manifest] WARNING: {fronts_dir} does not exist; skipping.")

    print(f"[build_manifest] Discovered {len(results)} image files on disk (source={source}).")
    return results


def parse_image_rows(image_paths: list[tuple[Path, bool]]) -> tuple[pd.DataFrame, int]:
    """Parse each image's basename into its DVM-CAR identifiers.

    Basename format: "{Maker}$${Genmodel}$${Year}$${Color}$${Genmodel_ID}$${adv_num}$$image_{N}.jpg"

    Returns (parsed_df, n_skipped_bad_basename). `parsed_df` has columns:
    image_path, image_name, is_front, adv_id.
    """
    rows = []
    n_skipped = 0
    for rel_path, is_front in image_paths:
        basename = rel_path.name
        parts = basename.split("$$")
        if len(parts) < 7:
            n_skipped += 1
            continue
        genmodel_id = parts[4]
        adv_num = parts[5]
        adv_id = f"{genmodel_id}$${adv_num}"
        rows.append(
            {
                "image_path": rel_path.as_posix(),
                "image_name": basename,
                "is_front": is_front,
                "adv_id": adv_id,
            }
        )

    parsed_df = pd.DataFrame(rows, columns=["image_path", "image_name", "is_front", "adv_id"])
    return parsed_df, n_skipped


def build_manifest(
    ad_df: pd.DataFrame,
    image_df: pd.DataFrame | None,
    parsed_images: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Join parsed image rows with Ad_table (and optionally Image_table) to
    produce the final manifest, vectorized via pandas merge.

    Returns (manifest_df, stats) where stats counts drops by reason.
    """
    stats: dict[str, int] = {}

    n_before_join = len(parsed_images)

    ad_cols = [
        "Adv_ID",
        "Genmodel_ID",
        "Maker",
        "Genmodel",
        "Reg_year",
        "Price",
        "Adv_year",
        "Color",
        "Bodytype",
        "Runned_Miles",
    ]
    ad_slim = ad_df[[c for c in ad_cols if c in ad_df.columns]].drop_duplicates(subset=["Adv_ID"])

    merged = parsed_images.merge(ad_slim, left_on="adv_id", right_on="Adv_ID", how="left")

    n_unjoined = merged["Adv_ID"].isna().sum()
    stats["dropped_no_ad_table_match"] = int(n_unjoined)
    merged = merged[merged["Adv_ID"].notna()].copy()

    # Optional viewpoint merge on Image_name (basename).
    if image_df is not None and "Image_name" in image_df.columns and "Predicted_viewpoint" in image_df.columns:
        vp_slim = image_df[["Image_name", "Predicted_viewpoint"]].drop_duplicates(subset=["Image_name"])
        merged = merged.merge(vp_slim, left_on="image_name", right_on="Image_name", how="left")
        merged["viewpoint"] = merged["Predicted_viewpoint"].fillna(-1).astype(int)
    else:
        merged["viewpoint"] = -1

    manifest = pd.DataFrame(
        {
            "image_path": merged["image_path"],
            "adv_id": merged["adv_id"],
            "genmodel_id": merged["Genmodel_ID"],
            "brand": merged["Maker"],
            "model": merged["Genmodel"],
            "year": pd.to_numeric(merged["Reg_year"], errors="coerce"),
            "price_gbp": pd.to_numeric(merged["Price"], errors="coerce"),
            "advert_year": pd.to_numeric(merged["Adv_year"], errors="coerce"),
            "color": merged["Color"],
            "bodytype": merged["Bodytype"],
            "mileage": pd.to_numeric(merged["Runned_Miles"], errors="coerce"),
            "viewpoint": merged["viewpoint"],
            "is_front": merged["is_front"],
        }
    )

    stats["n_before_filters"] = len(manifest)

    # Filter: missing/<=0 price_gbp.
    bad_price = manifest["price_gbp"].isna() | (manifest["price_gbp"] <= 0)
    stats["dropped_bad_price"] = int(bad_price.sum())
    manifest = manifest[~bad_price]

    # Filter: year out of [MIN_YEAR, MAX_YEAR].
    bad_year = manifest["year"].isna() | (manifest["year"] < MIN_YEAR) | (manifest["year"] > MAX_YEAR)
    stats["dropped_bad_year"] = int(bad_year.sum())
    manifest = manifest[~bad_year]

    # Filter: missing advert_year.
    bad_advert_year = manifest["advert_year"].isna()
    stats["dropped_missing_advert_year"] = int(bad_advert_year.sum())
    manifest = manifest[~bad_advert_year]

    manifest["year"] = manifest["year"].astype(int)
    manifest["advert_year"] = manifest["advert_year"].astype(int)
    manifest["price_gbp"] = manifest["price_gbp"].astype(float)

    stats["n_after_filters"] = len(manifest)
    stats["n_before_join"] = n_before_join

    return manifest.reset_index(drop=True), stats


def print_summary(manifest: pd.DataFrame, stats: dict[str, int], n_discovered: int, n_skipped_basename: int) -> None:
    print("\n" + "=" * 70)
    print("BUILD MANIFEST SUMMARY")
    print("=" * 70)
    print(f"Images discovered on disk:              {n_discovered}")
    print(f"Skipped (unparseable basename):         {n_skipped_basename}")
    print(f"Parsed images going into join:          {stats.get('n_before_join', 'n/a')}")
    print(f"Dropped (no Ad_table match):             {stats.get('dropped_no_ad_table_match', 'n/a')}")
    print(f"Rows before value filters:               {stats.get('n_before_filters', 'n/a')}")
    print(f"Dropped (missing/<=0 price_gbp):         {stats.get('dropped_bad_price', 'n/a')}")
    print(f"Dropped (year outside [{MIN_YEAR}, {MAX_YEAR}]):    {stats.get('dropped_bad_year', 'n/a')}")
    print(f"Dropped (missing advert_year):           {stats.get('dropped_missing_advert_year', 'n/a')}")
    print(f"Rows after filters (before --limit):     {stats.get('n_after_filters', len(manifest))}")
    print(f"Final manifest rows (written to CSV):    {len(manifest)}")
    print("-" * 70)
    if len(manifest) > 0:
        print("describe() for year and price_gbp:")
        print(manifest[["year", "price_gbp"]].describe())
        print("-" * 70)
        print("Top 10 brands by row count:")
        print(manifest["brand"].value_counts().head(10))
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the unified DVM-CAR manifest CSV from real data.")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="DVM-CAR data root, containing tables/ and images/ subdirectories.",
    )
    parser.add_argument("--out", type=str, default="data/manifest.csv", help="Output path for the manifest CSV.")
    parser.add_argument(
        "--source",
        type=str,
        default="resized",
        choices=["resized", "fronts", "both"],
        help="Which image tree(s) to scan: resized_DVM, confirmed_fronts, or both.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, randomly subsample the final manifest to at most N rows (seed=42), for quick tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_path = Path(args.out)

    print(f"[build_manifest] Loading Ad_table.csv from {data_root / 'tables' / 'Ad_table.csv'} ...")
    ad_df = load_ad_table(data_root)
    print(f"[build_manifest] Loaded Ad_table with {len(ad_df)} rows.")

    image_df = load_image_table(data_root)

    image_paths = discover_images(data_root, args.source)
    n_discovered = len(image_paths)

    parsed_images, n_skipped_basename = parse_image_rows(image_paths)
    print(
        f"[build_manifest] Parsed {len(parsed_images)} image basenames "
        f"({n_skipped_basename} skipped for malformed basenames)."
    )

    manifest, stats = build_manifest(ad_df, image_df, parsed_images)

    if args.limit is not None and len(manifest) > args.limit:
        manifest = manifest.sample(n=args.limit, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"[build_manifest] --limit applied: subsampled to {len(manifest)} rows (seed={RANDOM_SEED}).")

    print_summary(manifest, stats, n_discovered, n_skipped_basename)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest[MANIFEST_COLUMNS].to_csv(out_path, index=False)
    print(f"[build_manifest] Wrote {len(manifest)} rows to {out_path}")


if __name__ == "__main__":
    main()
