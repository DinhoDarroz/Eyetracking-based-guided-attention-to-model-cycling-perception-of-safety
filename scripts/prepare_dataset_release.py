#!/usr/bin/env python3
"""Prepare a public EG-PCS dataset release folder.

The script exports the comparison pickle to CSV/Parquet, normalizes public
relative paths, optionally copies image/gaze assets, optionally exports split
files, and writes a checksum manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = ("dataset", "image_l", "image_r", "score")
OPTIONAL_COLUMNS = (
    "has_eyetracker",
    "survey_id",
    "trial_id",
    "npy_file_l",
    "npy_file_r",
)


def read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_pickle(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def ensure_jpg(value: object) -> str:
    name = str(value)
    return name if name.lower().endswith(".jpg") else f"{name}.jpg"


def boolish(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def basename_or_empty(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    value_s = str(value).strip()
    if not value_s:
        return ""
    return Path(value_s).name


def source_gaze_path(gaze_root: Path, value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    value_s = str(value).strip()
    if not value_s:
        return None
    path = Path(value_s)
    return path if path.is_absolute() else gaze_root / path


def copy_one(src: Path, dst: Path, *, hardlink: bool = False) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    if hardlink:
        try:
            os.link(src, dst)
            return True
        except OSError:
            pass
    shutil.copy2(src, dst)
    return True


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(root: Path) -> None:
    out = root / "checksums_sha256.txt"
    with out.open("w", encoding="utf-8") as f:
        for path in iter_files(root):
            if path == out:
                continue
            rel = path.relative_to(root).as_posix()
            f.write(f"{sha256_file(path)}  {rel}\n")


def normalize_comparisons(
    df: pd.DataFrame,
    gaze_subdir: str,
    gaze_root: Path | None = None,
) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required comparison columns: {missing}")

    keep_cols = [c for c in [*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS] if c in df.columns]
    out = df[keep_cols].copy()

    out["dataset"] = out["dataset"].astype(str)
    out["image_l"] = out["image_l"].map(ensure_jpg)
    out["image_r"] = out["image_r"].map(ensure_jpg)
    out["score"] = out["score"].astype(int)

    if "score_classification" not in out.columns:
        out["score_classification"] = out["score"] + 1

    if "has_eyetracker" in out.columns:
        out["has_eyetracker"] = out["has_eyetracker"].map(boolish)
    else:
        out["has_eyetracker"] = False
    out["has_eyetracker_source"] = out["has_eyetracker"]

    for col in ("npy_file_l", "npy_file_r"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].map(basename_or_empty)

    out["image_l_relpath"] = "images/" + out["dataset"] + "/" + out["image_l"]
    out["image_r_relpath"] = "images/" + out["dataset"] + "/" + out["image_r"]
    out["gaze_l_relpath"] = out["npy_file_l"].map(
        lambda x: f"gaze_maps/{gaze_subdir}/{x}" if x else ""
    )
    out["gaze_r_relpath"] = out["npy_file_r"].map(
        lambda x: f"gaze_maps/{gaze_subdir}/{x}" if x else ""
    )

    if gaze_root is not None:
        gaze_l_exists = out["npy_file_l"].map(lambda x: bool(x) and (gaze_root / str(x)).exists())
        gaze_r_exists = out["npy_file_r"].map(lambda x: bool(x) and (gaze_root / str(x)).exists())
        out.loc[~gaze_l_exists, ["npy_file_l", "gaze_l_relpath"]] = ""
        out.loc[~gaze_r_exists, ["npy_file_r", "gaze_r_relpath"]] = ""
        out["has_eyetracker"] = out["has_eyetracker_source"] & gaze_l_exists & gaze_r_exists

    return out


def export_splits(
    splits_dir: Path,
    output_dir: Path,
    gaze_subdir: str,
    gaze_root: Path | None = None,
) -> None:
    if not splits_dir.exists():
        return
    out_dir = output_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(splits_dir.glob("*.pkl")) + sorted(splits_dir.glob("*.pickle")):
        try:
            split_df = normalize_comparisons(read_table(path), gaze_subdir, gaze_root)
        except Exception as exc:
            print(f"[WARN] Skipping split {path}: {exc}")
            continue
        split_df.to_csv(out_dir / f"{path.stem}.csv", index=False)


def write_release_readme(output_dir: Path) -> None:
    text = """# EG-PCS Dataset

This release contains pairwise perceived cycling safety comparisons, street-view
images, and fixation-based gaze maps for the EG-PCS project.

## Contents

- `comparisons/comparisons.csv`: canonical comparison table.
- `comparisons/comparisons.parquet`: columnar copy of the comparison table when Parquet support is installed.
- `comparisons/comparisons_df.pickle`: compatibility copy of the original table.
- `images/`: image files organized by `dataset`/city.
- `gaze_maps/`: NumPy `.npy` fixation-based gaze maps.
- `data_dictionary.csv`: field-level documentation.
- `checksums_sha256.txt`: SHA-256 checksums for integrity verification.

## Labels

The `score` column uses `-1` for left image preferred, `0` for tie, and `+1`
for right image preferred.

## Gaze Maps

Gaze maps were generated with
`survey_eye_tracker/build_fixation_based_attention_maps_ogama_like.py` in the
EG-PCS code repository. They are stored as `.npy` arrays and referenced from the
comparison table through `npy_file_l`, `npy_file_r`, `gaze_l_relpath`, and
`gaze_r_relpath`.

## Citation

Please cite both the EG-PCS paper and this dataset DOI.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparisons", type=Path, default=Path("comparisons_df.pickle"))
    parser.add_argument("--images-root", type=Path, default=Path("images"))
    parser.add_argument("--gaze-root", type=Path, default=Path("survey_eye_tracker/Eyetracker_attention_maps/864x508"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--include-splits", action="store_true", help="Include split CSVs in the public release.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gaze-subdir", default="864x508")
    parser.add_argument("--copy-assets", action="store_true", help="Copy images and gaze maps into the release folder.")
    parser.add_argument("--hardlink-assets", action="store_true", help="Hardlink assets when possible, falling back to copy.")
    parser.add_argument("--write-checksums", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparisons").mkdir(exist_ok=True)

    raw_df = read_table(args.comparisons)
    df = normalize_comparisons(raw_df, args.gaze_subdir, args.gaze_root)

    df.to_csv(output_dir / "comparisons" / "comparisons.csv", index=False)
    try:
        df.to_parquet(output_dir / "comparisons" / "comparisons.parquet", index=False)
    except Exception as exc:
        print(f"[WARN] Parquet export skipped: {exc}")
    shutil.copy2(args.comparisons, output_dir / "comparisons" / "comparisons_df.pickle")

    for support_file in ("data_dictionary.csv", "CITATION.cff", "DATASET_CARD.md", "DATA_LICENSE.txt"):
        src = Path(support_file)
        if src.exists():
            shutil.copy2(src, output_dir / support_file)
    scripts_out = output_dir / "scripts"
    scripts_out.mkdir(exist_ok=True)
    for script_name in ("load_dataset.py", "validate_dataset_release.py"):
        src = Path("scripts") / script_name
        if src.exists():
            shutil.copy2(src, scripts_out / script_name)
    write_release_readme(output_dir)
    if args.include_splits:
        export_splits(args.splits_dir, output_dir, args.gaze_subdir, args.gaze_root)

    if args.copy_assets:
        missing_images = 0
        missing_gaze = 0
        for _, row in df.iterrows():
            for side in ("l", "r"):
                img_src = args.images_root / str(row["dataset"]) / str(row[f"image_{side}"])
                img_dst = output_dir / str(row[f"image_{side}_relpath"])
                if not copy_one(img_src, img_dst, hardlink=args.hardlink_assets):
                    missing_images += 1

                gaze_name = row.get(f"npy_file_{side}", "")
                if gaze_name:
                    gaze_src = source_gaze_path(args.gaze_root, gaze_name)
                    gaze_dst = output_dir / str(row[f"gaze_{side}_relpath"])
                    if gaze_src is None or not copy_one(gaze_src, gaze_dst, hardlink=args.hardlink_assets):
                        missing_gaze += 1
        print(f"[prepare] Missing image references: {missing_images}")
        print(f"[prepare] Missing gaze references: {missing_gaze}")

    if args.write_checksums:
        write_checksums(output_dir)

    print(f"[prepare] Wrote release folder: {output_dir}")
    print(f"[prepare] Rows: {len(df):,}")
    print(f"[prepare] Gaze rows: {int(df['has_eyetracker'].sum()):,}")


if __name__ == "__main__":
    main()
