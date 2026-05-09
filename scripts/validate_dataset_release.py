#!/usr/bin/env python3
"""Validate an EG-PCS public dataset release folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "dataset",
    "image_l",
    "image_r",
    "score",
    "has_eyetracker",
    "image_l_relpath",
    "image_r_relpath",
    "gaze_l_relpath",
    "gaze_r_relpath",
}


def nonempty(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(str(value).strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--max-npy-checks", type=int, default=0, help="0 means check all gaze npy files.")
    args = parser.parse_args()

    root = args.dataset_root
    comparisons_path = root / "comparisons" / "comparisons.csv"
    if not comparisons_path.exists():
        raise FileNotFoundError(f"Missing {comparisons_path}")

    df = pd.read_csv(comparisons_path)
    missing_cols = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    bad_scores = sorted(set(df["score"].dropna().astype(int)).difference({-1, 0, 1}))
    if bad_scores:
        raise ValueError(f"Unexpected score values: {bad_scores}")

    missing_images = []
    for col in ("image_l_relpath", "image_r_relpath"):
        missing_images.extend(str(p) for p in df[col] if nonempty(p) and not (root / str(p)).exists())

    missing_gaze = []
    gaze_paths = []
    for col in ("gaze_l_relpath", "gaze_r_relpath"):
        for p in df[col]:
            if nonempty(p):
                path = root / str(p)
                gaze_paths.append(path)
                if not path.exists():
                    missing_gaze.append(str(p))

    unique_gaze_paths = sorted(set(gaze_paths))
    npy_to_check = unique_gaze_paths
    if args.max_npy_checks > 0:
        npy_to_check = unique_gaze_paths[: args.max_npy_checks]

    bad_npy = []
    shapes = {}
    for path in npy_to_check:
        if not path.exists():
            continue
        try:
            arr = np.load(path)
            shapes[str(arr.shape)] = shapes.get(str(arr.shape), 0) + 1
            if arr.ndim not in (2, 3):
                bad_npy.append(f"{path.relative_to(root)} shape={arr.shape}")
        except Exception as exc:
            bad_npy.append(f"{path.relative_to(root)} error={exc}")

    print(f"Rows: {len(df):,}")
    print(f"Datasets: {df['dataset'].value_counts().to_dict()}")
    print(f"Scores: {df['score'].value_counts(dropna=False).to_dict()}")
    print(f"Gaze rows: {int(df['has_eyetracker'].fillna(False).astype(bool).sum()):,}")
    print(f"Missing image references: {len(missing_images):,}")
    print(f"Missing gaze references: {len(missing_gaze):,}")
    print(f"Unique gaze maps referenced: {len(unique_gaze_paths):,}")
    print(f"Checked gaze map shapes: {shapes}")

    if missing_images:
        raise FileNotFoundError(f"Missing image files, first examples: {missing_images[:10]}")
    if missing_gaze:
        raise FileNotFoundError(f"Missing gaze files, first examples: {missing_gaze[:10]}")
    if bad_npy:
        raise ValueError(f"Invalid gaze npy files, first examples: {bad_npy[:10]}")

    print("Dataset release validation passed.")


if __name__ == "__main__":
    main()

