#!/usr/bin/env python3
"""Minimal loader example for a released EG-PCS dataset folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    root = args.dataset_root
    df = pd.read_csv(root / "comparisons" / "comparisons.csv")
    row = df.iloc[int(args.index)]

    image_l = Image.open(root / row["image_l_relpath"]).convert("RGB")
    image_r = Image.open(root / row["image_r_relpath"]).convert("RGB")

    gaze_l = None
    gaze_r = None
    if isinstance(row.get("gaze_l_relpath"), str) and row["gaze_l_relpath"]:
        gaze_l = np.load(root / row["gaze_l_relpath"])
    if isinstance(row.get("gaze_r_relpath"), str) and row["gaze_r_relpath"]:
        gaze_r = np.load(root / row["gaze_r_relpath"])

    print("row_index:", int(args.index))
    print("score:", int(row["score"]))
    print("left_image_size:", image_l.size)
    print("right_image_size:", image_r.size)
    print("left_gaze_shape:", None if gaze_l is None else gaze_l.shape)
    print("right_gaze_shape:", None if gaze_r is None else gaze_r.shape)


if __name__ == "__main__":
    main()

