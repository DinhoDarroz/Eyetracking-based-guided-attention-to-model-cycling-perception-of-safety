#!/usr/bin/env python3
"""
build_fixation_based_attention_maps_ogama_like.py

Generate OGAMA-like fixation-based gaze attention maps from OGAMA's stats_fixations.txt.

Outputs (configurable):
  1) FULL screenshot overlay (PNG)         -> OGAMA-like visualization in ROI
  2) ROI gaze maps at map_res (NPY)        -> primary saved gaze maps
  3) Optional ROI gaze maps at grid_res    -> patch-grid supervision (e.g., 14x14)
  4) Optional debug overlays for map/grid  -> sanity-check visualization

OGAMA-aligned properties:
- Uses FIXATIONS (stats_fixations.txt)
- Weights by fixation duration (Length)
- Gaussian blur in SCREEN PIXEL space (before ROI cropping)
- No log scaling
- No per-image min/max normalization for saved NPYS (uses sum-to-1 normalization)

Notes:
- cv2.resize expects target size as (width, height)
- When map_res="orig", ROI native resolution is used (e.g., 864x508 for the provided ROIs)
"""

import os
import json
import argparse
import logging
from glob import glob
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd
import cv2

# =====================================================================================
# DEFAULTS
# =====================================================================================

DEFAULT_OUT_DIR = "/home/csantiago/survey_eye_tracker/Eyetracker_attention_maps"
MAX_COMPARISONS_DEFAULT = 65

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# List of user session folders (relative to base_dir)
USERS = [
    "cycling932844b29e6175a85d195cbee96ce34057d0b2cb0b9bb90018e0301ef2460b82/2022_10_10_12_39_43",
    "cycling132c5e4c5b0a45e274e7fb849fecd22e62edf409bd5f1b1322ddeb6d11f90d7d/2022_10_10_13_21_15",
    "cycling0a3df224a10f3472c2a9c568a927406a49b012186f0983b9e10bcd883b4d5fcd/2022_10_18_14_08_36",
    "cyclingbd1af6d2f4bda83c3d5d6dfc93817421d804a644ab12251d1033c885730217a4/2022_11_02_15_30_21",
    "cycling28b744c8c0b8b330c7f678d5b23aa2ce614a5ae8143e96173fe3cde26ec2297e/2022_11_28_09_08_48",
    "cycling145b3ad29cb766fb22e4cfba1d750db8c17470b8d07a6f01aad5918e20ccbe80/2022_11_29_09_51_39",
    "cycling5876966995d4b61ed7073ec1ea1a92e1d3bcfbb02705d2bb441819922aaa89db/2022_11_29_10_23_22",
    "cycling4eea1bbed89e15ea4b3ecbc10b941272711810e0f2648161ece9d5bcb9839dba/2022_11_29_10_56_24",
    "cycling8ffc01ebc87eb6aa9285e7688c79a4a6b63cf21a119820f13f054cd0e2fdd987/2022_11_29_13_54_31",
    "cycling7e8315cff8453c95082b56e5b4745609cfda7bddd20bbc92c8f3f88dea3fd715/2022_12_01_09_09_42",
    "cycling5e970a9dfb4a47cae2526d10a49e351fa97d26d6e24798cddf9e8ad77f6379fc/2022_12_01_09_52_46",
    "cyclingd22a19aa45e85ca027d29be0fe3b839383d8566f1997284a96d2f97b8b5b9e63/2022_12_01_10_23_11",
    "cycling684fdee4e2ba556e4e23a3f68062835cf9796cec92ffdbe9ce53171345f32e7b/2022_12_01_10_46_42",
    "cycling08ab6849b6ce9851d50c230e82c8b2ba0564ffc3836a99e1333cd536cd9b1bdd/2022_12_02_09_46_55",
    "cyclingc08377f1e6826ffd8f74f4e1515d85319a2705749e0bb560f47e2e9c5c48186d/2022_12_02_10_18_23",
    "cyclingdbdde36bbe3344b160d31a87c5d85169c36650245f3d312494627ff1464bb2e4/2022_12_02_12_53_18",
    "cycling5aa98a95dbd30e4ffd9d5f18d19cc095f093954581f2842e9daed80395793b90/2022_12_02_13_25_43",
    "cyclingad8bc642880020eb31d2c1d5d10857bc864a01936bb335fe7a30e584ab21ebf8/2022_12_02_13_53_36",
    "cycling469572b0c7fe5cc0c5f020ebae513ffeae62ec445e8b5c19154ba2e3ee1f6de4/2022_12_05_09_46_56",
    "cycling61e4dc72e3a5c92061a3b8c78ea0f11334dcab587b2abebe99315c92213be055/2022_12_05_12_39_03",
    "cycling4c845f8ebd5f514f1fdc690d2ab60d5f8beb818b464cea0fe96ca4f97a4f773e/2022_12_08_09_19_45",
    "cycling2846319a17ec7fcad28ab540e7a7b18c9432e63357a06e9d15630eeb1ae62be3/2022_12_08_13_20_21",
    "cycling14fd071ffaf930135bd748b9623a06847a49559438ae21c6cd8845252bae5462/2022_12_09_11_29_21",
]

# =====================================================================================
# CLI
# =====================================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Build OGAMA-like fixation-based gaze maps (with selectable output resolution)."
    )
    p.add_argument('--base_dir', required=True,
                   help='Root directory containing cycling... folders')
    p.add_argument('--out_dir', default=DEFAULT_OUT_DIR,
                   help='Root output directory')
    p.add_argument('--blur_sigma', type=float, default=40.0,
                   help='Gaussian sigma in SCREEN PIXELS (30–60 typical for 1920x1200)')
    p.add_argument('--max_comparisons', type=int, default=MAX_COMPARISONS_DEFAULT,
                   help='Limit comparisons per session')
    p.add_argument('--npy_only', action='store_true',
                   help='Only save .npy files (skip PNG overlays)')

    # Main output map resolution (saved as .npy). Options:
    #   - "orig"     -> use ROI native size per-side (e.g., 864x508)
    #   - "WxH"      -> custom fixed size (e.g., 224x224, 864x508)
    p.add_argument('--map_res', default="224x224",
                   help='Primary saved map resolution: "orig" or "WxH" (e.g., 864x508, 224x224)')

    # Optional patch-grid output resolution (also saved as .npy). Options:
    #   - "none"     -> disabled
    #   - "WxH"      -> e.g., 14x14, 16x16
    p.add_argument('--grid_res', default="none",
                   help='Optional patch-grid map resolution: "none" or "WxH" (e.g., 14x14)')

    return p.parse_args()

# =====================================================================================
# LOADERS
# =====================================================================================

def load_ui_params(survey_dir: str) -> Dict[str, Any]:
    with open(os.path.join(survey_dir, 'ui_params.json'), 'r') as f:
        return json.load(f)

def load_comparisons(survey_dir: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(survey_dir, 'comparisons.csv'),
        header=None,
        names=['timestamp', 'TrialID', 'left', 'right']
    )

def load_fixations(survey_dir: str) -> pd.DataFrame:
    path = os.path.join(survey_dir, 'stats_fixations.txt')
    df = pd.read_csv(path, sep='\t', comment='#')
    df = df[['TrialID', 'Length', 'PosX', 'PosY']].copy()
    df['TrialID'] = df['TrialID'].astype(int)
    df['Length'] = df['Length'].astype(float)
    df['PosX'] = df['PosX'].astype(float)
    df['PosY'] = df['PosY'].astype(float)
    return df

def load_trial_image(survey_dir: str, trial_id: int) -> Optional[np.ndarray]:
    patterns = [
        f"{trial_id}-*.png",
        f"{trial_id}-*.jpg",
        f"{trial_id}-*.jpeg",
        f"{trial_id}-*.bmp",
    ]
    for pat in patterns:
        files = glob(os.path.join(survey_dir, pat))
        if files:
            img = cv2.imread(files[0])
            if img is None:
                logging.warning(f"Failed to read screenshot {files[0]} (trial {trial_id})")
                return None
            return img

    logging.warning(f"No screenshot for trial {trial_id} in {survey_dir}; skipping trial.")
    return None


# =====================================================================================
# CORE LOGIC
# =====================================================================================

def build_fixation_map(fix_df: pd.DataFrame, trial_id: int, full_w: int, full_h: int) -> np.ndarray:
    base = np.zeros((full_h, full_w), dtype=np.float32)
    sel = fix_df[fix_df['TrialID'] == trial_id]
    for _, r in sel.iterrows():
        x = int(round(r['PosX']))
        y = int(round(r['PosY']))
        if 0 <= x < full_w and 0 <= y < full_h:
            base[y, x] += float(r['Length'])
    return base

def gaussian_blur_px(arr: np.ndarray, sigma_px: float) -> np.ndarray:
    if sigma_px <= 0:
        return arr
    return cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px)

def normalize_to_prob(arr: np.ndarray) -> np.ndarray:
    s = float(arr.sum())
    if s > 0:
        return arr / s
    return arr

def _clip_roi(roi: Any, w: int, h: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if roi is None or not isinstance(roi, (list, tuple)) or len(roi) != 2:
        return None
    (x0, y0), (x1, y1) = roi
    x0 = int(round(x0)); y0 = int(round(y0))
    x1 = int(round(x1)); y1 = int(round(y1))
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0), (x1, y1)

def _roi_wh(roi_xy: Tuple[Tuple[int, int], Tuple[int, int]]) -> Tuple[int, int]:
    (x0, y0), (x1, y1) = roi_xy
    return (x1 - x0), (y1 - y0)

def _parse_res(spec: str) -> Optional[Tuple[int, int]]:
    s = str(spec).strip().lower()
    if s in ("none", ""):
        return None
    if "x" not in s:
        raise ValueError(f"Resolution must be WxH or 'orig'/'none', got: {spec}")
    w_str, h_str = s.split("x", 1)
    w = int(w_str)
    h = int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid resolution: {spec}")
    return (w, h)

def resize_any(arr: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    th, tw = target_wh[1], target_wh[0]
    src_h, src_w = arr.shape[:2]
    if (src_w, src_h) == (tw, th):
        return arr
    if tw < src_w and th < src_h:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    return cv2.resize(arr, (tw, th), interpolation=interp)

def crop_roi(arr: np.ndarray, roi_xy: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
    (x0, y0), (x1, y1) = roi_xy
    return arr[y0:y1, x0:x1]

# =====================================================================================
# VISUALIZATION HELPERS
# =====================================================================================

def save_roi_overlay_from_map(out_path: str, roi_map: np.ndarray, roi_bgr: np.ndarray) -> None:
    if roi_map.size == 0 or roi_bgr.size == 0:
        return

    heat = roi_map.astype(np.float32).copy()
    heat -= heat.min()
    m = float(heat.max())
    if m > 0:
        heat /= m

    heat8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    cmap = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(roi_bgr, 0.6, cmap, 0.4, 0.0)
    cv2.imwrite(out_path, overlay)

def save_fullres_attention_overlay(out_path: str,
                                   full_map: np.ndarray,
                                   roi_xy: Tuple[Tuple[int, int], Tuple[int, int]],
                                   trial_img_bgr: np.ndarray) -> None:
    (x0, y0), (x1, y1) = roi_xy
    att = full_map[y0:y1, x0:x1].astype(np.float32)
    img = trial_img_bgr[y0:y1, x0:x1]
    save_roi_overlay_from_map(out_path, att, img)

# =====================================================================================
# NAMING
# =====================================================================================

def _sanitize_for_filename(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)

def make_npy_name(user_id: str, trial_id: int, img_id: str, side: str) -> str:
    return f"survey{user_id}_trial{trial_id}_{img_id}_{side}_eyetrack.npy"

def _subdir_for_res(res_wh: Tuple[int, int]) -> str:
    return f"{res_wh[0]}x{res_wh[1]}"

# =====================================================================================
# MAIN
# =====================================================================================

def main():
    args = parse_args()

    map_res_spec = str(args.map_res).strip().lower()
    grid_res_wh = _parse_res(args.grid_res)

    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    total_saved_map = 0
    total_saved_grid = 0

    for survey_num, rel_path in enumerate(USERS, start=1):
        survey_dir = os.path.join(args.base_dir, rel_path)
        if not os.path.isdir(survey_dir):
            continue

        if not os.path.exists(os.path.join(survey_dir, 'stats_fixations.txt')):
            continue

        user_id = _sanitize_for_filename(rel_path.split("/")[0])

        logging.info(f"[Session {survey_num}] user={user_id} dir={survey_dir}")

        ui = load_ui_params(survey_dir)
        comps = load_comparisons(survey_dir)
        fix_df = load_fixations(survey_dir)

        if comps.empty:
            continue
        if len(comps) > args.max_comparisons:
            comps = comps.iloc[:args.max_comparisons].reset_index(drop=True)

        for _, row in comps.iterrows():
            trial_id = int(row['TrialID'])

            # Screenshot is needed for (H,W) and for overlays; always load once per trial.
            trial_img = load_trial_image(survey_dir, trial_id)
            if trial_img is None:
                continue

            full_h, full_w = trial_img.shape[:2]

            full_map = build_fixation_map(fix_df, trial_id, full_w, full_h)
            if float(full_map.sum()) <= 0.0:
                continue

            full_map = gaussian_blur_px(full_map, args.blur_sigma)

            for side in ['left', 'right']:
                stim = os.path.basename(str(row[side]).strip())
                img_id = os.path.splitext(stim)[0]

                roi_raw = ui.get(f'image_{side}')
                roi_xy = _clip_roi(roi_raw, full_w, full_h)
                if roi_xy is None:
                    continue

                roi_w, roi_h = _roi_wh(roi_xy)

                # map_res can be "orig" (ROI native) or a fixed WxH
                if map_res_spec == "orig":
                    map_res_wh = (roi_w, roi_h)
                else:
                    map_res_wh = _parse_res(map_res_spec)
                    if map_res_wh is None:
                        raise ValueError(f"Invalid map_res: {args.map_res}")

                # Crop ROI from blurred full_map, resize, then normalize to probability
                roi_map = crop_roi(full_map, roi_xy)
                if roi_map.size == 0:
                    continue

                map_res_map = resize_any(roi_map, map_res_wh).astype(np.float32)
                map_prob = normalize_to_prob(map_res_map)
                if float(map_prob.sum()) <= 0.0:
                    continue

                # Output folders: separate by resolution to match dataset subdir conventions
                map_out_dir = os.path.join(out_root, _subdir_for_res(map_res_wh))
                os.makedirs(map_out_dir, exist_ok=True)

                name_npy = make_npy_name(user_id=user_id, trial_id=trial_id, img_id=img_id, side=side)
                np.save(os.path.join(map_out_dir, name_npy), map_prob)
                total_saved_map += 1

                # Optional patch-grid output
                if grid_res_wh is not None:
                    grid_out_dir = os.path.join(out_root, _subdir_for_res(grid_res_wh))
                    os.makedirs(grid_out_dir, exist_ok=True)

                    grid_map = resize_any(map_prob, grid_res_wh).astype(np.float32)
                    grid_prob = normalize_to_prob(grid_map)
                    if float(grid_prob.sum()) > 0.0:
                        np.save(os.path.join(grid_out_dir, name_npy), grid_prob)
                        total_saved_grid += 1

                # Optional PNG overlays
                if not args.npy_only:
                    (x0, y0), (x1, y1) = roi_xy
                    roi_bgr = trial_img[y0:y1, x0:x1]

                    # Full-res overlay (ROI section of screenshot + ROI section of blurred fixation map)
                    full_png_dir = os.path.join(out_root, "FULLRES_OVERLAYS")
                    os.makedirs(full_png_dir, exist_ok=True)

                    name_full = f"survey{user_id}_trial{trial_id}_{img_id}_{side}_FULLRES.png"
                    save_fullres_attention_overlay(
                        os.path.join(full_png_dir, name_full),
                        full_map, roi_xy, trial_img
                    )

                    # Map-res overlay (map_prob resized to ROI size for visualization)
                    map_png_dir = os.path.join(out_root, f"OVERLAY_{_subdir_for_res(map_res_wh)}")
                    os.makedirs(map_png_dir, exist_ok=True)

                    map_vis = resize_any(map_prob, (roi_w, roi_h))
                    name_map_png = name_npy.replace(".npy", ".png")
                    save_roi_overlay_from_map(os.path.join(map_png_dir, name_map_png), map_vis, roi_bgr)

                    # Grid overlay (grid_prob upsampled to ROI size)
                    if grid_res_wh is not None:
                        grid_png_dir = os.path.join(out_root, f"OVERLAY_{_subdir_for_res(grid_res_wh)}")
                        os.makedirs(grid_png_dir, exist_ok=True)

                        if 'grid_prob' in locals() and isinstance(grid_prob, np.ndarray) and grid_prob.size > 0:
                            grid_vis = resize_any(grid_prob, (roi_w, roi_h))
                            save_roi_overlay_from_map(os.path.join(grid_png_dir, name_map_png), grid_vis, roi_bgr)

        logging.info(f"[Session {survey_num}] done")

    logging.info(f"Completed. Saved {total_saved_map} map-res NPYS.")
    if grid_res_wh is not None:
        logging.info(f"Completed. Saved {total_saved_grid} grid-res NPYS.")

if __name__ == '__main__':
    main()
