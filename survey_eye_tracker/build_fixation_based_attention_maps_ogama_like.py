#!/usr/bin/env python3
"""
build_fixation_based_attention_maps_ogama_like.py

Generate OGAMA-like fixation-based attention maps and export:
  1) FULL-RES attention overlays (PNG)  -> OGAMA-equivalent visualization
  2) 14x14 probability maps (.npy)      -> ViT supervision
  3) 14x14 upsampled overlays (PNG)     -> debug visualization

Key properties (OGAMA-aligned):
- Uses FIXATIONS (stats_fixations.txt)
- Weights by fixation duration (Length)
- Gaussian blur in PIXEL space (before cropping)
- No log scaling
- No per-image min/max normalization for data

Differences vs OGAMA (intentional):
- Final projection to 14x14
- Final normalization to sum=1
"""

import os
import json
import argparse
import logging
from glob import glob

import numpy as np
import pandas as pd
import cv2

# =====================================================================================
# CONFIG
# =====================================================================================

GLOBAL_OUT_DIR = "/home/csantiago/survey_eye_tracker/outputs/224x224"

HEATMAP_RES = (224, 224)
MAX_COMPARISONS = 65

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# List of user session folders (relative to base_dir)
USERS = [
    "cycling932844b29e6175a85d195cbee96ce34057d0b2cb0b9bb90018e0301ef2460b82/2022_10_10_12_39_43", # 1
    "cycling132c5e4c5b0a45e274e7fb849fecd22e62edf409bd5f1b1322ddeb6d11f90d7d/2022_10_10_13_21_15", # 2
    "cycling0a3df224a10f3472c2a9c568a927406a49b012186f0983b9e10bcd883b4d5fcd/2022_10_18_14_08_36", # 3
    "cyclingbd1af6d2f4bda83c3d5d6dfc93817421d804a644ab12251d1033c885730217a4/2022_11_02_15_30_21", # 5
    "cycling28b744c8c0b8b330c7f678d5b23aa2ce614a5ae8143e96173fe3cde26ec2297e/2022_11_28_09_08_48", # 6
    "cycling145b3ad29cb766fb22e4cfba1d750db8c17470b8d07a6f01aad5918e20ccbe80/2022_11_29_09_51_39", # 7
    "cycling5876966995d4b61ed7073ec1ea1a92e1d3bcfbb02705d2bb441819922aaa89db/2022_11_29_10_23_22", # 8
    "cycling4eea1bbed89e15ea4b3ecbc10b941272711810e0f2648161ece9d5bcb9839dba/2022_11_29_10_56_24", # 9
    # "cycling34d1088782c31c6a960b5e97b47c878d06e548939499f39c90ed9e4209a128c3/2022_11_29_11_27_11", # 10 /!\
    "cycling8ffc01ebc87eb6aa9285e7688c79a4a6b63cf21a119820f13f054cd0e2fdd987/2022_11_29_13_54_31", # 11
    "cycling7e8315cff8453c95082b56e5b4745609cfda7bddd20bbc92c8f3f88dea3fd715/2022_12_01_09_09_42", # 12
    "cycling5e970a9dfb4a47cae2526d10a49e351fa97d26d6e24798cddf9e8ad77f6379fc/2022_12_01_09_52_46", # 13
    "cyclingd22a19aa45e85ca027d29be0fe3b839383d8566f1997284a96d2f97b8b5b9e63/2022_12_01_10_23_11", # 14
    "cycling684fdee4e2ba556e4e23a3f68062835cf9796cec92ffdbe9ce53171345f32e7b/2022_12_01_10_46_42", # 15
    # "cycling48a65889a9ec7a72c521c4c040cd9c8604a43cf428cd7ae3150d5518d505968c/2022_12_01_14_01_05", # 18 /!\
    "cycling08ab6849b6ce9851d50c230e82c8b2ba0564ffc3836a99e1333cd536cd9b1bdd/2022_12_02_09_46_55", # 19
    "cyclingc08377f1e6826ffd8f74f4e1515d85319a2705749e0bb560f47e2e9c5c48186d/2022_12_02_10_18_23", # 20
    # "cyclingf7a6481ec3f7781563130f31e1bb429200913d03ea0dba8327f3a69132d978f1/2022_12_02_10_56_09", # 21 /!\
    "cyclingdbdde36bbe3344b160d31a87c5d85169c36650245f3d312494627ff1464bb2e4/2022_12_02_12_53_18", # 22
    "cycling5aa98a95dbd30e4ffd9d5f18d19cc095f093954581f2842e9daed80395793b90/2022_12_02_13_25_43", # 23
    "cyclingad8bc642880020eb31d2c1d5d10857bc864a01936bb335fe7a30e584ab21ebf8/2022_12_02_13_53_36", # 24
    "cycling469572b0c7fe5cc0c5f020ebae513ffeae62ec445e8b5c19154ba2e3ee1f6de4/2022_12_05_09_46_56", # 25
    "cycling61e4dc72e3a5c92061a3b8c78ea0f11334dcab587b2abebe99315c92213be055/2022_12_05_12_39_03", # 26
    "cycling4c845f8ebd5f514f1fdc690d2ab60d5f8beb818b464cea0fe96ca4f97a4f773e/2022_12_08_09_19_45", # 27
    "cycling2846319a17ec7fcad28ab540e7a7b18c9432e63357a06e9d15630eeb1ae62be3/2022_12_08_13_20_21", # 28
    "cycling14fd071ffaf930135bd748b9623a06847a49559438ae21c6cd8845252bae5462/2022_12_09_11_29_21", # 29
]


# =====================================================================================
# CLI
# =====================================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Build OGAMA-like fixation-based attention maps (ViT-ready)."
    )
    p.add_argument('--base_dir', required=True,
                   help='Root directory containing cycling... folders')
    p.add_argument('--blur_sigma', type=float, default=40.0,
                   help='Gaussian sigma in PIXELS (30–60 recommended for 1920x1200)')
    p.add_argument('--npy_only', action='store_true',
                   help='Only save .npy files (skip all PNGs)')
    return p.parse_args()

# =====================================================================================
# LOADERS
# =====================================================================================

def load_ui_params(survey_dir):
    with open(os.path.join(survey_dir, 'ui_params.json'), 'r') as f:
        return json.load(f)

def load_comparisons(survey_dir):
    return pd.read_csv(
        os.path.join(survey_dir, 'comparisons.csv'),
        header=None,
        names=['timestamp', 'TrialID', 'left', 'right']
    )

def load_fixations(survey_dir):
    """Load OGAMA GazeFixations table."""
    path = os.path.join(survey_dir, 'stats_fixations.txt')
    df = pd.read_csv(path, sep='\t', comment='#')
    df = df[['TrialID', 'Length', 'PosX', 'PosY']].copy()
    df['TrialID'] = df['TrialID'].astype(int)
    df['Length'] = df['Length'].astype(float)
    df['PosX'] = df['PosX'].astype(float)
    df['PosY'] = df['PosY'].astype(float)
    return df

def load_trial_image(survey_dir, trial_id):
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
                raise IOError(f"Failed to read {files[0]}")
            return img
    raise FileNotFoundError(f"No screenshot for trial {trial_id}")

# =====================================================================================
# CORE LOGIC
# =====================================================================================

def build_fixation_map(fix_df, trial_id, full_w, full_h):
    base = np.zeros((full_h, full_w), dtype=np.float32)
    sel = fix_df[fix_df['TrialID'] == trial_id]
    for _, r in sel.iterrows():
        x = int(round(r['PosX']))
        y = int(round(r['PosY']))
        if 0 <= x < full_w and 0 <= y < full_h:
            base[y, x] += r['Length']
    return base

def gaussian_blur_px(arr, sigma_px):
    if sigma_px <= 0:
        return arr
    return cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px)

def normalize_to_prob(arr):
    s = float(arr.sum())
    if s > 0:
        return arr / s
    return arr

def crop_and_resize(arr, roi, target_res):
    (x0, y0), (x1, y1) = roi
    crop = arr[y0:y1, x0:x1]
    return cv2.resize(crop, target_res, interpolation=cv2.INTER_AREA)

# =====================================================================================
# VISUALIZATION HELPERS
# =====================================================================================

def save_fullres_attention_overlay(out_path, full_map, roi, trial_img_bgr):
    (x0, y0), (x1, y1) = roi
    att = full_map[y0:y1, x0:x1].astype(np.float32)
    img = trial_img_bgr[y0:y1, x0:x1]
    if att.size == 0:
        return
    att -= att.min()
    if att.max() > 0:
        att /= att.max()
    heat8 = (att * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, cmap, 0.4, 0.0)
    cv2.imwrite(out_path, overlay)

def save_14x14_overlay(out_path, heat14, crop_bgr):
    h, w = crop_bgr.shape[:2]
    heat = cv2.resize(heat14, (w, h), interpolation=cv2.INTER_CUBIC)
    heat -= heat.min()
    if heat.max() > 0:
        heat /= heat.max()
    heat8 = (heat * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(crop_bgr, 0.6, cmap, 0.4, 0.0)
    cv2.imwrite(out_path, overlay)

# =====================================================================================
# MAIN
# =====================================================================================

def _sanitize_for_filename(s: str) -> str:
    # Keep alphanum, dash, underscore; replace everything else with underscore.
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)

def make_npy_name(survey_id: int, trial_id: int, img_id: str, side: str) -> str:
    """
    side: 'left' or 'right'
    -> survey{survey_id}_trial{trial_id}_{img_id}_{side}_eyetrack.npy
    """
    return f"survey{survey_id}_trial{trial_id}_{img_id}_{side}_eyetrack.npy"
    

def main():
    args = parse_args()
    os.makedirs(GLOBAL_OUT_DIR, exist_ok=True)

    total_saved = 0

    for survey_num, rel_path in enumerate(USERS, start=1):
        survey_dir = os.path.join(args.base_dir, rel_path)
        if not os.path.isdir(survey_dir):
            continue

        if not os.path.exists(os.path.join(survey_dir, 'stats_fixations.txt')):
            continue

        # NEW: user/session id from folder name (first component of rel_path)
        user_id = rel_path.split("/")[0]  # e.g., "cycling9328...."
        user_id = _sanitize_for_filename(user_id)

        logging.info(f"[Survey {survey_num}] user={user_id} dir={survey_dir}")

        ui = load_ui_params(survey_dir)
        comps = load_comparisons(survey_dir)
        fix_df = load_fixations(survey_dir)

        if comps.empty:
            continue
        if len(comps) > MAX_COMPARISONS:
            comps = comps.iloc[:MAX_COMPARISONS].reset_index(drop=True)

        trial0 = int(comps.iloc[0]['TrialID'])
        img0 = load_trial_image(survey_dir, trial0)
        full_h, full_w = img0.shape[:2]

        for _, row in comps.iterrows():
            trial_id = int(row['TrialID'])
            full_map = build_fixation_map(fix_df, trial_id, full_w, full_h)
            if full_map.sum() == 0:
                continue

            full_map = gaussian_blur_px(full_map, args.blur_sigma)

            trial_img = None
            if not args.npy_only:
                trial_img = load_trial_image(survey_dir, trial_id)

            for side in ['left', 'right']:
                stim = os.path.basename(str(row[side]).strip())
                roi = ui.get(f'image_{side}')
                if roi is None:
                    continue

                # ---------- FULL RES OVERLAY ----------
                if not args.npy_only:
                    name_full = f"user{user_id}_survey{survey_num}_trial{trial_id}_{stim}_{side}_FULLRES.png"
                    save_fullres_attention_overlay(
                        os.path.join(GLOBAL_OUT_DIR, name_full),
                        full_map, roi, trial_img
                    )

                # ---------- 14x14 MAP ----------
                crop14 = crop_and_resize(full_map, roi, HEATMAP_RES)
                crop14 = normalize_to_prob(crop14.astype(np.float32))
                if crop14.sum() == 0:
                    continue

                # survey_id must be the USER HASH ONLY (no date)
                survey_id = user_id   # e.g. cycling08ab6849b6ce9851d50c230e82c8b2ba0564ffc3836a99e1333cd536cd9b1bdd
                
                img_id = os.path.splitext(stim)[0]  # remove .jpg / .png if present
                
                name_npy = make_npy_name(
                    survey_id=survey_id,
                    trial_id=trial_id,
                    img_id=img_id,
                    side=side,
                )
                
                np.save(os.path.join(GLOBAL_OUT_DIR, name_npy), crop14)
                total_saved += 1
                
                # ---------- 14x14 OVERLAY (optional) ----------
                if not args.npy_only:
                    (x0, y0), (x1, y1) = roi
                    crop_bgr = trial_img[y0:y1, x0:x1]
                
                    name_png = name_npy.replace(".npy", ".png")
                
                    save_14x14_overlay(
                        os.path.join(GLOBAL_OUT_DIR, name_png),
                        crop14,
                        crop_bgr,
                    )
        
        logging.info(f"[Survey {survey_num}] done")

    logging.info(f"Completed. Saved {total_saved} attention maps.")


if __name__ == '__main__':
    main()
