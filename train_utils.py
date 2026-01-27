"""
Utility helpers for train.py.

This file intentionally contains:
- reporting / summarization logic
- lightweight helpers shared by train.py
- NO training loops
- NO dataset loading
- NO imports from train.py (to avoid circular deps)

train.py may import from here, never the opposite.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch import nn
import timm

import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import timm

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ArgsCheckReport:
    """Report returned by validate_and_normalize_args()."""
    warnings: List[str]
    errors: List[str]


def _warn(warnings: List[str], msg: str) -> None:
    warnings.append(msg)


def _err(errors: List[str], msg: str) -> None:
    errors.append(msg)

# =============================================================================================== #
# Args dependency test
# =============================================================================================== #

def validate_and_normalize_args(args, strict: bool = False, verbose: bool = True) -> ArgsCheckReport:
    """
    Validate and normalize run arguments.

    Goals:
      1) Normalize dependent defaults (e.g., ranking_margin_ties).
      2) Warn about arguments that will be ignored due to other settings.
      3) Catch clearly invalid combinations early (optionally strict).

    Args:
        args: argparse Namespace
        strict: if True -> raise ValueError on any detected error
        verbose: if True -> print warnings/errors

    Returns:
        ArgsCheckReport(warnings, errors)
    """
    warnings: List[str] = []
    errors: List[str] = []

    # ------------------------------------------------------------------
    # Basic numeric sanity
    # ------------------------------------------------------------------
    if getattr(args, "base_lr", 0.0) <= 0:
        _err(errors, f"--base_lr must be > 0 (got {getattr(args, 'base_lr', None)})")

    if getattr(args, "weight_decay", 0.0) < 0:
        _err(errors, f"--weight_decay must be >= 0 (got {getattr(args, 'weight_decay', None)})")

    if getattr(args, "backbone_lr_scale", 0.1) <= 0:
        _err(errors, f"--backbone_lr_scale must be > 0 (got {getattr(args, 'backbone_lr_scale', None)})")

    if getattr(args, "k", 1) < 1:
        _err(errors, f"--k (grad accumulation) must be >= 1 (got {getattr(args, 'k', None)})")

    if getattr(args, "grad_clip", 0.0) < 0:
        _err(errors, f"--grad_clip must be >= 0 (got {getattr(args, 'grad_clip', None)})")

    if getattr(args, "max_epochs", 1) < 1:
        _err(errors, f"--max_epochs must be >= 1 (got {getattr(args, 'max_epochs', None)})")

    # ------------------------------------------------------------------
    # Ties margin default 
    # ------------------------------------------------------------------
    if getattr(args, "ranking_margin_ties", None) is None:
        args.ranking_margin_ties = args.ranking_margin

    # If ties are OFF, ties margin + ties loss weight are irrelevant
    if not getattr(args, "ties", False):
        if getattr(args, "ties_w", 0.0) not in (0.0, 0):
            _warn(warnings, "--ties is OFF, so --ties_w will be ignored.")
        if getattr(args, "ranking_margin_ties", None) is not None:
            # It's harmless, but signal it.
            _warn(warnings, "--ties is OFF, so --ranking_margin_ties will be ignored.")

    # If ties are ON, make sure ties margin is sensible
    if getattr(args, "ties", False) and getattr(args, "ranking_margin_ties", 0.0) < 0:
        _err(errors, f"--ranking_margin_ties must be >= 0 when ties are enabled (got {args.ranking_margin_ties}).")

    # ------------------------------------------------------------------
    # Scheduler sanity checks 
    # ------------------------------------------------------------------
    scheduler = getattr(args, "scheduler", "warmup_cosine")

    if scheduler == "none":
        if getattr(args, "warmup_frac", 0.0) != 0.0:
            _warn(warnings, "[INFO] --scheduler none: ignoring --warmup_frac (no warmup used).")
        if getattr(args, "eta_min", 1e-6) != 1e-6:
            _warn(warnings, "[INFO] --scheduler none: ignoring --eta_min (no cosine used).")

    if scheduler not in ["warmup_cosine", "onecycle"]:
        if getattr(args, "warmup_frac", 0.0) != 0.0:
            _warn(
                warnings,
                "[INFO] --warmup_frac is only used by warmup_cosine/onecycle; "
                f"it will be ignored for scheduler={scheduler}."
            )

    if scheduler not in ["warmup_cosine", "cosine", "warm_restarts"]:
        if getattr(args, "eta_min", 1e-6) != 1e-6:
            _warn(
                warnings,
                "[INFO] --eta_min is only used by warmup_cosine/cosine/warm_restarts; "
                f"it will be ignored for scheduler={scheduler}."
            )

    if scheduler != "warm_restarts":
        if getattr(args, "T_0", 10) != 10 or getattr(args, "T_mult", 2) != 2:
            _warn(
                warnings,
                "[INFO] T_0/T_mult are only used by warm_restarts; "
                f"they will be ignored for scheduler={scheduler}."
            )

    # Validate scheduler-specific value ranges
    warmup_frac = float(getattr(args, "warmup_frac", 0.0))
    if warmup_frac < 0.0 or warmup_frac > 1.0:
        _err(errors, f"--warmup_frac must be in [0,1] (got {warmup_frac}).")

    if scheduler == "warm_restarts":
        if getattr(args, "T_0", 1) < 1:
            _err(errors, f"--T_0 must be >= 1 for warm_restarts (got {getattr(args, 'T_0', None)}).")
        if getattr(args, "T_mult", 1) < 1:
            _err(errors, f"--T_mult must be >= 1 for warm_restarts (got {getattr(args, 'T_mult', None)}).")

    if scheduler in ["warmup_cosine", "cosine", "warm_restarts"]:
        if getattr(args, "eta_min", 0.0) < 0:
            _err(errors, f"--eta_min must be >= 0 (got {getattr(args, 'eta_min', None)}).")

    # ------------------------------------------------------------------
    # Model-type dependencies (important for “ignored args” correctness)
    # ------------------------------------------------------------------
    model = getattr(args, "model", "rcnn")

    # Classification-only model ignores ranking-related knobs
    if model == "sscnn":
        if getattr(args, "rank_w", 0.0) not in (0.0, 0):
            _warn(warnings, "--model sscnn: --rank_w is ignored.")
        if getattr(args, "ties_w", 0.0) not in (0.0, 0):
            _warn(warnings, "--model sscnn: --ties_w is ignored.")
        if getattr(args, "ranking_margin", 0.0) != 0.3:
            _warn(warnings, "--model sscnn: --ranking_margin is ignored.")
        if getattr(args, "attn_w", 0.0) not in (0.0, 0) and getattr(args, "gaze", "off") != "off":
            _warn(warnings, "--model sscnn: gaze alignment loss is not applicable; --attn_w will be ignored.")

    # Ranking-only model ignores classification knobs
    if model == "rcnn":
        if getattr(args, "use_class_weights", False):
            _warn(warnings, "--model rcnn: --use_class_weights is ignored (no CE loss).")
        if float(getattr(args, "label_smoothing", 0.0)) > 0:
            _warn(warnings, "--model rcnn: --label_smoothing is ignored (no CE loss).")

    # ------------------------------------------------------------------
    # Gaze dependencies
    # ------------------------------------------------------------------
    gaze_mode = getattr(args, "gaze", "use")
    attn_w = float(getattr(args, "attn_w", 0.0) or 0.0)

    if gaze_mode == "off":
        if attn_w != 0.0:
            _warn(warnings, "--gaze off: gaze alignment is disabled; setting --attn_w to 0.")
            args.attn_w = 0.0
    else:
        # gaze is on/use/only
        if attn_w < 0:
            _err(errors, f"--attn_w must be >= 0 (got {attn_w}).")

        # In the code, gaze alignment only makes sense if the model returns attn maps.
        # That is true for rcnn/rsscnn when return_attn is enabled.
        if model not in ("rcnn", "rsscnn") and attn_w > 0:
            _warn(warnings, f"--gaze {gaze_mode} with --attn_w>0 but model={model}; gaze KL is not used.")

    # ------------------------------------------------------------------
    # Finetuning dependencies
    # ------------------------------------------------------------------
    if not getattr(args, "finetune", False):
        # num_ft_blocks won’t matter if backbone is frozen
        if getattr(args, "num_ft_blocks", 1) != 1:
            _warn(warnings, "--finetune is OFF: --num_ft_blocks is ignored.")
    else:
        # Finetune is ON
        n_blocks = getattr(args, "num_ft_blocks", 1)
        
        if n_blocks == 0:
            _warn(warnings, "[WARNING] --finetune is ON but --num_ft_blocks=0. The backbone will remain FROZEN (only head trains).")
        elif n_blocks < 0:
            _err(errors, f"--num_ft_blocks must be >= 0 (got {n_blocks}).")

    # ------------------------------------------------------------------
    # Pooling dependencies (New)
    # ------------------------------------------------------------------
    pooling = getattr(args, "pooling", "cls")
    if pooling == "topk":
        if getattr(args, "pool_k", 1) < 1:
            _err(errors, f"--pool_k must be >= 1 (got {getattr(args, 'pool_k', None)}).")
            
    # ------------------------------------------------------------------
    # Emit + optionally fail
    # ------------------------------------------------------------------
    if verbose:
        for m in warnings:
            print(m)
        for e in errors:
            print("[ERROR]", e)

    if strict and errors:
        raise ValueError("Argument validation failed:\n" + "\n".join(errors))

    return ArgsCheckReport(warnings=warnings, errors=errors)
    
# =============================================================================================== #
# Backbone factory
# =============================================================================================== #

# ========================================================================= #
# CONFIGURATION & PRESETS for Transformations
# ========================================================================= #

# Default ImageNet-style fallback (used only if timm config cannot be resolved)
DEFAULT_SPECS = {
    "input_size": (3, 224, 224),
    "crop_pct": 0.9,
    "interpolation": "bicubic",
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}


# -----------------------------------------------------------------------------------------------
# Backbone alias → timm model id mapping
# -----------------------------------------------------------------------------------------------
BACKBONE_ALIAS_TO_TIMM_ID = {
    # --- The "Power 5"  ---
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "beitv2_base_patch16_224": "beitv2_base_patch16_224.in1k_ft_in22k",
    "deit3_base_patch16_224": "deit3_base_patch16_224.fb_in22k_ft_in1k",
    "siglip_base_patch16_224": "vit_base_patch16_siglip_224",
    "vit_base_patch16_clip_224": "vit_base_patch16_clip_224.openai",

    # --- Modern High-Performance Alternates ---
    "dinov2_base": "vit_base_patch14_dinov2.lvd142m",          # ✅ NEW (no registers)
    "dinov2_reg_base": "vit_base_patch14_reg4_dinov2.lvd142m",
    "eva02_base": "eva02_base_patch14_448.mim_in22k_ft_in1k",
    "convnext_base": "convnext_base.fb_in22k_ft_in1k",

    # --- Original / Canonical ViT ---
    "vit_base_patch16_224": "vit_base_patch16_224.augreg_in21k_ft_in1k",  # ✅ NEW (original ViT-B/16)

    # --- Legacy / Standard Transformers ---
    "vit_base_dino": "vit_base_patch16_224.dino",
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "deit_base": "deit_base_patch16_224.fb_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "deit_tiny": "deit_tiny_patch16_224.fb_in1k",
    "deit_base_distilled": "deit_base_distilled_patch16_224.fb_in1k",

    # --- CNNs (Mapped for Preprocessing Specs) ---
    # Note: train.py loads these via torchvision, but data.py needs these
    # to fetch mean/std/crop info from timm.
    "alex": "alexnet",
    "vgg": "vgg19",
    "dense": "densenet121",
    "resnet": "resnet50",
}


DEFAULT_SPECS = {
    "input_size": (3, 224, 224),
    "crop_pct": 0.875,
    "interpolation": "bilinear",
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}

def resolve_backbone(backbone_alias: str, *, pretrained: bool = True, strict: bool = True):
    """
    Single source of truth:
      - alias -> timm_id
      - timm_id -> pretrained preprocessing specs
      - (optionally) instantiate model with native img_size
    Returns:
      model, specs
    """
    timm_id = BACKBONE_ALIAS_TO_TIMM_ID.get(backbone_alias, backbone_alias)

    # 1) Resolve pretrained preprocessing config without downloading weights
    try:
        dummy = timm.create_model(timm_id, pretrained=False)
        cfg = timm.data.resolve_data_config({}, model=dummy)
    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to resolve preprocessing for '{backbone_alias}' (timm_id='{timm_id}'): {e}")
        cfg = {}

    specs = {
        "alias": backbone_alias,
        "timm_id": timm_id,
        "input_size": cfg.get("input_size", DEFAULT_SPECS["input_size"]),
        "crop_pct": cfg.get("crop_pct", DEFAULT_SPECS["crop_pct"]),
        "interpolation": cfg.get("interpolation", DEFAULT_SPECS["interpolation"]),
        "mean": cfg.get("mean", DEFAULT_SPECS["mean"]),
        "std": cfg.get("std", DEFAULT_SPECS["std"]),
    }
    specs["img_size"] = int(specs["input_size"][-1])

    # 2) Build the actual model consistent with specs
    img_size = specs["img_size"]
    
    # --- FIX START: Force vanilla attention ---
    kwargs = dict(
        pretrained=pretrained, 
        num_classes=0, 
        img_size=img_size,
        exportable=True,  # <--- CRITICAL: Forces vanilla attention layers (hook-friendly)
    )
    # --- FIX END ---

    try:
        model = timm.create_model(timm_id, **kwargs)
    except TypeError:
        # If model doesn't accept specific kwargs (like img_size or exportable), drop them safely
        # Note: most modern timm models support exportable=True, but we handle the edge case.
        kwargs.pop("img_size", None)
        try:
            model = timm.create_model(timm_id, **kwargs)
        except TypeError:
            # Fallback: remove exportable if that was the issue (unlikely but safe)
            kwargs.pop("exportable", None)
            model = timm.create_model(timm_id, **kwargs)

    return model, specs

def infer_vit_grid_size(backbone_model, model_specs: dict) -> tuple[int, int]:
    # Prefer explicit in specs if present
    img_size = int(model_specs.get("img_size", model_specs["input_size"][-1]))

    patch = None

    # timm ViT / DeiT: patch_embed.patch_size exists
    pe = getattr(backbone_model, "patch_embed", None)
    if pe is not None and hasattr(pe, "patch_size"):
        p = pe.patch_size
        patch = int(p[0] if isinstance(p, (tuple, list)) else p)

    # Some models expose patch_size directly
    if patch is None and hasattr(backbone_model, "patch_size"):
        p = backbone_model.patch_size
        patch = int(p[0] if isinstance(p, (tuple, list)) else p)

    if patch is None:
        raise RuntimeError("Could not infer patch size from backbone; add patch_size to resolve_backbone specs.")

    if img_size % patch != 0:
        # Still compute, but warn-worthy: token grids could be non-integer if resize/crop differs.
        raise RuntimeError(f"img_size ({img_size}) not divisible by patch_size ({patch}).")

    g = img_size // patch
    return (g, g)

# =================================================================================================
# Class weights
# =================================================================================================

def compute_class_weights_from_df(
    labels,
    use_ties: bool,
    enable_weights: bool,
):
    """
    Compute class weights for CrossEntropyLoss.

    If enable_weights=False, returns None.
    """
    if not enable_weights:
        return None

    labels = np.asarray(labels)

    if use_ties:
        # classes: [left, tie, right] → [0,1,2]
        num_classes = 3
    else:
        # classes: [left, right] → [0,1]
        num_classes = 2

    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0  # avoid div-by-zero

    weights = counts.sum() / counts
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


# =================================================================================================
# PairAugment description helpers
# =================================================================================================

def print_transform_policy(args, train_tfms=None, eval_tfms=None) -> None:
    """
    Print a concise, behavior-accurate summary of the current transform policy.
    Aligned with data.py:
      - Deterministic eval: build_eval_transforms() / ResizeCenterCropAlignGaze
      - Training aug: Augmentation (swap/hflip/rotation + RandomResizedCrop-style crop + photometric + erase)
    """

    def _as_str(x):
        return "ON" if bool(x) else "OFF"

    def _paired_label(flag):
        return "Paired" if bool(flag) else "Unpaired"

    def _fmt(x):
        if x is None:
            return "None"
        if isinstance(x, (tuple, list)):
            return "(" + ", ".join(str(v) for v in x) + ")"
        return str(x)

    tm = getattr(args, "transforms_meta", None)
    specs, eval_meta, train_meta, gaze_meta = {}, {}, {}, {}

    if isinstance(tm, dict):
        if isinstance(tm.get("model_specs"), dict):
            specs = tm.get("model_specs", {})
        if isinstance(tm.get("eval"), dict):
            eval_meta = tm.get("eval", {})
        if isinstance(tm.get("train"), dict):
            train_meta = tm.get("train", {})
        if isinstance(tm.get("gaze"), dict):
            gaze_meta = tm.get("gaze", {})

    train_class = train_tfms.__class__.__name__ if train_tfms is not None else str(tm.get("train_transform_class", "unknown"))
    eval_class = eval_tfms.__class__.__name__ if eval_tfms is not None else str(tm.get("eval_transform_class", "unknown"))

    print("\n================ TRANSFORM POLICY ================")

    if isinstance(tm, dict):
        bb = tm.get("backbone", getattr(args, "backbone", "unknown"))
        fam = tm.get("backbone_family", "unknown")
        print(f"  Backbone        : {bb} ({fam})")

    if specs:
        if "input_size" in specs:
            print(f"  Input Size      : {specs['input_size']}")
        if "crop_pct" in specs:
            print(f"  Crop %          : {specs['crop_pct']}")
        if "interpolation" in specs:
            print(f"  Interpolation   : {specs['interpolation']}")
        if "mean" in specs and "std" in specs:
            print(f"  Mean/Std        : mean={_fmt(specs['mean'])}, std={_fmt(specs['std'])}")

    if eval_meta:
        if "resize_dim" in eval_meta:
            print(f"  Eval Resize     : short_side={eval_meta['resize_dim']}")
        if "target_crop" in eval_meta:
            print(f"  Eval Crop       : center_crop={eval_meta['target_crop']}")
        if "eval_policy" in eval_meta:
            print(f"  Eval Pipeline   : {eval_meta['eval_policy']}")
    print(f"  Eval Tfms Class : {eval_class}")

    gaze_requested = gaze_meta.get("requested", str(getattr(args, "gaze", "off")).lower().strip() != "off")
    gaze_grid = gaze_meta.get("grid_size", getattr(args, "gaze_grid_size", None))
    gaze_loss = gaze_meta.get("use_gaze_loss", None)

    if gaze_grid is not None:
        print(f"  Gaze Enabled    : {_as_str(gaze_requested)} (grid={tuple(gaze_grid)})")
    else:
        print(f"  Gaze Enabled    : {_as_str(gaze_requested)}")
    if gaze_loss is not None:
        print(f"  Gaze Loss       : {_as_str(gaze_loss)}")

    print("\n================ AUGMENTATION PLAN ================")

    augment_level = train_meta.get("augment", getattr(args, "augment", "none"))
    if isinstance(augment_level, bool):
        augment_level = "heavy" if augment_level else "none"
    augment_level = str(augment_level).lower().strip()
    if augment_level not in ("none", "light", "heavy"):
        augment_level = "none"

    train_policy = train_meta.get("train_policy", "")
    if train_policy:
        print(f"Train policy      : {train_policy}")

    forced_reason = train_meta.get("forced_deterministic_reason", None)
    if forced_reason:
        print(f"Deterministic     : forced ({forced_reason})")

    print(f"Train Tfms Class  : {train_class}")

    is_aug = (train_tfms is not None and train_tfms.__class__.__name__ == "Augmentation")

    # Deterministic path (train_tfms == eval_tfms), used when augment=none or gaze is enabled
    if not is_aug:
        if augment_level != "none" and not gaze_requested:
            print("\n[WARNING]")
            print(f"  args.augment={augment_level} but training transform is not Augmentation (class={train_class}).")

        print("\n[Deterministic preprocessing]")
        if eval_meta:
            resize_dim = eval_meta.get("resize_dim", "unknown")
            target_crop = eval_meta.get("target_crop", "unknown")
            print(f"  - Resize(short side)->{resize_dim} (aspect preserved)")
            print(f"  - CenterCrop->{target_crop}")
            print("  - ToTensor -> Normalize")
            if gaze_requested:
                print("  - Gaze: resize to match image geometry, center-crop, then downsample to grid")
        else:
            print("  - Resize(short side) -> CenterCrop -> ToTensor -> Normalize")

        print("==================================================\n")
        return

    # Augmentation path (data.py: class Augmentation)
    pa = train_tfms
    ties_enabled = bool(getattr(args, "ties", True))

    print(f"Data augmentation : ON ({augment_level})")
    print("Augmentation type : Pairwise ranking augmentation (L/R views)")

    print("\n[Pairwise structure]")
    print(f"  - Swap left/right        : p={getattr(pa, 'swap_p', 0.0):g} (paired; label adjusted)")
    if ties_enabled:
        print("    • ties enabled         : tie label preserved on swap")
    else:
        print("    • ties disabled        : binary label inverted on swap")
    print(f"  - Horizontal flip        : p={getattr(pa, 'hflip_p', 0.0):g} ({_paired_label(getattr(pa, 'paired_hflip', True))})")

    print("\n[Geometric preprocessing + crop]")
    print(f"  - Resize(short side)     : {getattr(pa, 'resize_short', 'unknown')} (always)")
    print(f"  - Ensure min side >=     : {getattr(pa, 'out_size', 'unknown')} (always)")

    rot_deg = getattr(pa, "rot_deg", 0.0)
    rot_p = getattr(pa, "rot_p", 0.0)
    if rot_deg > 0.0 and rot_p > 0.0:
        print(f"  - Rotation               : p={rot_p:g}, ±{rot_deg:g}° ({_paired_label(getattr(pa, 'paired_rotation', True))})")
    else:
        print("  - Rotation               : OFF")

    cs = getattr(pa, "crop_scale", None)
    cr = getattr(pa, "crop_ratio", None)
    if cs is not None and cr is not None:
        print("  - Crop                   : RandomResizedCrop-style (one crop per side)")
        print(f"    • area fraction        : {cs[0]:.2f}–{cs[1]:.2f} of resized image")
        print(f"    • aspect ratio (w/h)   : {cr[0]:.2f}–{cr[1]:.2f}")
        print(f"    • crop size pairing    : {_paired_label(getattr(pa, 'paired_scale', True))}")
        print(f"    • crop position pairing: {_paired_label(getattr(pa, 'paired_crop', True))}")
        print(f"    • resized to out_size  : {getattr(pa, 'out_size', 'unknown')} (always)")
    else:
        print("  - Crop                   : center crop to out_size (one crop per side)")

    print("\n[Photometric augmentation]")
    cj = getattr(pa, "color_jitter", None)
    if cj is not None:
        print(f"  - Color jitter           : {cj} ({_paired_label(getattr(pa, 'paired_color_jitter', False))})")
    else:
        print("  - Color jitter           : OFF")

    gray_p = getattr(pa, "gray_p", 0.0)
    if gray_p > 0.0:
        print(f"  - Grayscale              : p={gray_p:g} ({_paired_label(getattr(pa, 'paired_gray', False))})")
    else:
        print("  - Grayscale              : OFF")

    blur_p = getattr(pa, "blur_p", 0.0)
    if blur_p > 0.0:
        print(
            f"  - Gaussian blur          : p={blur_p:g}, k={getattr(pa, 'blur_kernel', 'unknown')}, "
            f"sigma={_fmt(getattr(pa, 'blur_sigma', None))} ({_paired_label(getattr(pa, 'paired_color_jitter', False))})"
        )
        print("    • pairing note         : blur uses paired_color_jitter flag in the current implementation")
    else:
        print("  - Gaussian blur          : OFF")

    print("\n[Tensor augmentation]")
    erase_p = getattr(pa, "erase_p", 0.0)
    if erase_p > 0.0:
        es = getattr(pa, "erase_scale", (0.0, 0.0))
        er = getattr(pa, "erase_ratio", (0.0, 0.0))
        print(f"  - Random erasing         : p={erase_p:g} ({_paired_label(getattr(pa, 'paired_erase', True))})")
        print(f"    • area fraction        : {es[0]:.2f}–{es[1]:.2f}")
        print(f"    • aspect ratio         : {er[0]:.2f}–{er[1]:.2f}")
        if gaze_requested:
            print("    • gaze handling        : erased regions are zeroed in the downsampled gaze grid")
    else:
        print("  - Random erasing         : OFF")

    print("\n[Final deterministic steps]")
    print("  - ToTensor -> Normalize (backbone mean/std)")
    if gaze_requested:
        print("  - Gaze: geometric ops mirror image; then downsample to grid")

    print("==================================================\n")


# =================================================================================================
# Run plan helpers
# =================================================================================================

def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _infer_vit_blocks(model: nn.Module) -> Optional[Tuple[int, List[int]]]:
    """
    Infer ViT block structure and which blocks are trainable.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return None

    try:
        n_blocks = len(blocks)
    except Exception:
        return None

    trainable = []
    for i, blk in enumerate(blocks):
        if any(p.requires_grad for p in blk.parameters()):
            trainable.append(i)

    return n_blocks, trainable


def _summarize_optimizer(optimizer: torch.optim.Optimizer) -> List[str]:
    lines = []
    for i, g in enumerate(optimizer.param_groups):
        lr = g.get("lr")
        init_lr = g.get("initial_lr", None)
        wd = g.get("weight_decay")
        n = len(g.get("params", []))

        if init_lr is not None:
            lines.append(f"  - group {i}: lr={lr}, init_lr={init_lr}, wd={wd}, tensors={n}")
        else:
            lines.append(f"  - group {i}: lr={lr}, wd={wd}, tensors={n}")
    return lines


def print_run_plan(
    args,
    train_df=None,
    val_df=None,
    test_df=None,
    train_loader=None,
    val_loader=None,
    train_tfms=None,
    eval_tfms=None,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    optimizer_info: Optional[dict] = None,   # <-- ADD THIS
    scheduler=None,
):
    """
    Single authoritative summary of the training run.
    Call once, after model + loaders + transforms exist.
    """

    print("\n" + "=" * 100)
    print("RUN PLAN")
    print("=" * 100)

    # ---------------------------------------------------------------------------------------------
    # Core switches
    # ---------------------------------------------------------------------------------------------
    print("\n[Task]")
    print(f"  model        : {args.model}")
    print(f"  backbone     : {args.backbone}")
    
    # --- NEW: Feature Pooling Info ---
    print(f"  pooling      : {getattr(args, 'pooling', 'cls')}")
    if getattr(args, 'pooling', 'cls') == 'topk':
        print(f"  pool_k       : {getattr(args, 'pool_k', 10)}")

    print(f"  ties         : {args.ties}")
    
    # --- NEW: Attention/Gaze Info ---
    print(f"  gaze         : {args.gaze}")
    if args.gaze != "off":
        print(f"  attn_mode    : {getattr(args, 'attention_mode', 'last')}")
        if getattr(args, 'attention_mode', 'last') == 'topk':
            print(f"  attn_topk    : {getattr(args, 'attn_topk', 'all')}")

    print(f"  augment      : {args.augment}")
    finetune_on = bool(getattr(args, "finetune", False)) and int(getattr(args, "num_ft_blocks", 0)) > 0
    
    print(f"  finetune     : {finetune_on}")
    if finetune_on:
        print(f"  num_ft_blocks: {args.num_ft_blocks}")


    # ---------------------------------------------------------------------------------------------
    # Batching / throughput
    # ---------------------------------------------------------------------------------------------
    print("\n[Batching]")
    bs = getattr(args, "batch_size", None)
    k = max(1, int(getattr(args, "k", 1)))
    
    # Detect DataParallel wrapper
    num_gpus = 1
    if model is not None and model.__class__.__name__ == "DataParallel":
        try:
            num_gpus = len(getattr(model, "device_ids", []) or []) or 1
        except Exception:
            num_gpus = 1
    
    print(f"  batch_size   : {bs}")
    print(f"  grad accum   : k={k}")
    print(f"  num_gpus     : {num_gpus}")
    if bs is not None:
        print(f"  effective_bs : {bs * k * num_gpus}")
    if train_loader is not None:
        print(f"  batches/epoch: {len(train_loader)}")

    # ---------------------------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------------------------
    """
    print("\n[Data]")
    if train_df is not None:
        print(f"  train rows   : {len(train_df):,}")
    if val_df is not None:
        print(f"  val rows     : {len(val_df):,}")
    if test_df is not None:
        print(f"  test rows    : {len(test_df):,}")
    """
    # ---------------------------------------------------------------------------------------------
    # Transforms
    # ---------------------------------------------------------------------------------------------
    print("\n[Transforms]")
    
    # All transform details (including backbone specs if available) are printed here.
    print_transform_policy(args, train_tfms=train_tfms, eval_tfms=eval_tfms)


    # ---------------------------------------------------------------------------------------------
    # Loss recipe
    # ---------------------------------------------------------------------------------------------
    print("\n[Loss]")
    parts = []

    if args.model in ("sscnn", "rsscnn"):
        ce = "CE"
        if args.use_class_weights:
            ce += "(weighted)"
        if args.label_smoothing > 0:
            ce += f"(ls={args.label_smoothing:g})"
        parts.append(ce)

    if args.rank_w > 0:
        parts.append(f"{args.rank_w:g}·rank")

    if args.ties and args.ties_w > 0:
        parts.append(f"{args.ties_w:g}·ties")

    if args.gaze != "off" and args.attn_w > 0:
        parts.append(f"{args.attn_w:g}·KL(gaze↔attn)")

    print("  objective   :", " + ".join(parts))

    # ---------------------------------------------------------------------------------------------
    # Model / finetuning
    # ---------------------------------------------------------------------------------------------
    if model is not None:
        print("\n[Model]")
        total, trainable = _count_params(model)
        print(f"  parameters  : total={total:,}, trainable={trainable:,}")

        vit_info = _infer_vit_blocks(model)
        if vit_info is not None:
            n_blocks, trainable_blocks = vit_info
            print(f"  vit blocks  : {n_blocks}")
            if trainable_blocks:
                print(f"  unfrozen    : {trainable_blocks}")
            else:
                print("  unfrozen    : none (backbone frozen)")

    # ---------------------------------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------------------------------
    if optimizer is not None:
        print("\n[Optimizer]")
        print(f"  type        : {optimizer.__class__.__name__}")
    
        # Optional extra diagnostics produced by scripts/train_script.py.
        # This replaces all optimizer-construction prints.
        if optimizer_info:
            mode = optimizer_info.get("mode")
            if mode is not None:
                print(f"  mode        : {mode}")
    
            bb_scale = optimizer_info.get("backbone_scale")
            layer_decay = optimizer_info.get("layer_decay")
            partial_max_blocks = optimizer_info.get("partial_max_blocks")
            if bb_scale is not None:
                print(f"  bb_scale    : {bb_scale}")
            if layer_decay is not None:
                print(f"  layer_decay : {layer_decay}")
            if partial_max_blocks is not None:
                print(f"  partial_max_blocks: {partial_max_blocks}")
    
            n_trainable_blocks = optimizer_info.get("n_trainable_blocks")
            trainable_block_idxs = optimizer_info.get("trainable_block_idxs")
            if n_trainable_blocks is not None:
                print(f"  trainable_vit_blocks: {n_trainable_blocks}")
            if trainable_block_idxs is not None:
                show = trainable_block_idxs
                if isinstance(show, (list, tuple)) and len(show) > 20:
                    show = list(show[:20]) + ["..."]
                print(f"  vit_block_idxs: {show}")
    
            fallback = optimizer_info.get("fallback")
            if fallback:
                print("  note        : backbone blocks not found → fallback grouping (no LLRD)")
    
        for line in _summarize_optimizer(optimizer):
            print(line)

    # ---------------------------------------------------------------------------------------------
    # Scheduler semantics
    # ---------------------------------------------------------------------------------------------
    print("\n[Scheduler]")
    print(f"  type        : {args.scheduler}")

    k = max(1, int(getattr(args, "k", 1)))
    if train_loader is not None and args.max_epochs > 0:
        batches = len(train_loader)
        opt_steps_epoch = math.ceil(batches / k)
        total_steps = opt_steps_epoch * args.max_epochs

        #print(f"  grad accum  : k={k}")
        print(f"  opt steps  : {opt_steps_epoch}/epoch → {total_steps} total")

        if args.scheduler in ("warmup_cosine", "onecycle"):
            warmup_steps = int(total_steps * args.warmup_frac)
            print(f"  warmup     : {warmup_steps} steps ({args.warmup_frac:g})")

    print("=" * 100 + "\n")

def resolve_batch_size(args):
    """
    Resolve batch size based on finetuning configuration.

    Policy:
      - finetune = False        -> batch_size = 128
      - finetune = True:
          num_ft_blocks = 1     -> batch_size = 128
          num_ft_blocks = 4     -> batch_size = 64
          num_ft_blocks >= 8    -> batch_size = 32

    Explicit --batch_size always overrides this logic.
    """

    # Explicit override always wins
    if args.batch_size is not None:
        return args.batch_size

    # No finetuning → large batch
    if not args.finetune:
        return 128

    # Finetuning cases
    if args.num_ft_blocks <= 1:
        return 128
    elif args.num_ft_blocks <= 4:
        return 64
    else:
        return 32

