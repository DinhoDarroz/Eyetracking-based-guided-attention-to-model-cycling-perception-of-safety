"""
Utility helpers for train.py.

This file intentionally contains:
- reporting / summarization logic
- lightweight helpers shared by train.py


train.py may import from here, never the opposite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from backbone_registry import BACKBONE_ALIAS_TO_TIMM_ID, DEFAULT_SPECS, infer_vit_grid_size, resolve_backbone
from gaze_policy import normalize_gaze_mode


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

    if hasattr(args, "train_gaze_frac"):
        train_gaze_frac = float(getattr(args, "train_gaze_frac"))
        args.train_gaze_frac = train_gaze_frac
        if train_gaze_frac < 0.0 or train_gaze_frac > 1.0:
            _err(errors, f"--train_gaze_frac must be in [0,1] (got {train_gaze_frac}).")

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
    model = getattr(args, "model", "ranking")

    # Classification-only model ignores ranking-related knobs
    if model == "classification":
        if getattr(args, "rank_w", 0.0) not in (0.0, 0):
            _warn(warnings, "--model classification: --rank_w is ignored.")
        if getattr(args, "ties_w", 0.0) not in (0.0, 0):
            _warn(warnings, "--model classification: --ties_w is ignored.")
        if getattr(args, "ranking_margin", 0.0) != 0.3:
            _warn(warnings, "--model classification: --ranking_margin is ignored.")
        if getattr(args, "attn_w", 0.0) not in (0.0, 0) and normalize_gaze_mode(getattr(args, "gaze_mode", None)) != "disable":
            _warn(warnings, "--model classification: gaze alignment loss is not applicable; --attn_w will be ignored.")

    if model == "multitask":
        if getattr(args, "attn_w", 0.0) not in (0.0, 0) and normalize_gaze_mode(getattr(args, "gaze_mode", None)) != "disable":
            _warn(warnings, "--model multitask: gaze alignment loss is not applicable; --attn_w will be ignored.")

    # Ranking-only model ignores classification knobs
    if model == "ranking":
        if getattr(args, "use_class_weights", False):
            _warn(warnings, "--model ranking: --use_class_weights is ignored (no CE loss).")
        if float(getattr(args, "label_smoothing", 0.0)) > 0:
            _warn(warnings, "--model ranking: --label_smoothing is ignored (no CE loss).")

    # ------------------------------------------------------------------
    # Gaze dependencies
    # ------------------------------------------------------------------
    gaze_mode = normalize_gaze_mode(getattr(args, "gaze_mode", None))
    args.gaze_mode = gaze_mode

    attn_w = float(getattr(args, "attn_w", 0.0) or 0.0)
    if attn_w < 0.0:
        _err(errors, f"--attn_w must be >= 0 (got {attn_w}).")

    # KL is only used by the multitask_gaze objective in this codebase.
    model = str(getattr(args, "model", "ranking")).lower().strip()
    wants_kl = gaze_mode in ("diag", "align", "align+gaze")

    if (model != "multitask_gaze") and wants_kl:
        _warn(warnings, f"--gaze_mode={gaze_mode} requests KL diagnostics/supervision, but --model={model}; KL will be disabled.")

    if gaze_mode in ("disable", "diag", "guide", "gaze_bias") and attn_w > 0.0:
        _warn(warnings, f"--attn_w={attn_w} but --gaze_mode={gaze_mode}; KL will not contribute to the objective (w_kl_eff=0).")

    if gaze_mode in ("align", "align+gaze") and attn_w == 0.0:
        _warn(warnings, f"--gaze_mode={gaze_mode} but --attn_w=0; KL supervision is effectively disabled.")


    # ------------------------------------------------------------------
    # Finetuning dependencies
    # ------------------------------------------------------------------
    n_layers = int(getattr(args, "num_ft_layers", getattr(args, "num_ft_blocks", 1)))
    if not getattr(args, "finetune", False):
        # num_ft_layers will not matter if backbone is frozen.
        if n_layers != 1:
            _warn(warnings, "--finetune is OFF: --num_ft_layers is ignored.")
    else:
        # Finetune is ON
        if n_layers == 0:
            _warn(warnings, "[WARNING] --finetune is ON but --num_ft_layers=0. The backbone will remain FROZEN (only head trains).")
        elif n_layers < 0:
            _err(errors, f"--num_ft_layers must be >= 0 (got {n_layers}).")

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
    

# Backbone and gaze policy live in backbone_registry.py and gaze_policy.py.
# They are re-exported here to keep old scripts and notebooks compatible.


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

    Transform family:
      - PairwisePreprocessing handles deterministic eval preprocessing and optional
        training augmentation (swap/hflip/rotation + crop + photometric + erase).

    Gaze printing rules (centralized):
      - Gaze mode and dependencies come from args.gaze_cfg when present
      - When gaze is disabled, no gaze output/grid/out_size lines are printed
      - Gaze out_size is printed only when gaze is enabled and gaze_output == "guide"
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

    cfg = getattr(args, "gaze_cfg", None)

    gaze_mode = str(
        gaze_meta.get(
            "mode",
            getattr(cfg, "mode", getattr(args, "gaze_mode", "disable")),
        )
    ).lower().strip()

    load_gaze = bool(gaze_meta.get("load_gaze", getattr(cfg, "load_gaze", False)))
    if ("load_gaze" not in gaze_meta) and ("requested" in gaze_meta):
        load_gaze = bool(gaze_meta.get("requested", False))

    gaze_enabled = bool(load_gaze) and (gaze_mode != "disable")

    gaze_grid = gaze_meta.get("grid_size", getattr(args, "gaze_grid_size", None))

    gaze_output = None
    gaze_out_size = None
    if gaze_enabled:
        gaze_output = gaze_meta.get("gaze_output", getattr(cfg, "gaze_output", eval_meta.get("gaze_output", "align")))
        gaze_output = str(gaze_output).lower().strip()
        if gaze_output not in ("align", "guide"):
            gaze_output = "align"
        if gaze_output == "guide":
            gaze_out_size = gaze_meta.get("out_size", eval_meta.get("target_crop", None))

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

    print(f"  Gaze Mode       : {gaze_mode}")
    if not gaze_enabled:
        print("  Gaze Enabled    : OFF")
    else:
        if gaze_grid is not None:
            print(f"  Gaze Enabled    : ON (grid={tuple(gaze_grid)}, output={gaze_output})")
        else:
            print(f"  Gaze Enabled    : ON (output={gaze_output})")
        if gaze_out_size is not None:
            print(f"  Gaze Out Size   : {int(gaze_out_size)}x{int(gaze_out_size)}")

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

    is_aug = bool(train_tfms is not None and getattr(train_tfms, "augment", False))

    if not is_aug:
        if augment_level != "none":
            print("\n[WARNING]")
            print(f"  args.augment={augment_level} but training transform has augmentation disabled (class={train_class}).")

        print("\n[Deterministic preprocessing]")
        if eval_meta:
            resize_dim = eval_meta.get("resize_dim", "unknown")
            target_crop = eval_meta.get("target_crop", "unknown")
            print(f"  - Resize(short side)->{resize_dim} (aspect preserved)")
            print(f"  - CenterCrop->{target_crop}")
            print("  - ToTensor -> Normalize")
            if gaze_enabled:
                if gaze_output == "guide":
                    print("  - Gaze: mirror geometric ops; keep at out_size")
                else:
                    print("  - Gaze: mirror geometric ops; then downsample to grid")
        else:
            print("  - Resize(short side) -> CenterCrop -> ToTensor -> Normalize")
            if gaze_enabled:
                if gaze_output == "guide":
                    print("  - Gaze: mirror geometric ops; keep at out_size")
                else:
                    print("  - Gaze: mirror geometric ops; then downsample to grid")

        print("==================================================\n")
        return

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
            f"sigma={_fmt(getattr(pa, 'blur_sigma', None))} ({_paired_label(getattr(pa, 'paired_blur', getattr(pa, 'paired_color_jitter', False)))})"
        )
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
        if gaze_enabled:
            if gaze_output == "guide":
                print("    • gaze handling        : erased regions are zeroed in the out_size gaze map")
            else:
                print("    • gaze handling        : erased regions are zeroed in the downsampled gaze grid")
    else:
        print("  - Random erasing         : OFF")

    print("\n[Final deterministic steps]")
    print("  - ToTensor -> Normalize (backbone mean/std)")
    if gaze_enabled:
        if gaze_output == "guide":
            print("  - Gaze: geometric ops mirror image; keep at out_size")
        else:
            print("  - Gaze: geometric ops mirror image; then downsample to grid")

    print("==================================================\n")



# =================================================================================================
# Run plan helpers
# =================================================================================================

def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _infer_vit_layers(model: nn.Module) -> Optional[Tuple[int, List[int]]]:
    """
    Infer ViT encoder layer structure and which layers are trainable.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return None

    layers = getattr(backbone, "blocks", None)
    if layers is None:
        return None

    try:
        n_layers = len(layers)
    except Exception:
        return None

    trainable = []
    for i, layer in enumerate(layers):
        if any(p.requires_grad for p in layer.parameters()):
            trainable.append(i)

    return n_layers, trainable


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
    
    # --- Attention / gaze ---
    gaze_cfg = getattr(args, 'gaze_cfg', None)
    gaze_mode = str(getattr(gaze_cfg, 'mode', getattr(args, 'gaze_mode', 'disable')))
    print(f"  gaze_mode    : {gaze_mode}")
    print(f"  eyetracker   : {getattr(args, 'eyetracker_filter', 'all')}")

    if gaze_cfg is not None:
        print(f"  load_gaze    : {bool(getattr(gaze_cfg, 'load_gaze', False))}")
        print(f"  inject_gaze  : {bool(getattr(gaze_cfg, 'inject', False))}")
        print(f"  compute_kl   : {bool(getattr(gaze_cfg, 'compute_kl', False))}")
        print(f"  kl_in_loss   : {bool(getattr(gaze_cfg, 'use_kl_in_loss', False))}")
        print(f"  align_target : {getattr(gaze_cfg, 'align_target', getattr(args, 'gaze_align_target', 'attention'))}")
        if bool(getattr(gaze_cfg, 'attention_bias', False)):
            print(f"  attn_bias   : {getattr(args, 'gaze_attention_bias', 'none')} (strength={float(getattr(args, 'gaze_attention_bias_strength', 0.0))})")

    if bool(getattr(gaze_cfg, 'need_attn_maps', False)):
        print(f"  attn_mode    : {getattr(args, 'attention_mode', 'raw')}")

    print(f"  augment      : {args.augment}")
    n_ft_layers = int(getattr(args, "num_ft_layers", getattr(args, "num_ft_blocks", 0)))
    finetune_on = bool(getattr(args, "finetune", False)) and n_ft_layers > 0
    
    print(f"  finetune     : {finetune_on}")
    if finetune_on:
        print(f"  num_ft_layers: {n_ft_layers}")


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

    if args.model in ("classification", "multitask", "multitask_gaze"):
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

    gaze_cfg = getattr(args, 'gaze_cfg', None)
    if gaze_cfg is None:
        raise RuntimeError(
            "args.gaze_cfg is missing. Build it with gaze_policy.build_gaze_config(...) "
            "before printing the run plan."
        )

    gaze_mode = str(getattr(gaze_cfg, 'mode', 'disable'))

    use_kl_in_loss = bool(getattr(gaze_cfg, 'use_kl_in_loss', False))

    if use_kl_in_loss:
        parts.append(f"{args.attn_w:g}·KL(gaze↔attn)")


    print("  objective   :", " + ".join(parts))

    # ---------------------------------------------------------------------------------------------
    # Model / finetuning
    # ---------------------------------------------------------------------------------------------
    if model is not None:
        print("\n[Model]")
        total, trainable = _count_params(model)
        print(f"  parameters  : total={total:,}, trainable={trainable:,}")

        vit_info = _infer_vit_layers(model)
        if vit_info is not None:
            n_layers, trainable_layers = vit_info
            print(f"  vit layers  : {n_layers}")
            if trainable_layers:
                print(f"  unfrozen    : {trainable_layers}")
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
    
            n_trainable_layers = optimizer_info.get("n_trainable_layers", optimizer_info.get("n_trainable_blocks"))
            trainable_layer_idxs = optimizer_info.get("trainable_layer_idxs", optimizer_info.get("trainable_block_idxs"))
            if n_trainable_layers is not None:
                print(f"  trainable_vit_layers: {n_trainable_layers}")
            if trainable_layer_idxs is not None:
                show = trainable_layer_idxs
                if isinstance(show, (list, tuple)) and len(show) > 20:
                    show = list(show[:20]) + ["..."]
                print(f"  vit_layer_idxs: {show}")
    
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
