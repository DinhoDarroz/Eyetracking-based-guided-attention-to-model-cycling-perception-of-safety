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
    # Ties margin default (your original check)
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
    # Scheduler sanity checks (your original checks + stronger validation)
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
            # In your code, SSCNN doesn't return attn maps; gaze KL is not applicable.
            _warn(warnings, "--model sscnn: gaze alignment loss is not applicable; --attn_w will be ignored.")

    # Ranking-only model ignores classification knobs
    if model == "rcnn":
        if getattr(args, "use_class_weights", False):
            _warn(warnings, "--model rcnn: --use_class_weights is ignored (no CE loss).")
        if float(getattr(args, "label_smoothing", 0.0)) > 0:
            _warn(warnings, "--model rcnn: --label_smoothing is ignored (no CE loss).")

    # ------------------------------------------------------------------
    # Gaze dependencies (consistency with your pipeline behavior)
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

        # In your code, gaze alignment only makes sense if the model returns attn maps.
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
def resolve_preprocess_from_model(backbone_model, *, verbose: bool = False):
    """
    Resolve preprocessing parameters from an instantiated timm model.
    This guarantees consistency with the actual model img_size/default_cfg.
    """
    target_crop = 224
    crop_pct = 0.875
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    interpolation = "bilinear"

    try:
        from timm.data import resolve_data_config
        cfg = resolve_data_config({}, model=backbone_model)

        input_size = cfg.get("input_size", (3, 224, 224))
        target_crop = int(input_size[-1])
        crop_pct = float(cfg.get("crop_pct", crop_pct))
        mean = tuple(cfg.get("mean", mean))
        std = tuple(cfg.get("std", std))
        interpolation = str(cfg.get("interpolation", interpolation))

    except Exception as e:
        if verbose:
            print(f"[WARN] resolve_preprocess_from_model fallback to ImageNet defaults: {type(e).__name__}: {e}")

    resize_dim = int(round(target_crop / max(crop_pct, 1e-6)))

    if verbose:
        print(
            f"[preprocess] resolved from model -> crop={target_crop}, resize={resize_dim}, "
            f"crop_pct={crop_pct:.3f}, interp={interpolation}, mean={mean}, std={std}"
        )

    return target_crop, resize_dim, mean, std, interpolation, crop_pct



def build_transformer_backbone(name: str):
    """
    Build a transformer-style backbone from timm.
    
    CRITICAL CONFIGURATION:
      - global_pool="": Forces the model to return the raw feature map (Batch, SeqLen, Dim)
                        instead of a pooled vector. 
      - img_size=224:   Ensures we get exactly 14x14 patches (224/16 = 14) for 
                        gaze alignment.
    """
    
    # =========================================================================
    # 1. DINOv3 (The New State-of-the-Art)
    # =========================================================================
    if name == "dinov3_vitb16":
        # DINOv3 Base. Explicitly forcing 224 ensures 14x14 output.
        # Native resolution is often 256, but 224 works fine for finetuning.
        return timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m", 
            pretrained=True, 
            num_classes=0, 
            img_size=224,     
            global_pool="",
        )

    # =========================================================================
    # 2. BEiT v2 (The DINOv1 Replacement)
    # =========================================================================
    elif name == "beitv2_base_patch16_224":
        # Masked Image Modeling (MIM) specialist.
        # Uses .in1k_ft_in22k weights for best performance.
        return timm.create_model(
            "beitv2_base_patch16_224.in1k_ft_in22k",
            pretrained=True,
            num_classes=0,
            img_size=224,
            global_pool="",
        )

    # =========================================================================
    # 3. DeiT III (The Supervised Benchmark)
    # =========================================================================
    elif name == "deit3_base_patch16_224":
        # "Revenge of the ViT" - Strongest supervised baseline.
        return timm.create_model(
            "deit3_base_patch16_224.fb_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,
            img_size=224,
            global_pool="",
        )

    # =========================================================================
    # 4. SigLIP (The Semantic Expert)
    # =========================================================================
    elif name == "siglip_base_patch16_224":
        # Better than CLIP for zero-shot and semantics.
        return timm.create_model(
            "vit_base_patch16_siglip_224",
            pretrained=True,
            num_classes=0,
            img_size=224,
            global_pool="",
        )

    # =========================================================================
    # 5. CLIP (The Robust "Wildcard")
    # =========================================================================
    elif name == "vit_base_patch16_clip_224":
        # Standard OpenAI CLIP weights. Robust to noisy data.
        return timm.create_model(
            "vit_base_patch16_clip_224.openai",
            pretrained=True,
            num_classes=0,
            img_size=224,
            global_pool="",
        )

    # =========================================================================
    # Legacy / Other Backbones
    # =========================================================================
    
    # EVA-02 (Requires 448px for native performance, outputs 32x32 map)
    elif name == "eva02_base":
        return timm.create_model(
            "eva02_base_patch14_448.mim_in22k_ft_in1k",
            pretrained=True, 
            num_classes=0, 
            img_size=448, 
            global_pool=""
        )

    # DINO (v1) - The classic fallback
    elif name == "vit_base_dino" or name == "vit_base_patch16_224.dino":
         return timm.create_model(
            "vit_base_patch16_224.dino", 
            pretrained=True, 
            num_classes=0, 
            img_size=224, 
            global_pool=""
        )

    # DINOv2 with Registers (Patch 14 -> 16x16 output at 224px)
    elif name == "dinov2_reg_base":
        return timm.create_model(
            "vit_base_patch14_reg4_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224, 
            global_pool="",
        )

    # ConvNeXt
    elif name == "convnext_base":
        return timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        
    else:
        # Fallback for generic timm names (allows you to try others easily)
        try:
            return timm.create_model(
                name, 
                pretrained=True, 
                num_classes=0, 
                global_pool=""
            )
        except Exception:
            raise ValueError(f"Unknown transformer backbone: {name}")
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

def print_transform_policy(args, train_tfms=None, eval_tfms=None):
    """
    Print a concise, behavior-accurate summary of the transform policy.

    The function reports:
      - Backbone-specific preprocessing parameters if available (input size, crop pct, interpolation)
      - Evaluation preprocessing (deterministic)
      - Training policy:
          * augmentation disabled -> train == eval
          * augmentation enabled  -> paired, label-aware augmentation callable
    """

    # ------------------------------------------------------------------
    # Backbone specs and resolved eval geometry (preferred source: metadata)
    # ------------------------------------------------------------------
    tm = getattr(args, "transforms_meta", None)
    if isinstance(tm, dict) and isinstance(tm.get("model_specs", None), dict):
        specs = tm["model_specs"]
        if "input_size" in specs:
            print(f"  Input Size:    {specs['input_size']}")
        if "crop_pct" in specs:
            print(f"  Crop %:        {specs['crop_pct']}")
        if "interpolation" in specs:
            print(f"  Interpolation: {specs['interpolation']}")

        eval_meta = tm.get("eval", {})
        if isinstance(eval_meta, dict):
            if "resize_dim" in eval_meta:
                print(f"  Eval Resize:   {eval_meta['resize_dim']}")
            if "target_crop" in eval_meta:
                print(f"  Eval Crop:     {eval_meta['target_crop']}")

    print("\n================ AUGMENTATION PLAN ================")

    # ------------------------------------------------------------------
    # Read augment level (supports backward compatibility)
    # ------------------------------------------------------------------
    augment_level = getattr(args, "augment", "none")
    if isinstance(augment_level, bool):
        augment_level = "heavy" if augment_level else "none"
    augment_level = str(augment_level).lower().strip()

    if augment_level not in ("none", "light", "heavy"):
        augment_level = "none"

    # ------------------------------------------------------------------
    # Case 1: augmentation OFF
    # ------------------------------------------------------------------
    if augment_level == "none":
        print("Data augmentation : OFF")
        print("Train transforms  : deterministic (same as eval preprocessing)")
        print("Eval transforms   : deterministic")
        print("  - Resize(short side) → CenterCrop(out_size) → ToTensor → Normalize")
        print("==================================================\n")
        return

    # ------------------------------------------------------------------
    # Detect supported pairwise augmentation callable
    # ------------------------------------------------------------------
    is_supported_pairwise = (
        train_tfms is not None
        and train_tfms.__class__.__name__ == "Augmentation"
    )

    print(f"Data augmentation : ON ({augment_level})")
    print("Augmentation type : Pairwise, label-aware")

    if not is_supported_pairwise:
        print("\n[WARNING]")
        print("  - Expected Augmentation but found:", type(train_tfms))
        print("==================================================\n")
        return

    pa = train_tfms  # alias

    # ------------------------------------------------------------------
    # Paired structure and label behavior
    # ------------------------------------------------------------------
    print("\n[Pairwise structure]")
    print(f"  - Horizontal flip        : p={getattr(pa, 'hflip_p', 0.0):g}")
    print(f"  - Left/right swap        : p={getattr(pa, 'swap_p', 0.0):g}")

    ties_enabled = bool(getattr(args, "ties", True))
    if ties_enabled:
        print("  - Tie handling           : swap-safe (tie label preserved)")
    else:
        print("  - Binary labels          : label inverted on swap")

    # ------------------------------------------------------------------
    # Geometric augmentation (paired)
    # ------------------------------------------------------------------
    crop_p = getattr(pa, "crop_p", 0.0)
    crop_keep = getattr(pa, "crop_keep_area", None)
    rotation_p = getattr(pa, "rotation_p", 0.0)
    max_rot = getattr(pa, "max_rotation_deg", None)

    print("\n[Geometric augmentation] (paired)")
    if crop_keep is not None:
        print(f"  - Random crop            : p={crop_p:g}, keep_area≈{float(crop_keep):.2f}")
    else:
        print(f"  - Random crop            : p={crop_p:g}")

    if rotation_p and rotation_p > 0.0 and max_rot is not None:
        print(f"  - Small rotation         : p={rotation_p:g}, ±{float(max_rot):g}°")
    else:
        print("  - Small rotation         : OFF")

    # ------------------------------------------------------------------
    # Photometric augmentation (paired)
    # ------------------------------------------------------------------
    cj_p = getattr(pa, "color_jitter_p", 0.0)
    gray_p = getattr(pa, "gray_p", 0.0)

    print("\n[Photometric augmentation] (paired)")
    print(f"  - Color jitter           : p={cj_p:g}")
    print(f"  - Grayscale              : p={gray_p:g}")

    # ------------------------------------------------------------------
    # Tensor augmentation (paired)
    # ------------------------------------------------------------------
    erase_p = getattr(pa, "erase_p", 0.0)
    erase_scale = getattr(pa, "erase_scale", None)

    print("\n[Tensor augmentation] (paired)")
    print(f"  - Random erasing         : p={erase_p:g}")
    if erase_scale is not None:
        print(f"    • erased area range    : {float(erase_scale[0]):.2f}–{float(erase_scale[1]):.2f}")

    # ------------------------------------------------------------------
    # Effective preprocessing steps
    # ------------------------------------------------------------------
    print("\n[Deterministic steps]")
    print("  - Resize(short side) → Crop/Resize(out_size) → ToTensor → Normalize")

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
    print(f"  finetune     : {args.finetune}")
    if args.finetune:
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

    #print("=" * 100 + "\n")

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






