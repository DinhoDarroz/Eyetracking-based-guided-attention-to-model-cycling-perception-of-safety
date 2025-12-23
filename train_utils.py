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

# =============================================================================================== #
# Backbone factory (DeiT via torch.hub, others via timm)
# =============================================================================================== #

def build_transformer_backbone(name: str):
    """
    Build a transformer-style backbone.

    - DeiT models come from: torch.hub.load('facebookresearch/deit:main', ...)
    - DINO: torch.hub.load('facebookresearch/dino:main', ...)
    - Other ViT backbones: timm.create_model(...)
    """

    # --------------------------
    # DeiT models from torch.hub
    # --------------------------
    if name == "deit_base":
        return torch.hub.load(
            "facebookresearch/deit:main",
            "deit_base_patch16_224",
            pretrained=True,
        )
    elif name == "deit_small":
        return torch.hub.load(
            "facebookresearch/deit:main",
            "deit_small_patch16_224",
            pretrained=True,
        )
    elif name == "deit_tiny":
        return torch.hub.load(
            "facebookresearch/deit:main",
            "deit_tiny_patch16_224",
            pretrained=True,
        )
    elif name == "deit_base_distilled":
        return torch.hub.load(
            "facebookresearch/deit:main",
            "deit_base_distilled_patch16_224",
            pretrained=True,
        )

    # --------------------------
    # DINO v1 from torch.hub
    # --------------------------
    elif name == "vit_base_dino":
        return torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vitb16",
            pretrained=True,
        )

    # --------------------------
    # Dinov2, EVA, ViT-S via timm
    # --------------------------
    elif name == "vit_dinov2_base":
        return timm.create_model(
            "vit_base_patch14_reg4_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            img_size=224,
        )
    elif name == "eva02_base":
        return timm.create_model(
            "eva02_base_patch14_224.mim_in22k",
            pretrained=True,
            num_classes=0,
        )
    elif name == "vit_small":
        return timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,
        )
    elif name == "vit_base_dinov3":
        return timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m",  # timm DINOv3 ViT-B model name :contentReference[oaicite:0]{index=0}
            pretrained=True,
            num_classes=0,
            img_size=256,  # official DINOv3 ViT-B uses 256×256 inputs :contentReference[oaicite:1]{index=1}
        )

    else:
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

def print_augmentation_plan(args, train_tfms=None, eval_tfms=None):
    """
    Print a concise, behavior-accurate summary of the augmentation pipeline.
    """

    print("\n================ AUGMENTATION PLAN ================")

    # ------------------------------------------------------------------
    # Case 1: augmentation flag is OFF
    # ------------------------------------------------------------------
    if not getattr(args, "augment", False):
        print("Data augmentation : OFF")
        print("Train transforms  : deterministic (same as eval preprocessing)")
        print("  - Resize(short side) → Resize(out_size) → ToTensor → Normalize")
        print("==================================================\n")
        return

    # ------------------------------------------------------------------
    # Detect PairwiseAugmentationPipeline
    # ------------------------------------------------------------------
    is_pairwise_pipeline = (
        train_tfms is not None
        and train_tfms.__class__.__name__ == "PairwiseAugmentationPipeline"
    )

    print("Data augmentation : ON")
    print("Augmentation type : Pairwise, label-aware")

    if not is_pairwise_pipeline:
        print("\n[WARNING]")
        print("  - Expected PairwiseAugmentationPipeline but found:", type(train_tfms))
        print("==================================================\n")
        return

    pa = train_tfms  # alias for readability

    # ------------------------------------------------------------------
    # Gaze policy (run-level) and transform policy (transform-level)
    # ------------------------------------------------------------------
    gaze_enabled = (getattr(args, "gaze", "off") != "off")
    disable_aug_when_gaze = getattr(pa, "disable_aug_when_gaze", False)
    allow_swap_when_gaze = getattr(pa, "allow_swap_when_gaze", False)

    if gaze_enabled and disable_aug_when_gaze:
        print("\n[Gaze supervision policy]")
        print("  - Gaze data enabled in run")
        print("  - Augmentations are disabled on samples with eyetracker gaze (policy)")

        print("\n[Pairwise structure on gaze samples]")
        if allow_swap_when_gaze:
            print(f"  - Left/right swap        : p={pa.swap_p:g} (allowed on gaze samples)")
            print("  - Horizontal flip        : OFF on gaze samples")
        else:
            print("  - Swap / flip            : OFF on gaze samples (deterministic)")

        print("\n[Effective preprocessing on gaze samples]")
        print("  - Resize(short side) → Resize(out_size) → ToTensor → Normalize")

        # Note: non-gaze samples may still be augmented; say that explicitly.
        print("\n[Non-gaze samples]")
        print("  - Non-gaze samples follow the full augmentation plan below")

    # ------------------------------------------------------------------
    # Full pairwise augmentation (for samples where augmentation is enabled)
    # ------------------------------------------------------------------
    print("\n[Pairwise structure]")
    print(f"  - Horizontal flip        : p={pa.hflip_p:g}")
    print(f"  - Left/right swap        : p={pa.swap_p:g}")
    if args.ties:
        print("  - Tie handling           : swap-safe (tie label preserved)")
    else:
        print("  - Binary labels          : label inverted on swap")

    print("\n[Photometric augmentation] (paired)")
    print(f"  - Color jitter           : p={pa.color_jitter_p:g}")
    print(f"  - Grayscale              : p={pa.gray_p:g}")

    print("\n[Geometric augmentation] (paired)")
    print(f"  - Bottom-band crop       : p={pa.bottom_crop_p:g}")
    print(f"    • kept height fraction : {pa.bottom_keep_h[0]:.2f}–{pa.bottom_keep_h[1]:.2f}")
    print(f"    • x-jitter fraction    : {pa.bottom_x_jitter_frac:.3f}")

    print("\n[Tensor augmentation] (paired)")
    print(f"  - Random erasing         : p={pa.erase_p:g}")
    print(f"    • erased area range    : {pa.erase_scale[0]:.2f}–{pa.erase_scale[1]:.2f}")

    # Optional features: only print if they exist
    if hasattr(pa, "rotation_p") and getattr(pa, "rotation_p", 0.0) > 0.0:
        max_rot = getattr(pa, "max_rotation", None)
        if max_rot is not None:
            print("\n[Optional]")
            print(f"  - Small rotation         : p={pa.rotation_p:g}, ±{max_rot:g}°")

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
        wd = g.get("weight_decay")
        n = len(g.get("params", []))
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
    print(f"  ties         : {args.ties}")
    print(f"  gaze         : {args.gaze}")
    print(f"  augment      : {args.augment}")
    print(f"  finetune     : {args.finetune}")
    if args.finetune:
        print(f"  num_ft_blocks: {args.num_ft_blocks}")

    # ---------------------------------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------------------------------
    print("\n[Data]")
    if train_df is not None:
        print(f"  train rows   : {len(train_df):,}")
    if val_df is not None:
        print(f"  val rows     : {len(val_df):,}")
    if test_df is not None:
        print(f"  test rows    : {len(test_df):,}")

    # ---------------------------------------------------------------------------------------------
    # Transforms
    # ---------------------------------------------------------------------------------------------
    print("\n[Transforms]")
    print_augmentation_plan(args, train_tfms=train_tfms, eval_tfms=eval_tfms)

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

        print(f"  grad accum  : k={k}")
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

