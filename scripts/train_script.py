"""
Structured training loop built on Ignite.

This module keeps the original training behavior (gradient accumulation,
checkpointing, W&B logging, early stopping, Optuna pruning, etc.) but
organizes the flow into clearly named steps:

    1. Build optimizer and LR scheduler
    2. Create Ignite engines (train/validation/test)
    3. Attach metrics and event handlers
    4. Run the trainer and report the final objective

Use the public ``train`` function as the entry point.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Tuple
from timeit import default_timer as timer

import optuna
import torch
import torch.optim as optim
import wandb
# AMP (Automatic Mixed Precision) utilities
from torch.cuda.amp import autocast, GradScaler
from torch import nn

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, RunningAverage

from utils.accuracy import RankAccuracy, RankAccuracy_withMargin
from utils.losses import compute_loss
from utils.log import log

from train_utils import print_run_plan

class EarlyStopper:
    """Simple epoch-level early stopping helper."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = "max", start_epoch: int = 1):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.start_epoch = int(start_epoch)

        self.best = None
        self.best_epoch = None
        self.bad_epochs = 0

    def _is_improvement(self, current: float) -> bool:
        if self.best is None:
            return True

        if self.mode == "max":
            return current > (self.best + self.min_delta)
        return current < (self.best - self.min_delta)

    def update(self, epoch: int, current: float) -> Tuple[bool, bool]:
        """Return ``(should_stop, improved)`` after seeing ``current`` metric."""
        improved = False

        if epoch < self.start_epoch:
            return False, improved

        if self._is_improvement(current):
            self.best = float(current)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            improved = True
            return False, improved

        self.bad_epochs += 1
        should_stop = self.bad_epochs >= self.patience
        return should_stop, improved


# --------------------------------------------------------------------------------------------------------------------
# Data preparation helpers
# --------------------------------------------------------------------------------------------------------------------

def _prepare_batch(data: Dict[str, torch.Tensor], device: torch.device) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Prepare a single mini-batch produced by the DataLoader.

    Your DataLoader yields a *dict-like* batch containing (at minimum):
        - "image_l": Tensor [B, 3, H, W]   (left image)
        - "image_r": Tensor [B, 3, H, W]   (right image)

        - "score_r": Tensor [B] or [B,1]   (ranking label: -1/0/+1)
        - "score_c": Tensor [B]           (classification label: 0..C-1)

        - "gaze_l":  Tensor [B, 14, 14]   (optional gaze map, if gaze enabled)
        - "gaze_r":  Tensor [B, 14, 14]

        - "has_eyetracker": Tensor [B] bool (mask: True if gaze valid for BOTH images)

    Returns:
        inputs:
            (input_left, input_right)
            where each is on `device` and has shape [B, 3, H, W]

        labels:
            dictionary containing all targets and optional supervision signals
            needed by loss computation and/or metrics.
    """
    input_left, input_right = data["image_l"].to(device), data["image_r"].to(device)

    label_r = data["score_r"].to(device).float()
    label_c = data["score_c"].to(device).long()

    gaze_l, gaze_r = data["gaze_l"].to(device), data["gaze_r"].to(device)
    has_eye_mask = data["has_eyetracker"].to(device)

    # IMPORTANT: these keys match what utils.losses.compute_loss() expects:
    #   - label_r / label_c
    #   - gaze_l / gaze_r
    #   - has_eye_mask (optional)
    labels = {
        "label_r": label_r,
        "label_c": label_c,
        "gaze_l": gaze_l,
        "gaze_r": gaze_r,
        "has_eye_mask": has_eye_mask,
    }

    return (input_left, input_right), labels


def _build_metrics_output(args, forward_dict: Dict[str, Dict[str, torch.Tensor]], labels: Dict[str, torch.Tensor], loss: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Convert (model outputs + labels + loss) into a flat dictionary suitable for
    Ignite's `output_transform` in metrics.

    Why this exists:
        - Ignite metrics consume the *output of the engine step*.
        - Different model heads expose different outputs:
            rcnn   -> ranking scores only
            sscnn  -> classification logits only
            rsscnn -> both ranking scores + classification logits
        - To keep the metric attachment code clean, we standardize what the step
          returns per model type.

    Input forward_dict structure (from your models, e.g., nets/transformer.py):
        - forward_dict["left"]["output"]   : Tensor [B, 1] or [B]  (ranking score for left)
        - forward_dict["right"]["output"]  : Tensor [B, 1] or [B]  (ranking score for right)
        - forward_dict["logits"]["output"] : Tensor [B, C]         (classification logits)

    Output dictionary keys are intentionally aligned with train_script.py:
        - "loss" is numeric for RunningAverage (it passes loss.item()).
        - ranking metrics expect: ("rank_left", "rank_right", "label"/"label_r")
        - classification metrics expect: ("logits", "label"/"label_c")
    """
    
    if args.model == "rcnn":
        return {
            "loss": loss.item(),
            "rank_left": forward_dict["left"]["output"],
            "rank_right": forward_dict["right"]["output"],
            "label": labels["label_r"],
        }

    if args.model == "sscnn":
        return {
            "loss": loss.item(),
            "logits": forward_dict["logits"]["output"],
            "label": labels["label_c"].long(),
        }

    if args.model == "rsscnn":
        return {
            "loss": loss.item(),
            "rank_left": forward_dict["left"]["output"],
            "rank_right": forward_dict["right"]["output"],
            "logits": forward_dict["logits"]["output"],
            "label_r": labels["label_r"],
            "label_c": labels["label_c"],
        }

    raise ValueError(f"Unsupported model type: {args.model}")


# --------------------------------------------------------------------------------------------------------------------
# Optimizer / scheduler helpers
# --------------------------------------------------------------------------------------------------------------------

def _split_parameters(net: torch.nn.Module) -> Tuple[List[Tuple[str, torch.nn.Parameter]], List[Tuple[str, torch.nn.Parameter]]]:
    """
    Split parameters into two logical groups: "heads" vs "backbone".

    Why:
      - Your models have lightweight task-specific heads (ranking and/or classification)
        on top of a large pretrained backbone (CNN or ViT).
      - You often want different training behavior for these groups:
          * freeze backbone and train only heads
          * or finetune backbone with a smaller LR than heads

    How:
      - We iterate over net.named_parameters() and use name patterns to decide.
      - Any parameter whose name contains one of:
            ["rank_fc", "cross_fc"]
        is considered part of the "head" (ranking head + cross-branch classifier).
      - Everything else is treated as "backbone".

    Note:
      - This relies on your model naming conventions. If you rename layers
        in nets/transformer.py or nets/cnn.py, adjust these keys accordingly.
    """
    head_params, backbone_params = [], []
    for name, param in net.named_parameters():
        # Heads are identified by substring matches in the parameter name.
        if any(key in name for key in ["rank_fc", "cross_fc"]):
            head_params.append((name, param))
        else:
            backbone_params.append((name, param))
    return head_params, backbone_params


def _separate_decay(params):
    """
    Helper to split parameters into 'decay' (weights) and 'no_decay' (biases, layernorm).
    Used for AdamW to correctly apply weight decay only where appropriate.
    """
    decay = []
    no_decay = []
    for name, param in params:
        if not param.requires_grad:
            continue
        # Biases and LayerNorms/BatchNorms should NOT have weight decay
        if "bias" in name or "norm" in name or "len_sig" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay
    

def get_parameter_groups(
    model: nn.Module,
    *,
    base_lr: float,
    weight_decay: float,
    layer_decay: float = 0.9,
    backbone_scale: float = 0.1,
    partial_max_blocks: int = 4,
    verbose: bool = True,
):
    """
    Build optimizer parameter groups using an automatic fine-tuning policy for
    transformer-style backbones.

    Overview
    --------
    This utility constructs a list of parameter-group dictionaries compatible with
    PyTorch optimizers (typically AdamW). Each group contains:
      - "params": list of parameters
      - "lr": learning rate assigned to that group
      - "weight_decay": weight decay assigned to that group

    The function is designed for models that expose a "backbone" attribute and,
    for ViT-like architectures, a "backbone.blocks" sequence of transformer blocks.

    Policy (AUTO)
    -------------
    The policy is selected based on how many transformer blocks are actually trainable
    (i.e., contain at least one parameter with requires_grad=True):

      1) Head-only training ("head_only"):
         - No transformer blocks are trainable.
         - Only parameters outside the "backbone" namespace are grouped at base_lr.

      2) Partial fine-tuning ("partial"):
         - A small number of blocks are trainable (<= partial_max_blocks).
         - A two-tier learning rate scheme is used:
             * head lr     = base_lr
             * backbone lr = base_lr * backbone_scale (uniform across unfrozen blocks)

      3) Full fine-tuning ("full"):
         - Many blocks are trainable (> partial_max_blocks).
         - Layer-wise Learning Rate Decay (LLRD) is applied over the trainable blocks:
             * topmost trainable block uses lr = base_lr * backbone_scale
             * deeper blocks decay geometrically: lr *= layer_decay ** dist
               where dist is the distance from the topmost trainable block.

    Weight decay policy
    -------------------
    Standard practice is applied:
      - Bias parameters and normalization parameters receive weight_decay = 0.0.
      - All other parameters receive the configured weight_decay.

    Notes
    -----
    - Only parameters with requires_grad=True are included.
    - If transformer blocks cannot be identified, the function falls back to a
      simple two-tier grouping based on whether parameter names include "backbone".
    """
    if base_lr <= 0:
        raise ValueError(f"base_lr must be > 0 (got {base_lr})")
    if backbone_scale <= 0:
        raise ValueError(f"backbone_scale must be > 0 (got {backbone_scale})")
    if not (0.0 < layer_decay <= 1.0):
        raise ValueError(f"layer_decay must be in (0,1] (got {layer_decay})")
    if partial_max_blocks < 1:
        raise ValueError(f"partial_max_blocks must be >= 1 (got {partial_max_blocks})")

    # ------------------------------------------------------------------
    # Backbone discovery
    # ------------------------------------------------------------------
    # The code path targets ViT-like backbones where transformer blocks are
    # exposed as `model.backbone.blocks`. If this structure is not available,
    # a fallback policy is used that does not attempt LLRD.
    backbone = getattr(model, "backbone", None)
    blocks = getattr(backbone, "blocks", None) if backbone is not None else None

    # ------------------------------------------------------------------
    # Fallback: backbone blocks not discoverable
    # ------------------------------------------------------------------
    # When the block sequence cannot be found (e.g., CNN backbones, non-timm
    # structures, or custom wrappers), parameter grouping is performed by
    # name-based routing:
    #   - parameters containing "backbone" -> backbone group
    #   - all others                       -> head group
    # No LLRD is applied in this fallback mode.
    if blocks is None or not isinstance(blocks, (list, nn.ModuleList)) or len(blocks) == 0:
        if verbose:
            print("[Optimizer] get_parameter_groups: backbone blocks not found -> fallback (no LLRD).")

        head_decay, head_no_decay, bb_decay, bb_no_decay = [], [], [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            is_no_decay = ("bias" in name) or ("norm" in name)
            is_backbone = ("backbone" in name)

            if is_backbone:
                (bb_no_decay if is_no_decay else bb_decay).append(p)
            else:
                (head_no_decay if is_no_decay else head_decay).append(p)

        bb_lr = base_lr * backbone_scale

        groups = []
        if head_decay:
            groups.append({"params": head_decay, "lr": base_lr, "weight_decay": weight_decay})
        if head_no_decay:
            groups.append({"params": head_no_decay, "lr": base_lr, "weight_decay": 0.0})
        if bb_decay:
            groups.append({"params": bb_decay, "lr": bb_lr, "weight_decay": weight_decay})
        if bb_no_decay:
            groups.append({"params": bb_no_decay, "lr": bb_lr, "weight_decay": 0.0})

        return groups

    # ------------------------------------------------------------------
    # Determine the fine-tuning mode from trainable transformer blocks
    # ------------------------------------------------------------------
    # Trainable blocks are those for which at least one parameter has
    # requires_grad=True. This reflects the actual fine-tuning configuration
    # independent of CLI flags (e.g., partial unfreeze depth).
    trainable_block_idxs = []
    for i, blk in enumerate(blocks):
        if any(p.requires_grad for p in blk.parameters()):
            trainable_block_idxs.append(i)

    n_trainable_blocks = len(trainable_block_idxs)

    # Mode selection:
    #   - head_only: no transformer blocks are trainable
    #   - partial  : small number of blocks are trainable (stable two-tier LR)
    #   - full     : many blocks are trainable (apply true LLRD)
    if n_trainable_blocks == 0:
        mode = "head_only"
    elif n_trainable_blocks <= partial_max_blocks:
        mode = "partial"
    else:
        mode = "full"

    if verbose:
        print(
            f"  mode={mode} | "
            f"  base_lr={base_lr:.2e} | bb_top_lr={base_lr * backbone_scale:.2e} | "
            f"  layer_decay={layer_decay:.2f}"
        )
    # ------------------------------------------------------------------
    # Weight decay helper
    # ------------------------------------------------------------------
    # Biases and normalization parameters typically do not benefit from L2
    # regularization under AdamW; they are routed to weight_decay=0.0.
    def _wd_for(name: str) -> float:
        if ("bias" in name) or ("norm" in name):
            return 0.0
        return float(weight_decay)

    # ------------------------------------------------------------------
    # Group assembly helper
    # ------------------------------------------------------------------
    # groups_map provides stable grouping by (group_name, lr, wd). This ensures
    # parameters with the same LR and WD share the same optimizer group, which
    # keeps the optimizer configuration compact and predictable.
    groups_map = {}

    def _add_param(group_name: str, lr: float, wd: float, p: torch.nn.Parameter):
        key = (group_name, float(lr), float(wd))
        if key not in groups_map:
            groups_map[key] = {"params": [], "lr": float(lr), "weight_decay": float(wd)}
        groups_map[key]["params"].append(p)

    bb_top_lr = base_lr * backbone_scale

    # ------------------------------------------------------------------
    # Parameter routing
    # ------------------------------------------------------------------
    # Routing is performed by parameter name:
    #   - Non-backbone parameters  -> "head" (base_lr)
    #   - Backbone parameters      -> depends on mode:
    #       * partial: uniform bb_top_lr
    #       * full   : block-wise LLRD for "blocks.*" parameters
    #                 and smaller LR for embeddings (patch/pos/cls)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        wd = _wd_for(name)

        # Head parameters: anything outside the backbone namespace.
        if "backbone" not in name:
            _add_param("head", base_lr, wd, p)
            continue

        # Backbone parameters.
        if mode == "head_only":
            # This branch is rarely used because mode=head_only implies no trainable blocks,
            # but if some backbone parameter is trainable (e.g., backbone norm), it is
            # treated conservatively using bb_top_lr.
            _add_param("backbone", bb_top_lr, wd, p)
            continue

        if mode == "partial":
            # Partial fine-tuning uses a uniform LR for all trainable backbone parameters.
            # This tends to be more stable than LLRD when only a few blocks are unfrozen.
            _add_param("backbone", bb_top_lr, wd, p)
            continue

        # mode == "full": apply LLRD over transformer blocks.
        if "blocks" in name:
            try:
                parts = name.split(".")
                bidx = int(parts[parts.index("blocks") + 1])

                # The topmost trainable block (largest index) gets distance=0.
                top_idx = max(trainable_block_idxs) if trainable_block_idxs else (len(blocks) - 1)
                dist = max(0, top_idx - bidx)

                lr = bb_top_lr * (layer_decay ** dist)
                _add_param(f"block_{bidx}", lr, wd, p)
                continue
            except Exception:
                # When parsing fails, default to a conservative top backbone LR.
                _add_param("backbone_misc", bb_top_lr, wd, p)
                continue

        # Non-block backbone parameters:
        # Embedding-related parameters are often sensitive and benefit from smaller LRs
        # during full fine-tuning. Other backbone parameters (e.g., final norms) use
        # the top backbone LR.
        if any(k in name for k in ("patch_embed", "pos_embed", "cls_token")):
            lr = bb_top_lr * (layer_decay ** (len(blocks)))
            _add_param("embeddings", lr, wd, p)
        else:
            _add_param("backbone_other", bb_top_lr, wd, p)

    return list(groups_map.values())


def _build_optimizer(args, net: torch.nn.Module, is_transformer: bool, head_params: list, backbone_params: list):
    """
    Construct the optimizer according to the backbone family and fine-tuning policy.

    High-level flow
    ---------------
    1) The backbone type selects the optimizer family:
         - Transformer-style backbones -> AdamW (typical for ViT/DeiT/DINO/EVA/ConvNeXt-style transformers)
         - CNN-style backbones         -> Adam  (simple, uniform-LR baseline)

    2) The fine-tuning flag (args.finetune) selects the parameter routing policy:
         - finetune=True  -> optimize head + (some or all) backbone parameters
         - finetune=False -> optimize head parameters only

    Transformer policy details
    --------------------------
    - When finetuning transformers, parameter groups are produced by get_parameter_groups().
      This implements an automatic choice between:
         * head-only grouping
         * two-tier LR for partial unfreezing
         * true LLRD for deeper unfreezing

    - When transformers are frozen (finetune=False), only head parameters are optimized.

    CNN policy details
    ------------------
    - CNNs use a single Adam optimizer configuration with a uniform learning rate.
    - When finetune=False, only head parameters are optimized.
    - Weight decay is set to 0.0 for the CNN branch to match a common baseline setup.
    """
    base_lr = args.base_lr
    weight_decay = args.weight_decay

    # =========================================================================
    # POLICY A: TRANSFORMERS (AdamW)
    # =========================================================================
    if is_transformer:
        print(f"[Optimizer]")
        print(f"  Mode: Transformer (AdamW)") 

        # ---------------------------------------------------------------------
        # Fine-tuning enabled: build parameter groups (AUTO policy)
        # ---------------------------------------------------------------------
        if args.finetune:
            layer_decay = float(getattr(args, "layer_decay", 0.9))
            backbone_scale = float(getattr(args, "backbone_scale", 0.1))
            partial_max_blocks = int(getattr(args, "partial_max_blocks", 4))

            optimizer_params = get_parameter_groups(
                net,
                base_lr=base_lr,
                weight_decay=weight_decay,
                layer_decay=layer_decay,
                backbone_scale=backbone_scale,
                partial_max_blocks=partial_max_blocks,
                verbose=True,
            )

            return optim.AdamW(
                optimizer_params,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        # ---------------------------------------------------------------------
        # Frozen transformer backbone: optimize head only
        # ---------------------------------------------------------------------
        else:
            head_decay, head_no_decay = _separate_decay(head_params)

            return optim.AdamW(
                [
                    {"params": head_decay, "lr": base_lr, "weight_decay": weight_decay},
                    {"params": head_no_decay, "lr": base_lr, "weight_decay": 0.0},
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
            )

    # =========================================================================
    # POLICY B: CNNs (Adam)
    # =========================================================================
    else:
        # For CNNs, use a uniform-LR Adam configuration.

        if args.finetune:
            # Full fine-tuning: head + backbone parameters at the same LR.
            params_to_optimize = [p for p in net.parameters() if p.requires_grad]
        else:
            # Head-only training: optimize head parameters only.
            params_to_optimize = [p for (_, p) in head_params if p.requires_grad]

        return optim.Adam(
            params_to_optimize,
            lr=base_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )


def _build_scheduler(args, optimizer, accum_steps: int, steps_per_epoch: int, base_lr: float):
    """
    Build the LR scheduler used by the Ignite training loop.

    Key concept: "optimizer steps" vs "dataloader iterations"
      - You support gradient accumulation (args.k).
      - That means optimizer.step() happens only every accum_steps iterations.
      - Schedulers that step per-optimizer-step must be configured using the number of *optimizer steps*,
        not raw dataloader batches.

    Computation:
      - eff_steps_per_epoch = ceil(steps_per_epoch / accum_steps)
            steps_per_epoch is len(dataloader) (number of batches per epoch).
            dividing by accum_steps gives number of optimizer updates per epoch.
      - total_iters = args.max_epochs * eff_steps_per_epoch
            total number of optimizer updates over the entire run.

    Scheduler selection:
      - args.scheduler chooses the strategy.
      - Returns:
          (scheduler_instance_or_None, scheduler_type_string)
    """
    # Effective number of optimizer updates per epoch (accounts for gradient accumulation).
    eff_steps_per_epoch = math.ceil(steps_per_epoch / accum_steps)

    # Total number of optimizer steps across all epochs (this is the "time axis" for most schedulers).
    total_iters = args.max_epochs * eff_steps_per_epoch

    scheduler_type = args.scheduler
    scheduler = None

    # ------------------------------------------------------------------
    # 0) No scheduler: constant LR
    # ------------------------------------------------------------------
    if scheduler_type == "none":
        scheduler = None

    # ------------------------------------------------------------------
    # 1) Warmup + Cosine decay (implemented via LambdaLR)
    # ------------------------------------------------------------------
    elif scheduler_type == "warmup_cosine":
        # Warmup fraction is clamped to [0, 1] to avoid misconfiguration.
        warmup_frac = max(0.0, min(1.0, args.warmup_frac))
        warmup_iters = int(warmup_frac * total_iters)

        # eta_min is the absolute minimal LR (not a multiplier) used as the cosine floor.
        eta_min = args.eta_min

        def lr_lambda(step: int):
            """
            LambdaLR expects a multiplicative factor applied to each param_group's initial LR.

            Behavior:
              - During warmup: linear ramp from 0 → 1 over warmup_iters steps
              - After warmup: cosine decay from 1 → (eta_min/base_lr)

            Notes:
              - base_lr here is used to convert eta_min (absolute) into a factor.
              - We protect divisions with max(1, ...) to avoid zero division.
            """
            # Linear warmup phase: increase LR proportionally with step index.
            if warmup_iters > 0 and step < warmup_iters:
                return float(step) / float(max(1, warmup_iters))

            # Degenerate case: if warmup consumes all steps, hold at eta_min/base_lr.
            if total_iters == warmup_iters:
                return eta_min / base_lr

            # Cosine decay phase over the remaining steps.
            progress = min(1.0, max(0.0, (step - warmup_iters) / float(max(1, total_iters - warmup_iters))))

            # Convert eta_min to a multiplier relative to base_lr.
            eta_min_factor = eta_min / base_lr

            # Cosine interpolation between 1.0 and eta_min_factor.
            return eta_min_factor + (1 - eta_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ------------------------------------------------------------------
    # 2) Cosine decay (no warmup)
    # ------------------------------------------------------------------
    elif scheduler_type == "cosine":
        # Standard cosine annealing from initial LR to eta_min over total_iters steps.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_iters,
            eta_min=args.eta_min,
        )

    # ------------------------------------------------------------------
    # 3) OneCycleLR
    # ------------------------------------------------------------------
    elif scheduler_type == "onecycle":
        # OneCycleLR requires max_lr per param group.
        # We set max_lr to the optimizer's current LR per group, so the schedule
        # oscillates around those group-specific settings.
        max_lrs = [pg["lr"] for pg in optimizer.param_groups]

        # pct_start controls fraction of the cycle spent increasing LR.
        # Here, we reuse warmup_frac as the "increase" fraction (common in your sweep configs).
        warmup_frac = max(0.0, min(1.0, args.warmup_frac))
        pct_start = warmup_frac if warmup_frac > 0 else 0.3

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_iters,
            pct_start=pct_start,
            anneal_strategy="cos",
            cycle_momentum=False,  # Adam/AdamW style; momentum cycling not used
        )

    # ------------------------------------------------------------------
    # 4) CosineAnnealingWarmRestarts
    # ------------------------------------------------------------------
    elif scheduler_type == "warm_restarts":
        # Warm restarts: cosine cycles of length T_0, multiplied by T_mult each restart.
        # IMPORTANT: This scheduler uses "scheduler.step()" calls as its time steps.
        # In your training loop, you call scheduler.step() every optimizer update
        # (except for plateau), so T_0 is interpreted in optimizer-step units.
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.T_0,
            T_mult=args.T_mult,
            eta_min=args.eta_min,
        )

    # ------------------------------------------------------------------
    # 5) ReduceLROnPlateau
    # ------------------------------------------------------------------
    elif scheduler_type == "plateau":
        # Plateau scheduler steps on a validation metric (here: mode="max" for accuracy).
        # In your training loop, you do NOT call scheduler.step() each iteration for plateau;
        # instead you call it at epoch end with val accuracy.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler, scheduler_type


# ------------------------------------------------------------------------------------------------------------------
# Engine step factories
# ------------------------------------------------------------------------------------------------------------------

def _make_train_step(
    args,
    device: torch.device,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scheduler_type: str,
    accum_steps: int,
    grad_clip: float,
    logger,
    trial,
    scaler: GradScaler,
):
    """
    Factory that builds a single training step callable for Ignite.

    This function returns a closure (`train_step`) that:
      - Matches Ignite's required signature: (engine, batch)
      - Has access to all training configuration via lexical scoping
      - Supports gradient accumulation, gradient clipping, schedulers,
        Optuna pruning, and user-triggered early termination

    Returns
    -------
    Callable
        Ignite-compatible training step function.
    """

    def train_step(engine, data):
        """
        Single training iteration executed by Ignite.

        Parameters
        ----------
        engine : ignite.engine.Engine
            Ignite engine instance.
        data : tuple
            Batch produced by the DataLoader.

        Returns
        -------
        dict
            Dictionary of metrics for logging.
        """

        # -----------------------------
        # User-requested hard stop
        # -----------------------------
        if os.path.exists("SKIP_TRIAL"):
            print("[USER REQUEST] SKIPPING THIS RUN NOW.")
            os.remove("SKIP_TRIAL")

            if trial is not None:
                raise optuna.TrialPruned()

            engine.terminate()
            return {"skipped": True}

        # -----------------------------
        # Optional timing
        # -----------------------------
        if logger:
            start = timer()

        # -----------------------------
        # Forward pass (AMP-capable)
        # -----------------------------
        inputs, labels = _prepare_batch(data, device)
        
        # AMP is controlled by the scaler: if scaler is disabled, autocast does nothing.
        use_amp = scaler.is_enabled()
        
        with autocast(enabled=use_amp):
            forward_dict = net(*inputs)
            
        loss = compute_loss(args, forward_dict, labels)
        
        # Keep the *unscaled* loss for logging/metrics.
        raw_loss = loss
        
        # -----------------------------
        # NaN guard (debug safety)
        # -----------------------------
        if torch.isnan(loss):
            label_r = labels["label_r"]
            n_ties = (label_r == 0).sum().item()
            n_nonties = (label_r != 0).sum().item()
            print(
                f"[NaN DETECTED] epoch={engine.state.epoch} "
                f"iter={engine.state.iteration} "
                f"batch_size={label_r.size(0)}, "
                f"n_nonties={n_nonties}, n_ties={n_ties}"
            )
            raise ValueError("NaN loss detected; stopping for debugging.")
        
        # -----------------------------
        # Backward pass (accumulated, AMP-safe)
        # -----------------------------
        # IMPORTANT: divide by accum_steps BEFORE backward so the accumulated gradient matches
        # a single batch of size (batch_size * accum_steps).
        loss = loss / accum_steps
        
        if use_amp:
            # Scaled backward prevents fp16 gradient underflow
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # -----------------------------
        # Optimizer / scheduler step (only on accumulation boundary)
        # -----------------------------
        if engine.state.iteration % accum_steps == 0:
        
            # If using AMP, unscale gradients before clipping so clipping sees true magnitudes.
            if use_amp:
                scaler.unscale_(optimizer)
        
            # Optional gradient clipping (global norm)
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
        
            # Step optimizer (AMP-aware)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        
            # Clear gradients for next accumulation window
            optimizer.zero_grad(set_to_none=True)
        
            # Step per-optimizer-step schedulers (not plateau)
            if scheduler and scheduler_type != "plateau":
                scheduler.step()
        

        # -----------------------------
        # Logging
        # -----------------------------
        if logger:
            logger.info(f"TRAIN_STEP, {timer() - start:.4f}")

        return _build_metrics_output(args, forward_dict, labels, raw_loss)

    return train_step


def _make_inference_step(args, device: torch.device, net: torch.nn.Module):
    """
    Factory that builds an Ignite-compatible inference step.

    Used for validation and test engines. No gradients, no optimizer,
    deterministic behavior.
    """
    def inference_step(engine, data):
        # AMP inference is safe and reduces memory; it should match training AMP setting.
        use_amp = bool(getattr(args, "amp", False)) and bool(getattr(args, "cuda", False)) and (device.type == "cuda")
    
        with torch.no_grad():
            inputs, labels = _prepare_batch(data, device)
    
            with autocast(enabled=use_amp):
                forward_dict = net(*inputs)
                
            loss = compute_loss(args, forward_dict, labels)
    
            return _build_metrics_output(args, forward_dict, labels, loss)

    return inference_step


# --------------------------------------------------------------------------------------------------------------------
# Metric + handler helpers
# --------------------------------------------------------------------------------------------------------------------

def _attach_metrics(engines: List[Engine], args, device: torch.device) -> None:
    """
    Attach Ignite metrics to one or more engines (typically: train / validation / test).

    When this is activated:
      - This function is called during pipeline setup, after the Ignite Engine(s) are created
        and before engine.run(...) is executed.
      - It is executed once per program run (or once per re-build of engines), not every iteration.
      - The attached metrics are then computed automatically by Ignite during engine execution:
          * RunningAverage updates every iteration (batch) for the current engine.
          * Accuracy / RankAccuracy accumulate across the epoch and are finalized at EPOCH_COMPLETED.

    Why it exists:
      - Different model modes (rcnn / sscnn / rsscnn) emit different outputs
        (ranking scores vs classification logits), so the correct metrics must be
        attached based on args.model.
      - "full_accuracy" toggles whether ranking accuracy is computed with an
        explicit margin criterion or as a pure ordering comparison.

    Expected engine output dictionary keys:
      - rcnn:   {"loss", "rank_left", "rank_right", "label", ...}
      - sscnn:  {"loss", "logits", "label", ...}
      - rsscnn: {"loss", "rank_left", "rank_right", "label_r", "logits", "label_c", ...}

    Notes on metric names:
      - "loss"  : RunningAverage over per-iteration loss values (smoothed training curve).
      - "acc"   : ranking accuracy (rcnn or rsscnn ranking branch).
      - "c_acc" : classification accuracy (rsscnn classification branch only).
    """
    for engine in engines:
        # ---------------------------------------------------------------------
        # RCNN: ranking-only training/evaluation (no classification head metric).
        # ---------------------------------------------------------------------
        if args.model == "rcnn":
            # RunningAverage tracks a smoothed loss over iterations for this engine.
            RunningAverage(output_transform=lambda x: x["loss"], device=device).attach(engine, "loss")

            # Ranking accuracy can optionally enforce a margin criterion.
            # - full_accuracy=True  : prediction is only correct if the score difference
            #                         exceeds args.ranking_margin in the correct direction.
            # - full_accuracy=False : prediction is correct if ordering matches the label,
            #                         ignoring margin magnitude.
            if args.full_accuracy:
                RankAccuracy_withMargin(
                    output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label"], args.ranking_margin),
                    device=device,
                ).attach(engine, "acc")
            else:
                RankAccuracy(
                    output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label"]),
                    device=device,
                ).attach(engine, "acc")

        # ---------------------------------------------------------------------
        # SSCNN: classification-only training/evaluation (no ranking metric).
        # ---------------------------------------------------------------------
        elif args.model == "sscnn":
            RunningAverage(output_transform=lambda x: x["loss"], device=device).attach(engine, "loss")

            # Standard multiclass accuracy using predicted logits vs ground-truth label.
            # Assumes logits shape [B, num_classes] and label shape [B].
            Accuracy(output_transform=lambda x: (x["logits"], x["label"])).attach(engine, "acc")

        # ---------------------------------------------------------------------
        # RSSCNN: ranking + classification (attach both ranking acc and class acc).
        # ---------------------------------------------------------------------
        elif args.model == "rsscnn":
            RunningAverage(output_transform=lambda x: x["loss"], device=device).attach(engine, "loss")

            # Ranking metric uses label_r (pairwise ranking label: left/tie/right).
            if args.full_accuracy:
                RankAccuracy_withMargin(
                    output_transform=lambda x: (
                        x["rank_left"],
                        x["rank_right"],
                        x["label_r"],
                        args.ranking_margin,
                    ),
                    device=device,
                ).attach(engine, "acc")
            else:
                RankAccuracy(
                    output_transform=lambda x: (x["rank_left"], x["rank_right"], x["label_r"]),
                    device=device,
                ).attach(engine, "acc")

            # Classification accuracy attached under a distinct name to avoid collisions.
            Accuracy(output_transform=lambda x: (x["logits"], x["label_c"])).attach(engine, "c_acc")

        # ---------------------------------------------------------------------
        # Defensive programming: reject unknown model identifiers early.
        # ---------------------------------------------------------------------
        else:
            raise ValueError(f"Unsupported model type: {args.model}")


def _compute_class_breakdown(args, net, loader, device, split_name: str, epoch_idx: int, print_output: bool = True):
    """
    Computes a confusion matrix adapted to:
    - ties=True  → 3 classes  (0=left, 1=tie, 2=right)
    - ties=False → 2 classes  (0=left, 1=right)
    
    If print_output=False → do NOT print anything (useful for test set).
    """

    if args.model not in ["sscnn", "rsscnn"]:
        return None

    if args.ties:
        num_classes = 3
        class_names = ["left", "tie", "right"]
    else:
        num_classes = 2
        class_names = ["left", "right"]

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    net.eval()
    with torch.no_grad():
        for batch in loader:
            input_left = batch["image_l"].to(device)
            input_right = batch["image_r"].to(device)
            label_c = batch["score_c"].to(device).long()

            forward_dict = net(input_left, input_right)
            logits = forward_dict["logits"]["output"]
            preds = torch.argmax(logits, dim=1)

            for true_cls, pred_cls in zip(label_c.view(-1), preds.view(-1)):
                t = int(true_cls.item())
                p = int(pred_cls.item())
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    confusion[t, p] += 1

    if not print_output:
        return confusion

    print(f"\n[Epoch {epoch_idx}] {split_name} classification breakdown (true x pred)")
    print("Rows = true class, Cols = predicted class")
    print("Class mapping:", {i: name for i, name in enumerate(class_names)})
    print(confusion.cpu().numpy())

    for cls in range(num_classes):
        row = confusion[cls]
        total = int(row.sum().item())
        correct = int(row[cls].item())
        incorrect = total - correct

        if total == 0:
            print(f"  True class {cls} ({class_names[cls]}): no samples in this split.")
            continue

        print(
            f"\n  True class {cls} ({class_names[cls]}): total={total}, "
            f"correct={correct}, incorrect={incorrect} "
            f"({incorrect/total:.3f} misclass rate)"
        )

        if incorrect > 0:
            for pred_cls in range(num_classes):
                if pred_cls == cls:
                    continue
                count = int(row[pred_cls].item())
                if count > 0:
                    print(
                        f"    misclassified as {pred_cls} ({class_names[pred_cls]}): "
                        f"{count} ({count/incorrect:.3f} of misclassified)"
                    )

    print()
    return confusion


# --------------------------------------------------------------------------------------------------------------------
# Validation / logging handlers
# --------------------------------------------------------------------------------------------------------------------

def _attach_epoch_end_step(trainer: Engine, optimizer, accum_steps: int, scaler: GradScaler):
    """
    Flush a partially accumulated gradient at epoch end.

    With gradient accumulation, optimizer.step() runs only every accum_steps iterations.
    If the epoch ends mid-window, we must apply one final step.

    IMPORTANT: Under AMP, we must use scaler.step(...) (not optimizer.step()).
    """
    @trainer.on(Events.EPOCH_COMPLETED)
    def step_on_epoch_end(engine):
        if engine.state.iteration % accum_steps != 0:
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

def _make_validation_handler(
    args,
    net,
    trainer: Engine,
    evaluator: Engine,
    evaluator_test: Engine,
    optimizer,
    scheduler,
    scheduler_type: str,
    val_loader,
    test_loader,
    early_stopper: EarlyStopper | None,
    start_training: float,
    training_state: Dict[str, float | List[float]],
    device: torch.device,
):
    """
    Build and return an epoch-end validation/logging callback.

    When this is activated:
      - This function is called once during pipeline setup to create a handler function.
      - The returned 'log_validation_results' function is typically attached to the trainer as:
            trainer.on(Events.EPOCH_COMPLETED)(log_validation_results)
        (or equivalent).
      - As a result, 'log_validation_results' runs once per epoch at EPOCH_COMPLETED.

    Responsibilities of the handler:
      1) Run evaluation on validation and test dataloaders via separate evaluator engines.
      2) Update trainer state with validation metrics used by external components (e.g., Optuna pruning).
      3) Step schedulers that require validation feedback (ReduceLROnPlateau).
      4) Track best validation accuracy across epochs.
      5) Report intermediate results to Optuna and optionally prune trials.
      6) Optionally compute extra per-class / per-label breakdowns for classification models.
      7) Perform model mode switches (eval/train) and optional partial_eval hooks.
      8) Apply early stopping and terminate training if criteria are met.
      9) Emit a consolidated metrics dictionary through the project's logging utility.
    """
    def log_validation_results(engine):
        # ---------------------------------------------------------------------
        # 1) Switch model to evaluation mode and run evaluators
        # ---------------------------------------------------------------------
        net.eval()

        # Evaluator engines compute metrics over the full dataloader and store them
        # in evaluator.state.metrics (and evaluator_test.state.metrics).
        evaluator.run(val_loader)
        evaluator_test.run(test_loader)

        # Persist validation accuracy in trainer metrics for external consumers.
        engine.state.metrics["val_acc"] = evaluator.state.metrics["acc"]

        current_val_acc = float(evaluator.state.metrics["acc"])
        current_train_acc = engine.state.metrics.get("acc")
        current_test_acc = evaluator_test.state.metrics.get("acc")

        # ---------------------------------------------------------------------
        # 2) Scheduler stepping (only for validation-dependent schedulers)
        # ---------------------------------------------------------------------
        # ReduceLROnPlateau requires a monitored metric value; it should be stepped
        # once per epoch after validation, not every iteration.
        if scheduler and scheduler_type == "plateau":
            scheduler.step(current_val_acc)

            # Optional console sanity print: helps confirm plateau is receiving the metric
            # and that LR updates occur as expected.
            lr_head = optimizer.param_groups[0]["lr"]
            print(
                f"[Plateau sanity] "
                f"epoch={engine.state.epoch} "
                f"val_acc={current_val_acc:.4f} "
                f"lr={lr_head:.3e}"
            )

        # ---------------------------------------------------------------------
        # 3) Track validation accuracy history and best-so-far (selection-coupled)
        # ---------------------------------------------------------------------
        training_state["val_acc_history"].append(current_val_acc)
        
        # IMPORTANT: selection is based on validation only.
        # When validation improves, snapshot the corresponding train/test accuracies
        # from THIS SAME epoch (same weights).
        if current_val_acc > float(training_state["best_val_acc"]):
            training_state["best_val_acc"] = current_val_acc
            training_state["epoch_best_val"] = int(engine.state.epoch)
        
            # Snapshot train/test acc at the best-val epoch (may be None if metric missing)
            if current_train_acc is not None:
                training_state["train_acc_at_best_val"] = float(current_train_acc)
            if current_test_acc is not None:
                training_state["test_acc_at_best_val"] = float(current_test_acc)


        # ---------------------------------------------------------------------
        # 4) Optuna integration (report + pruning)
        # ---------------------------------------------------------------------
        # If this run is being driven by an Optuna trial, report the current best
        # objective value and allow Optuna to prune unpromising runs.
        if engine.state.trial is not None:
            current_epoch = engine.state.epoch
            engine.state.trial.report(training_state["best_val_acc"], step=current_epoch)
            if engine.state.trial.should_prune():
                raise optuna.TrialPruned()

        # ---------------------------------------------------------------------
        # 5) Optional detailed breakdown for classification-capable models
        # ---------------------------------------------------------------------
        epoch_idx = engine.state.epoch
        if args.model in ["sscnn", "rsscnn"]:
            # Computes extra diagnostic breakdowns (e.g., per-class accuracy) for the
            # classification output branch; used for interpretability/debugging.
            _compute_class_breakdown(args, net, val_loader, device, "Validation", epoch_idx)

        # ---------------------------------------------------------------------
        # 6) Optional model-specific evaluation behavior
        # ---------------------------------------------------------------------
        # Some backbones/wrappers expose a partial_eval method to evaluate in a
        # reduced or specialized mode (e.g., freezing stochastic components).
        if hasattr(net, "partial_eval"):
            net.partial_eval()

        # Restore training mode so the next epoch uses dropout, etc.
        net.train()

        # ---------------------------------------------------------------------
        # 7) Assemble a consolidated metrics dictionary for logging
        # ---------------------------------------------------------------------
        metrics = {
            # Train metrics are taken from the trainer engine metrics accumulated during the epoch.
            "accuracy_train": engine.state.metrics.get("acc"),

        
            # Validation/test metrics come from the evaluator engines (current epoch weights).
            "accuracy_validation": evaluator.state.metrics["acc"],
            "accuracy_test": evaluator_test.state.metrics["acc"],
            "loss_validation": evaluator.state.metrics["loss"],
            "loss_test": evaluator_test.state.metrics["loss"],

            "loss_train": engine.state.metrics["loss"],
        
            # Wall-clock time since training started (string for log consistency).
            "time": f"{timer() - start_training:.3f}",
        
            # Bookkeeping for reproducibility and alignment with logs.
            "epoch": engine.state.epoch,
            "iteration": engine.state.iteration,
        
            # -----------------------------
            # Selection-coupled "best" stats
            # -----------------------------
            # Best validation accuracy observed so far (selection criterion).
            "max_accuracy_validation": training_state["best_val_acc"],
        
            # Legacy keys (keep for backward compatibility):
            # These are NOT "max over epochs". They are train/test accuracy
            # at the epoch where validation was best.
            "max_accuracy_train": training_state["train_acc_at_best_val"],
            "max_accuracy_test": training_state["test_acc_at_best_val"],
        
            # Explicit names (recommended for thesis clarity).
            #"accuracy_train_at_best_val": training_state["train_acc_at_best_val"],
            #"accuracy_test_at_best_val": training_state["test_acc_at_best_val"],
            "epoch_best_val": training_state["epoch_best_val"],
        }
        # RSSCNN exposes an additional classification metric ("c_acc") alongside ranking accuracy.
        if args.model == "rsscnn":
            metrics.update(
                {
                    "c_accuracy_train": engine.state.metrics["c_acc"],
                    "c_accuracy_validation": evaluator.state.metrics["c_acc"],
                    "c_accuracy_test": evaluator_test.state.metrics["c_acc"],
                }
            )

        # ---------------------------------------------------------------------
        # 8) Early stopping (optional) based on a chosen validation metric
        # ---------------------------------------------------------------------
        if early_stopper is not None:
            # Select which validation signal to monitor. Defaults to validation accuracy
            # if the requested metric is not available.
            monitor_name = getattr(args, "early_stop_metric", "accuracy_validation")

            # Map of possible monitored signals computed at validation time.
            available = {
                "accuracy_validation": float(evaluator.state.metrics.get("acc", 0.0)),
                "loss_validation": float(evaluator.state.metrics.get("loss", 0.0)),
            }
            if args.model == "rsscnn":
                available["c_accuracy_validation"] = float(evaluator.state.metrics.get("c_acc", 0.0))

            if monitor_name not in available:
                monitor_name = "accuracy_validation"

            current_value = float(available[monitor_name])

            # Update early stopper state and determine whether training should terminate.
            should_stop, _ = early_stopper.update(engine.state.epoch, current_value)

            # Log early stopping diagnostics for transparency and post-hoc analysis.
            #metrics.update(
            #    {
            #        "early_stop/metric": monitor_name,
            #        "early_stop/value": current_value,
            #        "early_stop/best": None if early_stopper.best is None else float(early_stopper.best),
            #        "early_stop/best_epoch": None if early_stopper.best_epoch is None else int(early_stopper.best_epoch),
            #        "early_stop/bad_epochs": int(early_stopper.bad_epochs),
            #    }
            #)

            if should_stop:
                # Human-readable stop reason for logs and W&B summary.
                stop_reason = (
                    f"Early stopping: no improvement in '{monitor_name}' "
                    f"for {early_stopper.patience} epoch(s)."
                )

                # Persist early-stop information into W&B run summary for quick inspection.
                if args.log_wandb and wandb.run is not None:
                    wandb.summary["early_stopped"] = True
                    wandb.summary["early_stop_reason"] = stop_reason
                    wandb.summary["early_stop_metric"] = monitor_name
                    wandb.summary["early_stop_mode"] = getattr(args, "early_stop_mode", "max")
                    wandb.summary["early_stop_patience"] = getattr(args, "early_stop_patience", 3)
                    wandb.summary["early_stop_min_delta"] = getattr(args, "early_stop_min_delta", 0.0)
                    wandb.summary["early_stop_best"] = None if early_stopper.best is None else float(early_stopper.best)
                    wandb.summary["early_stop_best_epoch"] = None if early_stopper.best_epoch is None else int(early_stopper.best_epoch)
                    wandb.summary["early_stop_stopped_epoch"] = int(engine.state.epoch)

                # Stop the trainer engine cleanly at the end of this epoch.
                engine.terminate()

        # ---------------------------------------------------------------------
        # 9) Emit metrics to the project's logger (console / file / W&B, depending on args)
        # ---------------------------------------------------------------------
        log(args, metrics)

    return log_validation_results

# --------------------------------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------------------------------

def train(device, net, dataloader, val_loader, test_loader, args, logger, trial=None):
    """
    Main training entrypoint (Ignite-based).

    This function orchestrates the full experiment lifecycle:
      - Model/device setup
      - Optimizer and scheduler creation (with transformer-aware parameter groups)
      - Ignite engine construction for training/validation/testing
      - Metric attachment and per-epoch validation hook
      - Checkpointing (top-k, best, and last)
      - Optional resume behavior (epoch and max_epochs overrides)
      - W&B finalization
      - Return a scalar objective suitable for Optuna/W&B sweeps

    Parameters
    ----------
    device : torch.device
        Target device to run the model on.
    net : torch.nn.Module
        Model to train.
    dataloader : torch.utils.data.DataLoader
        Training dataloader.
    val_loader : torch.utils.data.DataLoader
        Validation dataloader.
    test_loader : torch.utils.data.DataLoader
        Test dataloader (evaluated at epoch end).
    args : argparse.Namespace
        Experiment configuration (hyperparameters, logging, checkpointing, etc.).
    logger : logging.Logger or None
        Optional structured logger.
    trial : optuna.Trial or None, optional
        Optuna trial handle (enables pruning and user attribute logging).

    Returns
    -------
    float
        Final validation accuracy proxy used as the objective (smoothed when possible).
    """

    # ------------------------------------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------------------------------------

    # Gradient accumulation: perform one optimizer update every `accum_steps` iterations.
    # The argument name `k` is treated as the accumulation factor.
    accum_steps = max(1, getattr(args, "k", 1))

    # Gradient clipping: stabilizes training by limiting global norm of parameter gradients.
    grad_clip = getattr(args, "grad_clip", 0.0)

    # Optional early stopping controller (patience-based with configurable directionality).
    early_stopper = None
    if getattr(args, "early_stop", False):
        early_stopper = EarlyStopper(
            patience=getattr(args, "early_stop_patience", 3),
            min_delta=getattr(args, "early_stop_min_delta", 0.0),
            mode=getattr(args, "early_stop_mode", "max"),
            start_epoch=getattr(args, "early_stop_start_epoch", 1),
        )

    # Centralized state for cross-handler communication and summary statistics.
    # `val_acc_history` supports smoothing and robust final reporting.
    training_state: Dict[str, float | List[float] | int | None] = {
        "best_val_acc": 0.0,
        "train_acc_at_best_val": float("-inf"),
        "test_acc_at_best_val": float("-inf"),
        "epoch_best_val": None,
        "val_acc_history": [],
    }


    # ------------------------------------------------------------------------------------------------
    # Model and optimization setup
    # ------------------------------------------------------------------------------------------------
    """
    # Ensure the model is on the correct device.
    net = net.to(device)

    # Transformer-aware configuration: used to apply parameter-group policies (e.g., LR scaling).
    is_transformer = hasattr(net, "transformer")

    # Split parameters into head vs backbone to enable differential learning rates / weight decay.
    head_params, backbone_params = _split_parameters(net)

    # Construct optimizer using the project’s policy (e.g., AdamW, parameter groups, LR scaling).
    optimizer = _build_optimizer(args, net, is_transformer, head_params, backbone_params)
    """
    # Ensure the model is on the correct device.
    net = net.to(device)
    
    # IMPORTANT (DataParallel):
    # If net is torch.nn.DataParallel, attributes such as `.transformer` live under net.module.
    # We unwrap ONLY for configuration/optimizer construction.
    net_cfg = net.module if isinstance(net, torch.nn.DataParallel) else net
    
    # Transformer-aware configuration: used to apply parameter-group policies (e.g., LR scaling).
    is_transformer = hasattr(net_cfg, "transformer")
    
    # Split parameters into head vs backbone to enable differential learning rates / weight decay.
    head_params, backbone_params = _split_parameters(net_cfg)
    
     # Construct optimizer (Logic is now encapsulated in the function)
    optimizer = _build_optimizer(args, net_cfg, is_transformer, head_params, backbone_params)

    scheduler, scheduler_type = _build_scheduler(
        args, optimizer, accum_steps, len(dataloader), args.base_lr
    )
    
    # ---- RUN PLAN (now you have optimizer + scheduler) ----
    """
    print_run_plan(
        args,
        train_loader=dataloader,
        val_loader=val_loader,
        model=net_cfg,              # unwrap DP for correct inspection
        optimizer=optimizer,
        scheduler=scheduler,
        train_tfms=getattr(dataloader.dataset, "transform", None),
        eval_tfms=getattr(val_loader.dataset, "transform", None),
    )
    """
    # ------------------------------------------------------------------------------------------------
    # AMP (Automatic Mixed Precision)
    # ------------------------------------------------------------------------------------------------
    # AMP should be explicitly controlled by args.amp (you said you already have it in CLI).
    # We also require CUDA and an actual CUDA device.
    use_amp = bool(getattr(args, "amp", False)) and bool(getattr(args, "cuda", False)) and (device.type == "cuda")
    
    # GradScaler dynamically scales the loss to prevent fp16 underflow.
    # If use_amp=False, this becomes a no-op wrapper.
    scaler = GradScaler(enabled=use_amp)

    # ------------------------------------------------------------------------------------------------
    # Ignite engines: training + inference
    # ------------------------------------------------------------------------------------------------

    # Training engine: runs forward/backward, applies accumulation and optimizer/scheduler stepping.
    trainer = Engine(
        _make_train_step(
            args=args,
            device=device,
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_type=scheduler_type,
            accum_steps=accum_steps,
            grad_clip=grad_clip,
            logger=logger,
            trial=trial,
            scaler=scaler,  # AMP scaler (no-op if disabled)
        )
    )

    # Inference engines: identical step function, distinct engines to keep metrics/state separated.
    evaluator = Engine(_make_inference_step(args, device, net))
    evaluator_test = Engine(_make_inference_step(args, device, net))

    # Persist Optuna trial handle into trainer state for handlers that need access.
    trainer.state.trial = trial

    # Attach metrics to all engines (train/val/test) so that `engine.state.metrics` is populated.
    _attach_metrics([trainer, evaluator, evaluator_test], args, device)

    # Attach per-epoch hooks that depend on the training engine lifecycle (e.g., accumulation bookkeeping).
    _attach_epoch_end_step(trainer, optimizer, accum_steps, scaler=scaler)

    # ------------------------------------------------------------------------------------------------
    # Epoch-end validation/test evaluation handler
    # ------------------------------------------------------------------------------------------------

    # Timestamp used for end-to-end wall-clock time reporting.
    start_training = timer()

    # Validation handler runs at each epoch end:
    #   - evaluates validation and test sets
    #   - updates training_state (best tracking, history)
    #   - applies early stopping logic
    #   - manages scheduler stepping for plateau schedulers (if applicable)
    validation_handler = _make_validation_handler(
        args,
        net,
        trainer,
        evaluator,
        evaluator_test,
        optimizer,
        scheduler,
        scheduler_type,
        val_loader,
        test_loader,
        early_stopper,
        start_training,
        training_state,
        device,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, validation_handler)

    # ------------------------------------------------------------------------------------------------
    # Checkpointing policy
    # ------------------------------------------------------------------------------------------------
    # Safe W&B run name: avoids AttributeError when wandb is disabled or not initialized
    run_name = getattr(getattr(wandb, "run", None), "name", "no_wandb")

    # Top-k checkpoints scored by validation accuracy.
    # Filename prefix uses model/backbone identifiers to keep runs organized in the same directory.
    handler = ModelCheckpoint(
        args.model_dir,
        "{}_{}".format(args.model, args.backbone),
        n_saved=10,
        create_dir=True,
        require_empty=False,
        score_function=lambda engine: engine.state.metrics["val_acc"],
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"model": net})

    # Best checkpoint: retains only the single best checkpoint for the current W&B run name.
    handler_best = ModelCheckpoint(
        args.model_dir,
        "{}".format(run_name),
        n_saved=1,
        create_dir=True,
        require_empty=False,
        score_function=lambda engine: engine.state.metrics["val_acc"],
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler_best, {"model": net})

    # Last checkpoint: always overwrites to keep the most recent state (useful for resuming/debugging).
    handler_last = ModelCheckpoint(
        args.model_dir,
        "{}".format(run_name),
        n_saved=1,
        create_dir=True,
        require_empty=False,
        global_step_transform=lambda *_: trainer.state.epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler_last, {"model": net})

    # ------------------------------------------------------------------------------------------------
    # Resume support (epoch and max_epochs overrides)
    # ------------------------------------------------------------------------------------------------

    # When resuming, force Ignite engines to start from a user-provided epoch and to respect
    # a user-provided maximum number of epochs.
    if args.resume:

        def start_epoch(engine):
            engine.state.epoch = args.epoch

        def max_epoch(engine):
            engine.state.max_epochs = args.max_epochs

        for engine in [trainer, evaluator, evaluator_test]:
            engine.add_event_handler(Events.STARTED, start_epoch)
            engine.add_event_handler(Events.STARTED, max_epoch)

    # Ensure a clean gradient state before training begins.
    optimizer.zero_grad()

    # Reinitialize timing if a logger is enabled (supports step-level timing logs downstream).
    if logger:
        start_training = timer()

    # ------------------------------------------------------------------------------------------------
    # Run training loop (with guaranteed W&B finalization)
    # ------------------------------------------------------------------------------------------------
    try:
        trainer.run(dataloader, max_epochs=args.max_epochs)
    finally:
        # Ensure W&B run is properly closed even if training terminates early or raises.
        if args.log_wandb and wandb.run is not None:
            wandb.finish()

    # ------------------------------------------------------------------------------------------------
    # Objective computation (final reporting)
    # ------------------------------------------------------------------------------------------------

    # Extract validation accuracy history and best value from the shared state.
    val_acc_history: List[float] = training_state["val_acc_history"]  # type: ignore[assignment]
    best_val_acc = float(training_state["best_val_acc"])

    # Provide a smoothed objective when sufficient epochs exist; otherwise fall back gracefully.
    if len(val_acc_history) >= 3:
        final_val_acc = sum(val_acc_history[-3:]) / 3.0
    elif len(val_acc_history) > 0:
        final_val_acc = sum(val_acc_history) / len(val_acc_history)
    else:
        final_val_acc = best_val_acc

    # Persist key results into Optuna for downstream analysis and reproducibility.
    if trial is not None:
        trial.set_user_attr("best_val_acc", float(best_val_acc))
        trial.set_user_attr("final_val_acc", float(final_val_acc))

    # Return objective value (used by sweeps and external callers).
    return final_val_acc
