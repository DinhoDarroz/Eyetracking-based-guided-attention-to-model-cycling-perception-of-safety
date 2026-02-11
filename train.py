# coding: utf-8

import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import os
from glob import glob
import pickle
import pandas as pd
import numpy as np
import wandb

from train_utils import (
    validate_and_normalize_args,
    resolve_backbone,
    )

from train_main_utils import (
    _seed_everything,
    initialize_logging,
    initialize_wandb,
    read_data,
    _boolish_series_to_bool_mask,
    _ensure_dir,
    _split_paths,
    _load_or_split,
    _print_filtered_dataset_summary,
    _print_image_overlap_stats,
    _print_has_eyetracker_by_split,
    _print_split_sizes,
    _print_label_distribution_by_split,
    _compute_and_attach_class_weights,
    _parse_gpu_ids,
    _select_device,
    _load_state_dict_safely,
    _resolve_gaze_grid,
    _build_transforms_and_specs,
    _build_dataloaders,
    _build_model,
    _maybe_wrap_dataparallel,
    _maybe_resume,
    _cleanup_between_trials,
    scale_lr_and_eta_min_by_unfrozen_blocks,
)

from scripts.train_script import train
import warnings
import gc
import timm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import os
from torchvision import models
from typing import Tuple, Dict, Any, Iterable, Set, Optional


warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None  # default='warn'

def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Training subjective safety",
        allow_abbrev=False
    )

    # -------------------- BOOLEAN FLAGS --------------------
    parser.add_argument("--cuda", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--multi_gpu", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--gpu_ids", type=str, default="0",help="Comma-separated GPU ids, e.g. '0,1'")
    parser.add_argument("--resume", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--finetune", "--ft", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--ties", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--log_console", nargs="?", const=True, default=True, type=str2bool)
    parser.add_argument("--log_wandb", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--full_accuracy", nargs="?", const=True, default=False, type=str2bool)

    # -------------------- AUGMENTATION --------------------
    parser.add_argument(
        "--augment",type=str,default="none",choices=["none", "light", "heavy"],
        help="Augmentation level for training: none | light | heavy. (Applied only when gaze alignment is OFF.)",
    )
    parser.add_argument("--use_class_weights", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--use_seg", nargs="?", const=True, default=False, type=str2bool)

    # -------------------- SCHEDULER -------------------------
    parser.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=[
            "none",
            "warmup_cosine",
            "cosine",
            "onecycle",
            "warm_restarts",
            "plateau",
        ],
        help=(
            "LR scheduler to use:\n"
            "  none           : constant LR\n"
            "  warmup_cosine  : linear warmup + cosine decay (uses --warmup_frac, --eta_min)\n"
            "  cosine         : cosine decay over total iters (uses --eta_min)\n"
            "  onecycle       : OneCycleLR (uses --warmup_frac as pct_start)\n"
            "  warm_restarts  : CosineAnnealingWarmRestarts (uses --T_0, --T_mult, --eta_min)\n"
            "  plateau        : ReduceLROnPlateau on validation accuracy (uses --plateau_patience, --plateau_factor, --plateau_min_lr)"
        ),
    )

    # -------------------- WARMUP / COSINE OPTIONS -------------------------
    sched_warm_group = parser.add_argument_group("Warmup / cosine scheduler options")
    sched_warm_group.add_argument("--warmup_frac", type=float, default=0.3, help="Fraction of total optimizer steps used for warmup. Used by warmup_cosine and onecycle. Ignored by none, cosine, warm_restarts, plateau.")
    sched_warm_group.add_argument("--eta_min", type=float, default=1e-6, help="Minimum LR for cosine-style schedulers. Used by warmup_cosine, cosine, warm_restarts. Ignored by none, onecycle, plateau.")

    # -------------------- WARM RESTART OPTIONS -------------------------
    sched_wr_group = parser.add_argument_group("Warm restarts options")
    sched_wr_group.add_argument("--T_0", type=int, default=10, help="Initial optimizer steps before first restart (warm_restarts only).")
    sched_wr_group.add_argument("--T_mult", type=int, default=2, help="Multiplicative factor for cycle length (warm_restarts only).")

    # -------------------- PLATEAU OPTIONS -------------------------
    sched_plateau_group = parser.add_argument_group("Plateau scheduler options")
    sched_plateau_group.add_argument("--plateau_patience", type=int, default=2, help="Epochs with no validation improvement before reducing LR (plateau only).")
    sched_plateau_group.add_argument("--plateau_factor", type=float, default=0.5, help="LR reduction factor for ReduceLROnPlateau (plateau only).")
    sched_plateau_group.add_argument("--plateau_min_lr", type=float, default=1e-7, help="Minimum LR for ReduceLROnPlateau (plateau only).")
    
    # -------------------- EARLY STOPPING -------------------------
    es_group = parser.add_argument_group("Early stopping")
    es_group.add_argument("--early_stop", nargs="?", const=True, default=False, type=str2bool, help="Enable early stopping based on a validation metric.")
    es_group.add_argument("--early_stop_metric", type=str, default="accuracy_validation",
                          help="Metric name to monitor (e.g., accuracy_validation, loss_validation).")
    es_group.add_argument("--early_stop_mode", type=str, default="max", choices=["max", "min"],
                          help="max: higher is better (accuracy); min: lower is better (loss).")
    es_group.add_argument("--early_stop_patience", type=int, default=3,
                          help="Stop after this many epochs without improvement.")
    es_group.add_argument("--early_stop_min_delta", type=float, default=0.0,
                          help="Minimum change to qualify as an improvement.")
    es_group.add_argument("--early_stop_start_epoch", type=int, default=1,
                          help="Do not early-stop before this epoch.")

    # -------------------- GAZE & CITY FILTERS ----------------
    parser.add_argument(
        "--gaze_mode",
        type=str,
        default="disable",
        help=(
            "Gaze behavior:\n"
            "  disable    : no gaze loading, no KL diagnostics, no injection, no attention hooks\n"
            "  diag       : KL computed for diagnostics only (no gradient contribution)\n"
            "  guide      : gaze injected; KL computed for diagnostics only\n"
            "  egvit     : EG-ViT masking (mask patch tokens by gaze during training; no KL)\n"
            "  align      : KL added to total loss (attn_w * KL)\n"
            "  align+gaze : gaze injected and KL added to total loss (attn_w * KL)\n"
            "Legacy aliases: off->diag, align+guide->align+gaze\n"
        ),
    )

    parser.add_argument(
        "--eyetracker_filter",
        type=str,
        default="all",
        choices=["all", "only"],
        help=(
            "Dataset filter on has_eyetracker. "
            "all keeps all rows; only keeps rows where has_eyetracker==True."
        ),
    )

    parser.add_argument("--attention_mode", type=str, default="last", choices=["last", "rollout", "topk"],
    help=(
        "How to extract transformer attention maps:\n"
        "  last    : use CLS→patch attention from the last transformer block\n"
        "  rollout : rollout attention across all blocks (identity-augmented)\n"
        "  topk    : last-block CLS→patch attention, sparsified to top-k tokens"
        ),
    )
    parser.add_argument("--attn_topk", type=int,default=None,
    help=(
        "Number of patch tokens to keep when --attention_mode=topk. "
        "If None, all tokens are used."
        ),
    )

    parser.add_argument("--cities", type=str, default="all")

    # -------------------- EG-ViT (GAZE MASKING) ----------------
    egvit_group = parser.add_argument_group("EG-ViT gaze masking options")
    egvit_group.add_argument("--egvit_mask_type",type=str,default="separated",choices=["separated", "focused"],help="Mask construction strategy for --gaze_mode=egvit.",)
    egvit_group.add_argument("--egvit_keep_ratio",type=float,default=0.25,help="Fraction of patch tokens to keep for separated masks (e.g., 0.25 keeps top 25% patches).",)
    egvit_group.add_argument("--egvit_focus_hw",type=int,nargs=2,default=(7, 7),help="Focused mask window size in patch units: H W (used only when --egvit_mask_type=focused).",)
    egvit_group.add_argument("--egvit_drop_prob",type=float,default=0.0,help="Probability to disable EG-ViT masking per-sample during training (stochastic robustness).",)
    egvit_group.add_argument("--egvit_train_only",nargs="?",const=True,default=True,type=str2bool,help="If True, apply EG-ViT masking only during training; eval() becomes a vanilla ViT forward.",
    )
    # -------------------- GAZE GUIDANCE (Guide mode) --------------------
    parser.add_argument("--guidance_drop_prob", type=float, default=0.0, help="Stochastic gaze disable prob (guide mode).")
    parser.add_argument("--guidance_strength", type=float, default=2.5, help="Scale applied to injected guidance residual.")
    parser.add_argument("--guidance_bottleneck_dim", type=int, default=20, help="GII bottleneck dim (d').")
    parser.add_argument("--guidance_gaze_hidden_dim", type=int, default=30, help="Gaze token embedding dim (dg).")
    parser.add_argument("--guidance_conv_hidden_channels", type=int, default=64, help="GFF conv hidden channels.")
    parser.add_argument("--guide_train_only",nargs="?",const=True,default=False,type=str2bool,help="When True, disables Guide/GII gaze injection during eval() (validation/test).",)


    # -------------------- LR & OPTIMIZATION ------------------
    parser.add_argument("--base_lr", type=float, default=5e-6)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--k", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--rank_dropout", type=float, default=0.3)
    parser.add_argument("--cross_dropout", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    # -------------------- PATHS & BASIC PARAMS ---------------
    parser.add_argument("--comparisons", type=str, default="comparisons_df.pickle")
    parser.add_argument("--dataset", type=str, default="images/")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--model_dir", type=str, default="models/")
    parser.add_argument("--wandb_project", type=str, default="SubjectiveCyclingSafety")

    # -------------------- MODEL SETTINGS ---------------------
    parser.add_argument("--model", type=str, default="rcnn",
                        choices=["rsscnn", "sscnn", "rcnn"])
    
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov3_vitb16",
        choices=[
            # --- Strict 224x224 (≈14x14 token grid for ViT-B/16) ---
            "dinov3_vitb16",             # 1. DINOv3 (Dense Specialist)
            "beitv2_base_patch16_224",   # 2. BEiT v2 (Masked Modeling / Structure)
            "deit3_base_patch16_224",    # 3. DeiT III (Supervised Benchmark)
            "siglip_base_patch16_224",   # 4. SigLIP (Semantic Expert)
            "vit_base_patch16_clip_224", # 5. CLIP ViT-B/16 (Robust / Wildcard)
    
            # --- Modern High-Performance Alternates ---
            "dinov2_base",               # NEW: DINOv2 (no registers)
            "dinov2_reg_base",           # DINOv2 + registers
            "eva02_base",                # EVA-02 (448px recommended)
            "convnext_base",             # ConvNeXt (Modern CNN)
    
            # --- Canonical / Legacy Transformers ---
            "vit_base_patch16_224",      # NEW: Original ViT-B/16 (21k -> 1k)
            "vit_base_dino",             # DINO v1
            "vit_small",
            "deit_base",
            "deit_small",
            "deit_tiny",
            "deit_base_distilled",
    
            # --- CNN Baselines ---
            "alex",
            "vgg",
            "dense",
            "resnet",
        ],
        help="Model backbone to use. Default: dinov3_vitb16",
    )

    # === POOLING ARGUMENTS ===
    parser.add_argument(
        "--pooling",
        type=str,
        default="cls",
        choices=[
            "cls",              # CLS token only
            "mean",             # mean over CLS only (same as cls, explicit semantic)
            "patch_mean",       # mean over patch tokens
            "reg_mean",         # mean over register tokens
            "prefix_mean",      # mean over CLS + registers
            "cls_reg_concat",   # concat(CLS, mean(registers))
            "cls_reg_add",      # CLS + mean(registers)
            "concat",           # concat(CLS, patch_mean)
            "topk",             # mean of top-k patch tokens by norm
            "max",
            "cls_max_concat",
        ],
        help=(
            "Feature pooling strategy for transformer backbones. "
            "Options: "
            "cls | mean | patch_mean | reg_mean | prefix_mean | "
            "cls_reg_concat | cls_reg_add | concat | topk"
        ),
    )
    
    parser.add_argument(
        "--pool_k",
        type=int,
        default=10,
        help="Number of patch tokens used when --pooling topk is selected",
    )
    

    # -------------------- LOSSES ------------------------------
    parser.add_argument("--rank_w", type=float, default=1.0)
    parser.add_argument("--ties_w", type=float, default=1.0)
    parser.add_argument("--ranking_margin", type=float, default=0.7)
    parser.add_argument("--ranking_margin_ties", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--attn_w", type=float, default=1.0)
    parser.add_argument("--gaze_root", type=str, default="Eyetracker_attention_maps")
    parser.add_argument("--gaze_map_size", default="auto", help="Gaze map size selection: 'auto' or integer (e.g. 14, 16).")


    # -------------------- MISC -------------------------------
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--num_ft_blocks", type=int, default=1, help="Number of last transformer blocks to unfreeze when --finetune is set.")
    parser.add_argument(
        "--cnn_pool",
        type=str,
        default="flatten",
        choices=["gap", "flatten"],
        help="CNN head input: 'gap' uses global average pooling (small heads); "
             "'flatten' uses flattened spatial grid (VGG-style, huge heads).",
    )
    parser.add_argument("--backbone_freeze_epochs", type=int, default=4, help="Freeze backbone for first N epochs (requires --finetune).")
    
    return parser



# -------------------------------------------------------------------------------------------------
# Main entrypoint 
# -------------------------------------------------------------------------------------------------

def run_training_with_args(args, trial=None):
    """
    Runs one full training session given a filled args Namespace.
    Returns best validation accuracy from train().
    """

    # ==============================================================================================
    # 0) ARG VALIDATION / NORMALIZATION
    # ==============================================================================================
    validate_and_normalize_args(args, strict=False, verbose=True)
    #args.base_lr, args.eta_min = scale_lr_and_eta_min_by_unfrozen_blocks(args, lr_01=3e-4, lr_other=2e-5)

    print("=== Args ===")
    print(args, "\n")

    # ==============================================================================================
    # 1) REPRODUCIBILITY (SEEDS)
    # ==============================================================================================
    _seed_everything(args.seed, deterministic=False)


    # ==============================================================================================
    # 2) LOGGING / WANDB
    # ==============================================================================================
    logger = initialize_logging()

    if getattr(args, "log_wandb", False):
        initialize_wandb(args)

    # ==============================================================================================
    # 3) DATA INGESTION (POST-FILTERING, PRE-SPLIT)
    # ==============================================================================================
    print("Reading input data...")
    comparisons_df = read_data(args)

    # Summary describes the effective dataset used for splitting/training
    _print_filtered_dataset_summary(args, comparisons_df)

    # ==============================================================================================
    # 4) TRAIN / VAL / TEST SPLITS
    # ==============================================================================================
    X_train, X_val, X_test = _load_or_split(
        df=comparisons_df,
        seed=args.seed,
        comparisons_path=args.comparisons,
        splits_dir = "splits_eye",
        train_pct=0.7,
        val_pct=0.1,
        test_pct=0.2,
        load_if_exists=True,   # loads if files exist, otherwise splits
        save_splits=False,
    )


    # Sanity checks: image overlap and optional eyetracker balance across splits
    _print_image_overlap_stats(X_train, X_val, X_test)
    _print_has_eyetracker_by_split(X_train, X_val, X_test)
    _print_split_sizes(comparisons_df, X_train, X_val, X_test)

    # ==============================================================================================
    # 5) LABEL DISTRIBUTION + CLASS WEIGHTS (TRAIN ONLY)
    # ==============================================================================================
    _print_label_distribution_by_split(X_train, X_val, X_test)
    _compute_and_attach_class_weights(args, X_train)

    # ==============================================================================================
    # 6) BACKBONE RESOLUTION + TRANSFORMS
    # ==============================================================================================
    # Returns:
    #  - backbone_model / model_specs: resolved backbone + preprocessing specs
    #  - train_tfms / eval_tfms: training/eval transform pipelines
    #  - use_gaze_requested: whether gaze mode is enabled (args.gaze_mode != "disable")
    #  - use_gaze_loss: whether attention/gaze supervision is active for the chosen model config
    #  - is_cnn_backbone: whether the backbone family uses torchvision CNN path
    backbone_model, model_specs, train_tfms, eval_tfms, use_gaze_requested, use_gaze_loss, is_cnn_backbone = (
        _build_transforms_and_specs(args)
    )

    # ==============================================================================================
    # 7) DATALOADERS
    # ==============================================================================================
    train_loader, val_loader, test_loader = _build_dataloaders(
        args=args,
        logger=logger,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        train_tfms=train_tfms,
        eval_tfms=eval_tfms,
    )

    # ==============================================================================================
    # 8) DEVICE SELECTION
    # ==============================================================================================
    # Returns:
    #  - device: torch.device('cpu') or torch.device('cuda:n')
    #  - gpu_ids: list of GPU ids for DataParallel when enabled
    device, gpu_ids = _select_device(args)
    print("Device:", device)

    # ==============================================================================================
    # 9) MODEL INSTANTIATION
    # ==============================================================================================
    # Transformer path uses resolved backbone_model.
    # CNN path uses torchvision factory inside _build_model.
    net = _build_model(
        args=args,
        backbone_model=backbone_model,
        use_gaze_loss=use_gaze_loss,
        is_cnn_backbone=is_cnn_backbone,
    )

    # Move model parameters to the selected device before wrapping
    net.to(device)

    # Optional DataParallel wrapper for multi-GPU
    net = _maybe_wrap_dataparallel(args, net, gpu_ids)

    # Optional checkpoint restore (handles DP/non-DP key formats)
    _maybe_resume(args, net, device)

    # ==============================================================================================
    # 10) WANDB RUN NAME (OPTIONAL)
    # ==============================================================================================
    run_name = ""
    if getattr(args, "log_wandb", False) and wandb is not None and wandb.run is not None:
        run_name = wandb.run.name

    print("\n[Wandb]")
    print(f"  Run Name     : {run_name}")

    if trial is not None:
        trial.set_user_attr("wandb_run_name", run_name)

    # ==============================================================================================
    # 11) TRAIN LOOP
    # ==============================================================================================
    best_val_acc = train(
        device=device,
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        args=args,
        logger=logger,
        trial=trial,
        train_df=X_train,
        val_df=X_val,
        test_df=X_test,
        train_tfms=train_tfms,
        eval_tfms=eval_tfms,
    )

    # ==============================================================================================
    # 12) CLEANUP (BETWEEN TRIALS / RUNS)
    # ==============================================================================================
    _cleanup_between_trials(args, net, train_loader, val_loader, test_loader)

    return best_val_acc

if __name__ == "__main__":
    args = arg_parse().parse_args()
    run_training_with_args(args)
