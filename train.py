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
from data import (
    ComparisonsDataset,
    _get_interp_mode,
    build_eval_transforms,
    AUG_PRESETS,
    Augmentation,
)

from train_utils import (
    validate_and_normalize_args,
    resolve_backbone,
    infer_vit_grid_size,
    compute_class_weights_from_df,
    print_run_plan,
    )

import logging
from datetime import date
from scripts.train_script import train
import warnings
import gc
import timm
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import os
from torchvision import models


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
    parser.add_argument("--amp", nargs="?", const=True, default=False, type=str2bool, help="Enable Automatic Mixed Precision (AMP).")
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
    "--gaze",
        default="off",
        choices=["off", "use", "use_nobp", "only"],
        help=(
            "Gaze supervision mode:\n"
            "  off       : do not use gaze loss\n"
            "  use       : use gaze KL in total loss (backprop)\n"
            "  use_nobp  : compute/log gaze KL but DO NOT backprop KL\n"
            "  only      : train using gaze KL only (debug/ablation)\n"
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
    parser.add_argument("--gaze_subdir_fmt", default="{s}x{s}", help="Subfolder format under gaze_root (default: '14x14', '16x16', ...).")


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


def read_data(args):
    # ------------------------------------------------------------------
    # 0) LOAD
    # ------------------------------------------------------------------
    try:
        comparisons_df = pickle.load(open(args.comparisons, "rb"))
    except Exception:
        comparisons_df = pd.read_pickle(args.comparisons)

    print(f"[read_data] Loaded raw rows: {len(comparisons_df):,}")
    print(f"[read_data] Raw columns: {list(comparisons_df.columns)}")

    # ------------------------------------------------------------------
    # 1) KEEP ONLY COLUMNS THAT EXIST
    # ------------------------------------------------------------------
    cols_we_need = [
        "score",
        "image_l", "image_r",
        "dataset",
        "has_eyetracker",
        "npy_file_l", "npy_file_r",
        "survey_id", "trial_id",
    ]
    existing_cols = [c for c in cols_we_need if c in comparisons_df.columns]
    comparisons_df = comparisons_df[existing_cols].copy()

    print(f"[read_data] Columns kept: {existing_cols}")
    print(f"[read_data] Rows after column selection: {len(comparisons_df):,}")

    # ------------------------------------------------------------------
    # 2) CITY / DATASET FILTER
    # ------------------------------------------------------------------
    if "dataset" in comparisons_df.columns:
        print("\n[read_data] Available datasets:")
        print(comparisons_df["dataset"].value_counts())

        cities_arg = getattr(args, "cities", "all")
        if cities_arg.lower() != "all":
            selected_cities = [c.strip() for c in cities_arg.split(",") if c.strip()]
            print(f"\n[read_data] Filtering to cities: {selected_cities}")

            before = len(comparisons_df)
            comparisons_df = comparisons_df[
                comparisons_df["dataset"].isin(selected_cities)
            ].copy()
            after = len(comparisons_df)

            print(f"[read_data] Rows after city filter: {after}/{before}")
            if after == 0:
                print("[WARN] City filter removed all rows.")

    # ------------------------------------------------------------------
    # 3) IMAGE FILENAME NORMALIZATION
    # ------------------------------------------------------------------
    for side in ("image_l", "image_r"):
        if side in comparisons_df.columns:
            comparisons_df[side] = (
                comparisons_df[side]
                .astype(str)
                .apply(lambda x: x if x.lower().endswith(".jpg") else f"{x}.jpg")
            )

    # ------------------------------------------------------------------
    # 4) GAZE FLAG NORMALIZATION + FILTERING
    # ------------------------------------------------------------------
    if "has_eyetracker" in comparisons_df.columns:
        comparisons_df["has_eyetracker"] = (
            comparisons_df["has_eyetracker"]
            .replace({"True": True, "False": False, "true": True, "false": False})
            .fillna(False)
            .astype(bool)
        )

        print(
            "\n[read_data] has_eyetracker distribution:",
            comparisons_df["has_eyetracker"].value_counts(dropna=False).to_dict(),
        )

        if args.gaze == "only":
            before = len(comparisons_df)
            comparisons_df = comparisons_df[comparisons_df["has_eyetracker"]].copy()
            after = len(comparisons_df)

            print(f"[read_data] Rows after gaze=only filter: {after}/{before}")
            if after == 0:
                print("[WARN] --gaze only removed all rows.")

    # ------------------------------------------------------------------
    # 5) LABEL HANDLING / TIES
    # ------------------------------------------------------------------
    if "score" not in comparisons_df.columns:
        raise ValueError("[read_data] Missing required column: 'score'")

    if not args.ties:
        before = len(comparisons_df)
        comparisons_df = comparisons_df[comparisons_df["score"] != 0].copy()
        after = len(comparisons_df)

        print(f"\n[read_data] Rows after ties=False filter: {after}/{before}")
        if after == 0:
            print("[WARN] ties=False removed all rows.")

        comparisons_df["score_classification"] = comparisons_df["score"].replace(
            {-1: 0, +1: 1}
        )
    else:
        comparisons_df["score_classification"] = comparisons_df["score"] + 1

    # ------------------------------------------------------------------
    # 6) FINAL SANITY CHECK
    # ------------------------------------------------------------------
    print("\n[read_data] FINAL ROW COUNT:", len(comparisons_df))
    if len(comparisons_df) > 0:
        print(
            "[read_data] Final score distribution:",
            comparisons_df["score"].value_counts(dropna=False).to_dict(),
        )
    else:
        print("[FATAL] Dataset is EMPTY after all filters.")

    return comparisons_df



def initialize_logging():
    """Initialize run logs."""
    if 'logs' not in os.listdir():
        os.mkdir('logs')
    logging.basicConfig(format='%(message)s', filename=f'logs/{date.today().strftime("%d-%m-%Y")}.log')
    logger = logging.getLogger('timer')
    logger.setLevel(logging.INFO)
    #logger.info('HELLO')
    return logger


def initialize_wandb(args):
    """Initialize WandB run logs."""
    checkpoint_path = (
        os.path.join(args.model_dir, f"{args.resume_checkpoint}")
        if getattr(args, "resume_checkpoint", None)
        else None
    )

    wandb.init(
        project=args.wandb_project,
        config={
            "early_stop": getattr(args, "early_stop", False),
            "early_stop_metric": getattr(args, "early_stop_metric", "accuracy_validation"),
            "early_stop_mode": getattr(args, "early_stop_mode", "max"),
            "early_stop_patience": getattr(args, "early_stop_patience", 3),
            "early_stop_min_delta": getattr(args, "early_stop_min_delta", 0.0),
            "early_stop_start_epoch": getattr(args, "early_stop_start_epoch", 1),
            "dataset": args.comparisons,
            "gaze_mode": args.gaze,
            "ties": args.ties,
            "ties_w": args.ties_w,
            "rank_w": args.rank_w,
            "rank_margin": args.ranking_margin,
            "rank_margin_ties": args.ranking_margin_ties,
            "seed": args.seed,
            "epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "architecture_backbone": args.backbone,
            "architecture_model": args.model,
            "finetune_backbone": args.finetune,
            "num_ft_blocks": getattr(args, "num_ft_blocks", None),
            "base_lr": args.base_lr,
            "weight_decay": args.weight_decay,
            #"backbone_lr_scale": getattr(args, "backbone_lr_scale", None),
            "scheduler": args.scheduler,
            "warmup_frac": getattr(args, "warmup_frac", None),
            "eta_min": getattr(args, "eta_min", None),
            "T_0": getattr(args, "T_0", None),
            "T_mult": getattr(args, "T_mult", None),
            "rank_dropout": getattr(args, "rank_dropout", None),
            "cross_dropout": getattr(args, "cross_dropout", None),
            "label_smoothing": getattr(args, "label_smoothing", None),
            "use_class_weights": getattr(args, "use_class_weights", None),
            "augment": args.augment,
            "resume": args.resume,
            "resume_epoch": args.epoch,
            "checkpoint": checkpoint_path,
        },
    )

    wandb.define_metric("iteration")
    wandb.define_metric("epoch")
    wandb.define_metric("loss_train", step_metric="iteration")
    wandb.define_metric("loss_validation", step_metric="epoch")
    wandb.define_metric("loss_test", step_metric="epoch")
    wandb.define_metric("accuracy_validation", step_metric="epoch")
    wandb.define_metric("accuracy_test", step_metric="epoch")
    wandb.define_metric("max_accuracy_train", step_metric="epoch")
    wandb.define_metric("max_accuracy_validation", step_metric="epoch")
    wandb.define_metric("max_accuracy_test", step_metric="epoch")


def run_training_with_args(args, trial=None):
    """
    Run ONE full training session given a filled args Namespace.
    Returns best validation accuracy from train().
    """

    # Central consistency / dependency checks
    validate_and_normalize_args(args, strict=False, verbose=True)

    #args.batch_size = resolve_batch_size(args)
    print("=== Args ===")
    print(args, "\n")

    # =============================================================================================== #
    # 0) PARSE & SEED
    # =============================================================================================== #
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # =============================================================================================== #
    # 1) LOGGING / WANDB
    # =============================================================================================== #
    logger = initialize_logging()
    if args.log_wandb:
        initialize_wandb(args)

    # =============================================================================================== #
    # 2) DATA: LOAD + SUMMARIZE (POST-FILTERING, PRE-SPLIT)
    # =============================================================================================== #
    print("Reading input data...")
    comparisons_df = read_data(args)

    # This summary is intentionally focused on "what data will be used from now on".
    # (Detailed plan/architecture is printed later by RUN PLAN.)
    print("\n=== Effective Dataset (after all filters, before split) ===")
    print(f"Comparisons file : {args.comparisons}")
    print(f"Cities requested : {args.cities}")
    print(f"Gaze mode        : {args.gaze}  (rows kept depend on has_eyetracker + gaze setting)")
    print(f"Ties enabled     : {args.ties}  (ties=False removes score==0 rows)")
    print(f"Final row count  : {len(comparisons_df):,}")

    # Score distribution AFTER filtering (this is the distribution it's suupose to train on)
    if "score" in comparisons_df.columns:
        print("\nScore distribution (post-filtering):")
        score_counts = comparisons_df["score"].value_counts().sort_index()
        total_rows = len(comparisons_df)
        for s, c in score_counts.items():
            print(f"  score={s:>2}: {c:>6,} ({(100.0*c/total_rows):5.2f}%)")

    # Eyetracker availability AFTER filtering (helps explain what gaze mode kept/removed)
    if "has_eyetracker" in comparisons_df.columns:
        print("\nEyetracker availability (post-filtering):")
        et_counts = comparisons_df["has_eyetracker"].value_counts(dropna=False)
        for k, v in et_counts.items():
            print(f"  {str(k):>5}: {v:>6,} ({(100.0*v/len(comparisons_df)):5.2f}%)")

    # Minimal sanity peek (do not spam; run plan later covers the rest)
    print("\nExample rows (post-filtering):")
    print(comparisons_df.head(3))
    print("========================================================\n")

    # =============================================================================================== #
    # 3) TRAIN/VAL/TEST SPLIT
    # =============================================================================================== #
    """      
    # 1. Define the Grouping Variable
    # Ideally 'trial_id' represents a unique user session or trip. 
    # If 'trial_id' is not spatial, use the image filename itself to be 100% strict.
    groups = comparisons_df['survey_id'] 
    
    # Check if we have enough groups
    n_groups = len(groups.unique())
    print(f"Splitting based on {n_groups} unique trials/groups to prevent leakage.")
    
    # 2. Perform Grouped Split (Train vs Test)
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(comparisons_df, groups=groups))
    
    X_train = comparisons_df.iloc[train_idx]
    X_test = comparisons_df.iloc[test_idx]
    
    # 3. Perform Grouped Split (Train vs Val) - reusing the logic on X_train
    # We need to re-extract groups for the new X_train subset
    train_groups = X_train['survey_id']
    splitter_val = GroupShuffleSplit(test_size=0.13, n_splits=1, random_state=args.seed)
    sub_train_idx, sub_val_idx = next(splitter_val.split(X_train, groups=train_groups))
     
    X_val = X_train.iloc[sub_val_idx]
    X_train = X_train.iloc[sub_train_idx]
    """    
    X_train, X_test = train_test_split(comparisons_df, test_size=0.2, random_state=args.seed)
    X_train, X_val  = train_test_split(X_train       , test_size=0.13, random_state=args.seed)
    
    splits_dir = "splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    split_prefix = os.path.splitext(os.path.basename(args.comparisons))[0]
    
    train_path = os.path.join(splits_dir, f"{split_prefix}_train.pkl")
    val_path   = os.path.join(splits_dir, f"{split_prefix}_val.pkl")
    test_path  = os.path.join(splits_dir, f"{split_prefix}_test.pkl")
    
    X_train.to_pickle(train_path)
    X_val.to_pickle(val_path)
    X_test.to_pickle(test_path)
    
    print("\n[SPLITS] Saved train/val/test splits:")
    print(" -", train_path)
    print(" -", val_path)
    print(" -", test_path)
    
    total = len(comparisons_df)
    print("=== Splits (on the filtered dataset above) ===")
    print(f"- Train: {len(X_train):,}  [{len(X_train)/total:.2%}]")
    print(f"- Val  : {len(X_val):,}  [{len(X_val)/total:.2%}]")
    print(f"- Test : {len(X_test):,}  [{len(X_test)/total:.2%}]")
    print("========================================================\n")

    # =============================================================================================== #
    # 3b) LABEL DISTRIBUTION PER SPLIT + CLASS WEIGHTS (TRAIN ONLY)
    # =============================================================================================== #
    print("=== Label distribution per split (score, after filtering) ===")
    for part_name, df in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        counts = df["score"].value_counts().sort_index()
        total_part = len(df)
        print(f"- {part_name}: {total_part:,} samples")
        for cls_val, cls_count in counts.items():
            pct = 100.0 * cls_count / total_part
            print(f"    score={cls_val:>2d}: {cls_count:>6,} ({pct:5.2f}%)")
    print("============================================================")

    # Class weights are computed ONLY from the training split
    args.class_weights = compute_class_weights_from_df(
        X_train["score_classification"],
        use_ties=args.ties,
        enable_weights=args.use_class_weights,
    )

    # Optional: a compact confirmation (no duplication with RUN PLAN)
    if args.use_class_weights and args.class_weights is not None:
        cw = args.class_weights.detach().cpu().numpy().tolist()
        print(f"Class weights: ON  (computed from Train split) → {cw}")
    else:
        print("Class weights: OFF")
    print()

    # =============================================================================================== #
    # 4) TRANSFORMS & MODEL CONFIG
    # =============================================================================================== #
    # Transformer backbones:
    #   - resolve backbone + timm preprocessing specs
    #   - infer ViT token grid size for gaze alignment (or override via --gaze_map_size)
    #
    # CNN backbones:
    #   - use fixed ImageNet-style SPECS (no timm resolve, no grid inference)
    # =============================================================================================== #
    
    # ------BACKBONE FAMILIES--------
    TRANSFORMER_BACKBONES = [
        "dinov3_vitb16",
        "beitv2_base_patch16_224",
        "deit3_base_patch16_224",
        "siglip_base_patch16_224",
        "vit_base_patch16_clip_224",
        "dinov2_base",
        "dinov2_reg_base",
        "eva02_base",
        "vit_base_patch16_224",
        "vit_base_dino",
        "vit_small",
        "deit_base",
        "deit_small",
        "deit_tiny",
        "deit_base_distilled",
    ]
    
    CNN_BACKBONES = ["alex", "vgg", "dense", "resnet"]
    is_cnn_backbone = str(args.backbone).lower().strip() in CNN_BACKBONES
    
    # Fixed preprocessing for CNNs (exactly as requested)
    SPECS = {
        "input_size": (3, 224, 224),
        "crop_pct": 0.875,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "interpolation": "bilinear",
    }
    
    # 1) Resolve specs + (optionally) build transformer backbone for grid inference
    if is_cnn_backbone:
        backbone_model = None  # CNN path later uses torchvision models; no ViT grid exists here
        model_specs = {
            "alias": args.backbone,
            "timm_id": None,
            **SPECS,
            "img_size": int(SPECS["input_size"][-1]),
        }
    
        # For CNNs we cannot infer a token grid; use CLI override if provided,
        # otherwise default to 14 (works with your gaze folders like 14x14 by default).
        if str(args.gaze_map_size).lower() != "auto":
            forced = int(args.gaze_map_size)
            grid_h, grid_w = forced, forced
        else:
            grid_h, grid_w = 14, 14
    
    else:
        # Transformer path: resolve backbone + preprocessing specs once
        backbone_model, model_specs = resolve_backbone(args.backbone, pretrained=True, strict=True)
    
        # Determine gaze grid size from patch embedding
        grid_h, grid_w = infer_vit_grid_size(backbone_model, model_specs)
    
        # Optional override from CLI
        if str(args.gaze_map_size).lower() != "auto":
            forced = int(args.gaze_map_size)
            grid_h, grid_w = forced, forced
    
    args.gaze_grid_size = (int(grid_h), int(grid_w))
    args.gaze_map_size_int = int(grid_h)
    
    # 2) Decide whether gaze is enabled for alignment / supervision
    use_gaze_requested = (str(args.gaze).lower().strip() != "off")
    
    # Keep model-specific gaze-loss logic (independent of transform policy)
    use_gaze_loss = (
        args.model == "rsscnn"
        and use_gaze_requested
        and float(args.attn_w) > 0.0
    )
    
    # 3) Build eval transforms (always deterministic; optionally align gaze)
    eval_tfms, eval_meta = build_eval_transforms(
        model_specs,
        gaze_grid_size=args.gaze_grid_size,
        enable_gaze=use_gaze_requested,
    )
    
    # 4) Build training transforms
    augment_level = str(getattr(args, "augment", "none")).lower().strip()
    if augment_level not in ("none", "light", "heavy"):
        augment_level = "none"
    
    # Augmentation is not allowed when gaze is enabled
    if use_gaze_requested:
        train_tfms = eval_tfms
        train_meta = {
            "train_policy": "deterministic (same as eval)",
            "augment": "none",
            "forced_deterministic_reason": "gaze_enabled",
        }
    else:
        if augment_level == "none":
            train_tfms = eval_tfms
            train_meta = {
                "train_policy": "deterministic (same as eval)",
                "augment": "none",
            }
        else:
            preset = AUG_PRESETS[augment_level]
            train_tfms = Augmentation(
                augment=True,
                ties=bool(getattr(args, "ties", True)),
                resize_short=int(eval_meta["resize_dim"]),
                out_size=int(eval_meta["target_crop"]),
                interpolation=_get_interp_mode(eval_meta["interpolation"]),
                mean=tuple(eval_meta["mean"]),
                std=tuple(eval_meta["std"]),
                **preset,
            )
            train_meta = {
                "train_policy": f"pairwise augmentation ({augment_level})",
                "augment": augment_level,
                "params": dict(preset),
            }
    
    # Expected model image size (for logging / checks)
    args.expected_img_size = int(model_specs.get("img_size", model_specs["input_size"][-1]))
    
    # Record transform configuration for reproducibility
    args.transforms_meta = {
        "backbone": args.backbone,
        "backbone_family": "cnn" if is_cnn_backbone else "transformer",
        "model_specs": dict(model_specs),
        "gaze": {
            "requested": bool(use_gaze_requested),
            "use_gaze_loss": bool(use_gaze_loss),
            "grid_size": tuple(args.gaze_grid_size),
        },
        "eval": dict(eval_meta),
        "train": dict(train_meta),
        "train_transform_class": train_tfms.__class__.__name__,
        "eval_transform_class": eval_tfms.__class__.__name__,
    }


    # =============================================================================================== #
    # 5) DATA LOADERS
    # =============================================================================================== #
    use_gaze = (args.gaze != "off")

    train_set = ComparisonsDataset(
        dataframe=X_train,
        root_dir=args.dataset,
        transform=train_tfms,
        logger=logger,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
        map_size=args.gaze_map_size_int,
        gaze_subdir_fmt=args.gaze_subdir_fmt,
    )
    
    val_set = ComparisonsDataset(
        dataframe=X_val,
        root_dir=args.dataset,
        transform=eval_tfms,
        logger=logger,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
        map_size=args.gaze_map_size_int,
        gaze_subdir_fmt=args.gaze_subdir_fmt,
    )
    
    test_set = ComparisonsDataset(
        dataframe=X_test,
        root_dir=args.dataset,
        transform=eval_tfms,
        logger=logger,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
        map_size=args.gaze_map_size_int,
        gaze_subdir_fmt=args.gaze_subdir_fmt,
    )


    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # =============================================================================================== #
    # 6) DEVICE & MODEL
    # =============================================================================================== #
    # Define cpu/gpu device
    if args.cuda:
        assert torch.cuda.is_available(), "ERROR: --cuda was passed but CUDA is not available."
        if args.multi_gpu:
            gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
            assert len(gpu_ids) >= 2, "--multi_gpu requires at least 2 GPU ids, e.g. --gpu_ids 0,1"
            device = torch.device(f"cuda:{gpu_ids[0]}")
        else:
            device = torch.device(f"cuda:{args.cuda_id}")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    
    # ---------------------------------------------------------------------------
    # Gaze-loss switch (THIS MUST NOT BE COMMENTED OUT)
    # ---------------------------------------------------------------------------
    use_gaze_loss = (args.model == "rsscnn" and args.gaze != "off" and float(getattr(args, "attn_w", 0.0) or 0.0) > 0.0)
    
    # TRANSFORMER PATH
    if args.backbone in TRANSFORMER_BACKBONES:
        print(f"Using TRANSFORMER architecture: {args.backbone}")
        from nets.transformer import Transformer as Net
    
        # Instantiate the Siamese Transformer Wrapper
        net = Net(
            backbone=backbone_model,
            model=args.model,
            # Pooling & Architecture
            pooling=getattr(args, "pooling", "cls"),
            pool_k=getattr(args, "pool_k", 10),
            num_classes=3 if args.ties else 2,
            # Training Dynamics
            finetune=args.finetune,
            num_ft_blocks=args.num_ft_blocks,
            rank_dropout=args.rank_dropout,
            cross_dropout=args.cross_dropout,
            # Gaze & Attention
            use_attn_hook=use_gaze_loss,
            return_attn=use_gaze_loss,
            attention_mode=args.attention_mode,
            attn_topk=args.attn_topk,
            attn_out_hw=tuple(args.gaze_grid_size),
        )
    
        # If your Transformer wrapper uses this flag, keep it consistent
        net.attn_grad = use_gaze_loss
    
        # -----------------------------------------------------------------------
        # ensure attention map resolution == gaze map resolution
        # -----------------------------------------------------------------------
        if use_gaze_loss:
            if not hasattr(args, "gaze_grid_size") or args.gaze_grid_size is None:
                raise ValueError(
                    "args.gaze_grid_size is missing. "
                    "You must compute gaze_grid_size (e.g., (14,14) or (16,16)) before model creation."
                )
    
            # Update attn_cfg.out_hw if your Transformer has an AttnConfig dataclass
            try:
                from dataclasses import replace
                if hasattr(net, "attn_cfg"):
                    net.attn_cfg = replace(net.attn_cfg, out_hw=tuple(args.gaze_grid_size))
            except Exception as e:
                raise RuntimeError(f"Failed to set net.attn_cfg.out_hw to gaze_grid_size={args.gaze_grid_size}: {e}")
    
    elif args.backbone in CNN_BACKBONES:
        print(f"Using CNN architecture: {args.backbone}")
        from nets.cnn import CNN as Net
        from torchvision import models
    
        cnn_factory = {
            "alex": models.alexnet,
            "vgg": models.vgg19,
            "dense": models.densenet121,
            "resnet": models.resnet50,
        }
    
        flatten_spatial = (getattr(args, "cnn_pool", "gap") == "flatten")
    
        net = Net(
            backbone=cnn_factory[args.backbone],
            model=args.model,
            finetune=args.finetune,
            num_classes=3 if args.ties else 2,
            flatten_spatial=flatten_spatial,
            flat_dim_override=None,  # scratch training: no ckpt inference
            gaze_grid_size=getattr(args, "gaze_grid_size", (14, 14))[0] if hasattr(args, "gaze_grid_size") else 14,
        )
    
        #print(f"[CNN] cnn_pool={args.cnn_pool} -> flatten_spatial={flatten_spatial}, flat_dim={net.flat_dim}")

    
    else:
        known_models = TRANSFORMER_BACKBONES + CNN_BACKBONES
        raise ValueError(f"Invalid backbone '{args.backbone}'. Available: {known_models}")
    
    # Move to device
    net.to(device)
    
    # DataParallel (after moving to device)
    if args.cuda and args.multi_gpu:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
        print(f"[DataParallel] Using GPUs: {gpu_ids} (primary cuda:{gpu_ids[0]})")
    
    # Resume 
    if args.resume:
        print("\nResuming training.")
        checkpoint_name = os.path.join(args.model_dir, f"{args.resume_checkpoint}")
        print("Loading model:", checkpoint_name)
    
        state = torch.load(checkpoint_name, map_location=device)
        is_dp = isinstance(net, torch.nn.DataParallel)
    
        if not is_dp and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
    
        if is_dp and not any(k.startswith("module.") for k in state.keys()):
            state = {f"module.{k}": v for k, v in state.items()}
    
        net.load_state_dict(state, strict=True)
        print()

    # =============================================================================================== #
    # 7) RUN PLAN (centralized)
    # =============================================================================================== #
    """    
    print_run_plan(
        args,
        train_df=X_train,
        val_df=X_val,
        test_df=X_test,
        train_loader=train_loader,
        val_loader=val_loader,
        train_tfms=train_tfms,
        eval_tfms=eval_tfms,
        model=net,
        optimizer=None,
        scheduler=None,
    )
    """

    # =============================================================================================== #
    # 8) TRAIN
    # =============================================================================================== #
    run_name = ""
    if args.log_wandb and wandb.run is not None:
        run_name = wandb.run.name
    print("\n[Wandb]")
    print(f"  Run Name     : {run_name}")
    
    if trial is not None:
        trial.set_user_attr("wandb_run_name", run_name)

    best_val_acc = train(
        device,
        net,
        train_loader,
        val_loader,
        test_loader,
        args,
        logger,
        trial=trial,
        train_df=X_train,
        val_df=X_val,
        test_df=X_test,
        train_tfms=train_tfms,   
        eval_tfms=eval_tfms,
    )
    
    net_for_hooks = net.module if isinstance(net, torch.nn.DataParallel) else net
    if hasattr(net_for_hooks, "remove_attention_hooks"):
        net_for_hooks.remove_attention_hooks()


    # -------- GPU / memory cleanup BETWEEN trials --------
    try:
        del net
        del train_loader, val_loader, test_loader
    except NameError:
        pass

    gc.collect()
    if args.cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_acc


if __name__ == "__main__":
    args = arg_parse().parse_args()
    run_training_with_args(args)
