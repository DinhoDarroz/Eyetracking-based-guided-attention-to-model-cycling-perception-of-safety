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
from sklearn.model_selection import train_test_split
import numpy as np
import wandb
from data import ComparisonsDataset, CustomTransform, PairwiseAugmentationPipeline
from torchvision import transforms

from train_utils import (
    validate_and_normalize_args,
    build_transformer_backbone,
    compute_class_weights_from_df,
    print_augmentation_plan,
    print_run_plan,
    resolve_batch_size,
    PAIRWISE_AUG_PRESETS
)

import logging
from datetime import date
from scripts.train_script import train
import warnings
import gc
import timm

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
    parser.add_argument("--gaze", default="use", choices=["off", "use", "only"])
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
    parser.add_argument("--backbone", type=str,
                        default="vit_base_dino",
                        choices=[
                            "alex", "vgg", "dense", "resnet",
                            "deit_base", "deit_small", "deit_base_distilled", "deit_tiny",
                            "vit_base_dino", "vit_dinov2_base", "eva02_base", "vit_small", "vit_base_dinov3",
                        ])

    # -------------------- LOSSES ------------------------------
    parser.add_argument("--rank_w", type=float, default=1.0)
    parser.add_argument("--ties_w", type=float, default=1.0)
    parser.add_argument("--ranking_margin", type=float, default=0.7)
    parser.add_argument("--ranking_margin_ties", type=float, default=None)
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--attn_w", type=float, default=1.0)
    parser.add_argument("--gaze_root", type=str, default="Eyetracker_attention_maps")

    # -------------------- MISC -------------------------------
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--num_ft_blocks", type=int, default=1, help="Number of last transformer blocks to unfreeze when --finetune is set.")
    return parser


def read_data(args):
    # -------- Load --------
    try:
        comparisons_df = pickle.load(open(args.comparisons, 'rb'))
    except Exception:
        comparisons_df = pd.read_pickle(args.comparisons)

    # -------- Select columns that actually exist --------
    cols_we_need = [
        'score',
        'image_l', 'image_r',
        'dataset',
        'has_eyetracker',
        'npy_file_l', 'npy_file_r',
        'survey_id', 'trial_id',
    ]
    existing_cols = [c for c in cols_we_need if c in comparisons_df.columns]
    comparisons_df = comparisons_df[existing_cols].copy()

    # -------- OPTIONAL: print which datasets exist --------
    if 'dataset' in comparisons_df.columns:
        print("Available city datasets in this file:")
        print(comparisons_df['dataset'].value_counts())
        print()

    # -------- filter by city / dataset source --------
    if 'dataset' in comparisons_df.columns:
        cities_arg = getattr(args, 'cities', 'all')
        if cities_arg.lower() != 'all':
            selected_cities = [c.strip() for c in cities_arg.split(',') if c.strip()]
            print(f"Filtering to cities: {selected_cities}")

            before = len(comparisons_df)
            comparisons_df = comparisons_df[comparisons_df['dataset'].isin(selected_cities)].copy()
            after = len(comparisons_df)

            if after == 0:
                print("[WARN] City filter resulted in 0 rows. "
                      "Check your --cities argument and the 'dataset' values in the pickle.")
            else:
                print(f"City filter kept {after}/{before} rows.")

    # -------- Ensure filenames have .jpg suffix --------
    if 'image_l' in comparisons_df.columns:
        comparisons_df['image_l'] = comparisons_df['image_l'].astype(str).apply(
            lambda x: x if x.lower().endswith('.jpg') else f'{x}.jpg'
        )
    if 'image_r' in comparisons_df.columns:
        comparisons_df['image_r'] = comparisons_df['image_r'].astype(str).apply(
            lambda x: x if x.lower().endswith('.jpg') else f'{x}.jpg'
        )

    if 'has_eyetracker' in comparisons_df.columns:
        comparisons_df['has_eyetracker'] = (
            comparisons_df['has_eyetracker']
            .replace({'True': True, 'False': False, 'true': True, 'false': False})
            .fillna(False)
            .astype(bool)
        )

        if args.gaze == 'only':
            comparisons_df = comparisons_df[comparisons_df['has_eyetracker']].copy()
            if comparisons_df.empty:
                print("[WARN] --gaze only: 0 rows with has_eyetracker==True after filtering.")
        
        #elif args.gaze == 'off':
        #    comparisons_df = comparisons_df[~comparisons_df['has_eyetracker']].copy()
        #    if comparisons_df.empty:
        #        print("[WARN] --gaze off: 0 rows with has_eyetracker==False after filtering.")

    # -------- Labels / ties --------
    if not args.ties:
        comparisons_df = comparisons_df[comparisons_df['score'] != 0].copy()
        comparisons_df['score_classification'] = comparisons_df['score'].replace({-1: 0, +1: 1})
    else:
        comparisons_df['score_classification'] = comparisons_df['score'] + 1

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
            "backbone_lr_scale": getattr(args, "backbone_lr_scale", None),
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
    
    # Score distribution AFTER filtering (this is the distribution you will actually train on)
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
    X_train, X_test = train_test_split(comparisons_df, test_size=0.2, random_state=args.seed)
    X_train, X_val = train_test_split(X_train, test_size=0.13, random_state=args.seed)
    
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
    # 4) TRANSFORMS
    # =============================================================================================== #
    # Policy:
    #   - eval_tfms: deterministic preprocessing used for validation/test (and also used as fallback for train)
    #   - train_tfms:
    #       * if args.augment is ON AND gaze-alignment loss is NOT active -> use pairwise augmentations
    #       * otherwise -> fall back to eval_tfms for deterministic behavior
    #
    # Why disable augmentation when gaze alignment is active?
    #   - Your gaze alignment loss assumes spatial correspondence between (image ↔ gaze map).
    #   - Random crops / flips / geometry / color jitter would invalidate this correspondence unless you also
    #     transform the gaze maps identically.
    #   - So you currently choose the safe policy: do not augment if gaze alignment is being used.
    # =============================================================================================== #
    
    # "use_gaze_alignment" means: run has gaze supervision and the loss weight is > 0
    use_gaze_alignment = (args.gaze != "off" and float(getattr(args, "attn_w", 0.0)) > 0.0)
    
    # ----------------------------------------------------------------------------------------------- #
    # 4.1) Deterministic evaluation preprocessing (used for val/test, and as train fallback)
    # ----------------------------------------------------------------------------------------------- #
    # Equivalent to: Resize(short side=256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet stats)
    # Note: you wrap each torchvision transform with CustomTransform, presumably because your Dataset
    # stores images in dicts and CustomTransform applies transforms to the correct dict keys.
    eval_tfms = transforms.Compose(
        [
            CustomTransform(transforms.Resize(256)),
            CustomTransform(transforms.CenterCrop(224)),
            CustomTransform(transforms.ToTensor()),
            CustomTransform(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ),
        ]
    )
    
    # ----------------------------------------------------------------------------------------------- #
    # 4.2) Training transforms selection logic
    # ----------------------------------------------------------------------------------------------- #
    # Default: no augmentation -> match eval preprocessing (deterministic)
    train_tfms = eval_tfms
    
    # Enable PairwiseAugmentationPipeline only when:
    #   (a) user asked for augmentation, and
    #   (b) gaze-alignment loss is not active (otherwise misalignment risk)
    augment_level = getattr(args, "augment", "none")
    enable_pairwise_aug = (augment_level in ("light", "heavy")) and (not use_gaze_alignment)

    
    if enable_pairwise_aug:
        # ------------------------------------------------------------------------------------------- #
        # 4.3) Pairwise augmentation config (applies to TRAIN ONLY)
        # ------------------------------------------------------------------------------------------- #
        # The pipeline applies *paired* operations to both images of a comparison.
        # Key invariance operations:
        #   - Horizontal flip (paired): keeps label same
        #   - Swap left/right (paired): label is inverted for ranking, remapped for classification
        #
        # Mild photometric operations:
        #   - Color jitter (paired): small changes to brightness/contrast/saturation/hue
        #   - Grayscale (paired)
        #
        # Rare geometry operation:
        #   - Bottom-band crop (paired): small chance, acts like "sky removal"
        #
        # Tensor-level regularization:
        #   - Random erasing (paired): small chance
        #
        # IMPORTANT: You currently disable augmentation whenever gaze alignment is used. Therefore
        # the gaze-related parameters below are effectively inert in this branch, but kept for clarity.
        train_tfms = PairwiseAugmentationPipeline(
            augment=True,
            ties=args.ties,
    
            disable_aug_when_gaze=True,
            allow_swap_when_gaze=False,
    
            resize_short=256,
            out_size=224,
    
            **PAIRWISE_AUG_PRESETS[augment_level],
        )
    else:
        train_tfms = eval_tfms

    # =============================================================================================== #
    # 5) DATA LOADERS
    # =============================================================================================== #
    use_gaze = (args.gaze != 'off')

    train_set = ComparisonsDataset(
        dataframe=X_train,
        root_dir=args.dataset,
        transform=train_tfms,
        logger=logger,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
    )
    val_set = ComparisonsDataset(
        dataframe=X_val,
        root_dir=args.dataset,
        transform=eval_tfms,
        logger=logger,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
    )
    test_set = ComparisonsDataset(
        dataframe=X_test,
        root_dir=args.dataset,
        transform=eval_tfms,
        logger=logger,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
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

    use_gaze_loss = (args.model == "rsscnn" and args.gaze != "off" and args.attn_w > 0)

    TRANSFORMER_BACKBONES = [
        "deit_base",
        "deit_small",
        "deit_tiny",
        "deit_base_distilled",
        "vit_base_dino",
        "vit_dinov2_base",
        "eva02_base",
        "vit_small",
        "vit_base_dinov3",
    ]

    CNN_BACKBONES = ["alex", "vgg", "dense", "resnet"]

    if args.backbone in TRANSFORMER_BACKBONES:
        print("Using TRANSFORMER model (nets.transformer).")
        from nets.transformer import Transformer as Net
    elif args.backbone in CNN_BACKBONES:
        print("Using CNN model (nets.cnn).")
        from nets.cnn import CNN as Net
    else:
        raise Exception("Invalid model. To check available models run with -h.")

    cnn_backbones = {
        "alex": models.alexnet,
        "vgg": models.vgg19,
        "dense": models.densenet121,
        "resnet": models.resnet50,
    }

    if args.backbone in TRANSFORMER_BACKBONES:
        backbone_model = build_transformer_backbone(args.backbone)

        net = Net(
            backbone=backbone_model,
            model=args.model,
            num_classes=3 if args.ties else 2,
            finetune=args.finetune,
            num_ft_blocks=args.num_ft_blocks,
            rank_dropout=args.rank_dropout,
            cross_dropout=args.cross_dropout,
            use_attn_hook=use_gaze_loss,
            return_attn=use_gaze_loss,
            attention_mode=args.attention_mode,
            topk=args.attn_topk,
        )


        net.attn_grad = use_gaze_loss

    elif args.backbone in CNN_BACKBONES:
        net = Net(
            backbone=cnn_backbones[args.backbone],
            model=args.model,
            finetune=args.finetune,
            num_classes=3 if args.ties else 2,
        )
    else:
        raise Exception("Invalid model. To check available models run with -h.")

    net.to(device)

    if args.cuda and args.multi_gpu:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
        print(f"[DataParallel] Using GPUs: {gpu_ids} (primary cuda:{gpu_ids[0]})")

    if args.resume:
        print("\nResuming training.")
        checkpoint_name = os.path.join(args.model_dir, f"{args.resume_checkpoint}")
        print("Loading model:", checkpoint_name)
    
        state = torch.load(checkpoint_name, map_location=device)
    
        # If current model is DataParallel, it expects "module." keys.
        is_dp = isinstance(net, torch.nn.DataParallel)
    
        # If checkpoint keys have "module." but model is not DP, strip them.
        if not is_dp and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
    
        # If model is DP but checkpoint keys do not have "module.", add them.
        if is_dp and not any(k.startswith("module.") for k in state.keys()):
            state = {f"module.{k}": v for k, v in state.items()}
    
        net.load_state_dict(state, strict=True)
        print()


    # =============================================================================================== #
    # 7) RUN PLAN (centralized)
    # =============================================================================================== #
    
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

    # =============================================================================================== #
    # 8) TRAIN
    # =============================================================================================== #
    run_name = ''
    if args.log_wandb and wandb.run is not None:
        run_name = wandb.run.name
    print("Training:", run_name)

    if trial is not None:
        trial.set_user_attr("wandb_run_name", run_name)

    best_val_acc = train(device, net, train_loader, val_loader, test_loader, args, logger, trial=trial)

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


if __name__ == '__main__':
    args = arg_parse().parse_args()
    run_training_with_args(args)
