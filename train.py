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
from sklearn.model_selection import train_Igtest_split
import numpy as np
import wandb
from data import ComparisonsDataset, CustomTransform, PairwiseAugmentationPipeline
from torchvision import transforms

from train_utils import (
    build_transformer_backbone,
    compute_class_weights_from_df,
    print_augmentation_plan,
    print_run_plan,
    resolve_batch_size
)

import logging
from datetime import date
from scripts.train_script import train
import warnings
import gc
import timm

warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None  # default='warn'


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Training subjective safety",
        allow_abbrev=False
    )

    # -------------------- BOOLEAN FLAGS --------------------
    parser.add_argument("--cuda", action="store_true", default=False, help="use CUDA")
    parser.add_argument("--resume", action="store_true", default=False, help="resume training")
    parser.add_argument("--finetune", "--ft", action="store_true", default=False)
    parser.add_argument("--ties", action="store_true", default=False, help="enable ties (3 classes)")
    parser.add_argument("--log_console", action="store_true", default=False)
    parser.add_argument("--log_wandb", action="store_true", default=False)
    parser.add_argument("--full_accuracy", action="store_true", default=False)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--use_class_weights", action="store_true", default=False)
    parser.add_argument("--use_seg", action="store_true", default=False)

    # -------------------- SCHEDULER -------------------------
    parser.add_argument(
        "--scheduler",
        type=str,
        default="warmup_cosine",
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
    es_group.add_argument("--early_stop", action="store_true", default=False,
                          help="Enable early stopping based on a validation metric.")
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
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--rank_dropout", type=float, default=0.3)
    parser.add_argument("--cross_dropout", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    # -------------------- PATHS & BASIC PARAMS ---------------
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--comparisons", type=str, default="comparisons_df.pickle")
    parser.add_argument("--dataset", type=str, default="images/")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resume_checkpoint", type=str)
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
    parser.add_argument("--ranking_margin", type=float, default=0.3)
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

        elif args.gaze == 'off':
            comparisons_df = comparisons_df[~comparisons_df['has_eyetracker']].copy()
            if comparisons_df.empty:
                print("[WARN] --gaze off: 0 rows with has_eyetracker==False after filtering.")

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

    # Ensure ties margin default is set
    if getattr(args, "ranking_margin_ties", None) is None:
        args.ranking_margin_ties = args.ranking_margin

    # ---- Scheduler sanity checks / friendly warnings ----
    if args.scheduler == "none":
        if args.warmup_frac != 0.0:
            print("[INFO] --scheduler none: ignoring --warmup_frac (no warmup used).")

    if args.scheduler not in ["warmup_cosine", "onecycle"]:
        if args.warmup_frac != 0.0:
            print("[INFO] --warmup_frac is only used by warmup_cosine/onecycle. "
                  "It will be ignored for scheduler =", args.scheduler)

    if args.scheduler not in ["warmup_cosine", "cosine", "warm_restarts"]:
        if args.eta_min != 1e-6:
            print("[INFO] --eta_min is only used by warmup_cosine/cosine/warm_restarts. "
                  "It will be ignored for scheduler =", args.scheduler)

    if args.scheduler != "warm_restarts":
        if args.T_0 != 10 or args.T_mult != 2:
            print("[INFO] T_0/T_mult are only used by warm_restarts. "
                  "They will be ignored for scheduler =", args.scheduler)
    
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
    
    use_gaze_alignment = (args.gaze != "off" and getattr(args, "attn_w", 0.0) > 0.0)
    
    eval_tfms = transforms.Compose([CustomTransform(transforms.Resize(256)),
                                     CustomTransform(transforms.CenterCrop(224)),
                                     CustomTransform(transforms.ToTensor()),
                                     CustomTransform(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                          std=[0.229, 0.224, 0.225])),
                                    
                                     ])
    
    if args.augment and (not use_gaze_alignment):
        train_tfms = PairwiseAugmentationPipeline(
            augment=True,
            ties=args.ties,
    
            # Gaze handling (won't matter here since gaze is off by condition)
            disable_aug_when_gaze=True,
            allow_swap_when_gaze=False,
    
            # Paired invariances
            hflip_p=0.25,
            swap_p=0.25,
    
            # Paired photometric
            color_jitter_p=0.05,
            jitter_brightness=0.30,
            jitter_contrast=0.30,
            jitter_saturation=0.30,
            jitter_hue=0.05,
            gray_p=0.05,
    
            # Paired geometry: bottom-band crop (sky removal)
            bottom_crop_p=0.25,
            bottom_keep_h=(0.65, 0.75),
            bottom_x_jitter_frac=0.04,
    
            # Tensor augmentation: random erasing
            erase_p=0.1,
            erase_scale=(0.05, 0.08),
            erase_ratio=(0.3, 3.3),
            erase_value=0.0,
    
            resize_short=256,
            out_size=224,
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
    if args.cuda:
        assert torch.cuda.is_available(), (
            "ERROR: --cuda was passed but CUDA is not available. "
            "Refusing to fall back to CPU."
        )
        device = torch.device(f"cuda:{args.cuda_id}")
    else:
        device = torch.device("cpu")
    
    print("Device:", device)
    print("Parsing model...")

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

    if args.resume:
        print("\nResuming training.")
        checkpoint_name = os.path.join(args.model_dir, f"{args.resume_checkpoint}")
        print("Loading model:", checkpoint_name)
        net.load_state_dict(torch.load(checkpoint_name, map_location=device))
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
