import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import logging
from datetime import date
import random
import math
import pickle


try:
    import wandb
except Exception:
    wandb = None
    
from backbone_registry import infer_vit_grid_size, resolve_backbone
from gaze_policy import build_gaze_config
from train_utils import compute_class_weights_from_df, print_run_plan

from data import (
    ComparisonsDataset,
    build_eval_transforms,
    build_train_eval_preprocessing,
)
# -------------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------------

def _seed_everything(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

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
    args_config = dict(vars(args))
    args_config.update(
        {
            # Keep the old names for compatibility with existing dashboards, but
            # also log the raw CLI names so recovery sweeps are easy to group.
            "architecture_backbone": args.backbone,
            "architecture_model": args.model,
            "finetune_backbone": args.finetune,
            "epochs": args.max_epochs,
            "dataset": args.comparisons,
            "rank_margin": args.ranking_margin,
            "rank_margin_ties": args.ranking_margin_ties,
            "resume_epoch": args.epoch,
            "checkpoint": checkpoint_path,
            "gaze_mode": str(getattr(args, "gaze_mode", "disable")),
            "use_nobp": bool(getattr(args, "use_nobp", False)),
        }
    )

    wandb.init(
        project=args.wandb_project,
        config=args_config,
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

        if str(getattr(args, "eyetracker_filter", "all")).lower().strip() == "only":
            before = len(comparisons_df)
            comparisons_df = comparisons_df[comparisons_df["has_eyetracker"]].copy()
            after = len(comparisons_df)

            print(f"[read_data] Rows after eyetracker_filter=only: {after}/{before}")
            if after == 0:
                print("[WARN] --eyetracker_filter only removed all rows.")

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


def apply_backbone_hparam_overrides(args) -> None:
    """
    Override selected hyperparameters based on backbone.

    Current policy:
      - dinov3_vitb16:
          num_ft_blocks=4, ranking_margin=2.0,
          attn_w(raw)=1.5, attn_w(rollout)=0.01
      - deit3_base_patch16_224:
          num_ft_blocks=4, ranking_margin=2.0,
          attn_w(raw)=0.75, attn_w(rollout)=0.01
      - vit_base_patch16_224:
          num_ft_blocks=8, ranking_margin=3.2,
          attn_w(raw)=2.0, attn_w(rollout)=0.25

    Only modifies attributes when the backbone is explicitly listed here.
    Also keeps ranking_margin_ties aligned when it was not explicitly set.
    Attention mode aliases:
      - "last" and "cls" are treated as "raw" for attn_w override.
    """
    backbone = str(getattr(args, "backbone", "")).strip().lower()
    attn_mode = str(getattr(args, "attention_mode", "raw")).strip().lower()
    if attn_mode in ("last", "cls"):
        attn_mode = "raw"
    if attn_mode not in ("raw", "rollout"):
        attn_mode = "raw"

    overrides = {
        "dinov3_vitb16": {
            "num_ft_blocks": 4,
            "ranking_margin": 2.0,
            "attn_w": {"raw": 1.5, "rollout": 0.01},
        },
        "deit3_base_patch16_224": {
            "num_ft_blocks": 4,
            "ranking_margin": 2.0,
            "attn_w": {"raw": 0.75, "rollout": 0.01},
        },
        "vit_base_patch16_224": {
            "num_ft_blocks": 8,
            "ranking_margin": 3.2,
            "attn_w": {"raw": 2.0, "rollout": 0.25},
        },
    }

    cfg = overrides.get(backbone)
    if cfg is None:
        return

    old_num_ft_blocks = getattr(args, "num_ft_blocks", None)
    old_ranking_margin = getattr(args, "ranking_margin", None)
    old_ranking_margin_ties = getattr(args, "ranking_margin_ties", None)
    old_attn_w = getattr(args, "attn_w", None)

    args.num_ft_blocks = int(cfg["num_ft_blocks"])
    args.ranking_margin = float(cfg["ranking_margin"])
    args.attn_w = float(cfg["attn_w"][attn_mode])

    # Keep ties margin synchronized when it was not explicitly set.
    if old_ranking_margin_ties is None:
        args.ranking_margin_ties = float(args.ranking_margin)

    args._backbone_override_info = {
        "backbone": backbone,
        "applied": True,
        "old_num_ft_blocks": old_num_ft_blocks,
        "new_num_ft_blocks": args.num_ft_blocks,
        "old_ranking_margin": old_ranking_margin,
        "new_ranking_margin": args.ranking_margin,
        "old_ranking_margin_ties": old_ranking_margin_ties,
        "new_ranking_margin_ties": getattr(args, "ranking_margin_ties", None),
        "attention_mode": attn_mode,
        "old_attn_w": old_attn_w,
        "new_attn_w": getattr(args, "attn_w", None),
    }
    
def _boolish_series_to_bool_mask(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip().isin(["1", "true", "t", "yes", "y"])


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _split_paths(comparisons_path: str, splits_dir: str) -> tuple[str, str, str]:
    split_prefix = os.path.splitext(os.path.basename(comparisons_path))[0]
    train_path = os.path.join(splits_dir, f"{split_prefix}_train.pkl")
    val_path = os.path.join(splits_dir, f"{split_prefix}_val.pkl")
    test_path = os.path.join(splits_dir, f"{split_prefix}_test.pkl")
    return train_path, val_path, test_path


def _load_or_split(
    df: pd.DataFrame,
    seed: int,
    comparisons_path: str,
    splits_dir: str = "splits",
    train_pct: float = 0.67,
    val_pct: float = 0.13,
    test_pct: float = 0.20,
    load_if_exists: bool = True,
    save_splits: bool = True,
    train_gaze_frac: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standard random split + optional post-split gaze rebalancing.

    train_gaze_frac:
      - None: no gaze rebalancing (default behavior)
      - float in [0, 1]: target fraction of all has_eyetracker=True rows that should be in TRAIN.
        Typical range: [train_pct, 1.0]. 1.0 forces all gaze rows into train (when possible).
    """

    _ensure_dir(splits_dir)

    total = float(train_pct + val_pct + test_pct)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"train_pct + val_pct + test_pct must sum to 1.0, got {total}")

    if test_pct <= 0.0 or test_pct >= 1.0:
        raise ValueError("test_pct must be in (0, 1)")

    if val_pct < 0.0 or train_pct < 0.0:
        raise ValueError("train_pct and val_pct must be >= 0")

    # ----------------------------
    # Resolve split cache filenames
    # ----------------------------
    split_prefix = os.path.splitext(os.path.basename(comparisons_path))[0]

    gaze_tag = ""
    if train_gaze_frac is not None:
        g = float(train_gaze_frac)
        g = max(0.0, min(1.0, g))
        gaze_tag = f"_trainGaze{int(round(100.0 * g)):03d}"

    train_path = os.path.join(splits_dir, f"{split_prefix}{gaze_tag}_train.pkl")
    val_path = os.path.join(splits_dir, f"{split_prefix}{gaze_tag}_val.pkl")
    test_path = os.path.join(splits_dir, f"{split_prefix}{gaze_tag}_test.pkl")

    if load_if_exists and all(os.path.exists(p) for p in (train_path, val_path, test_path)):
        return (
            pd.read_pickle(train_path),
            pd.read_pickle(val_path),
            pd.read_pickle(test_path),
        )

    # ----------------------------
    # 1) Baseline random split
    # ----------------------------
    X_train, X_test = train_test_split(
        df,
        test_size=test_pct,
        shuffle=True,
        random_state=int(seed),
    )

    remaining = 1.0 - test_pct
    if remaining <= 0.0:
        raise ValueError("test_pct leaves no data for train/val")

    val_size_of_train = val_pct / remaining

    X_train, X_val = train_test_split(
        X_train,
        test_size=val_size_of_train,
        shuffle=True,
        random_state=int(seed),
    )

    # ----------------------------
    # 2) Optional gaze rebalancing
    # ----------------------------
    if train_gaze_frac is not None:
        if "has_eyetracker" in X_train.columns and "has_eyetracker" in X_val.columns and "has_eyetracker" in X_test.columns:
            frac = float(train_gaze_frac)
            frac = max(0.0, min(1.0, frac))

            X_train = X_train.copy()
            X_val = X_val.copy()
            X_test = X_test.copy()

            tr_g = _boolish_series_to_bool_mask(X_train["has_eyetracker"])
            va_g = _boolish_series_to_bool_mask(X_val["has_eyetracker"])
            te_g = _boolish_series_to_bool_mask(X_test["has_eyetracker"])

            total_gaze = int(tr_g.sum() + va_g.sum() + te_g.sum())
            if total_gaze > 0:
                desired_train_gaze = int(math.ceil(frac * float(total_gaze)))
                current_train_gaze = int(tr_g.sum())
                need = desired_train_gaze - current_train_gaze

                if need > 0:
                    # Candidate gaze rows to pull into train
                    val_gaze_idx = X_val.index[va_g].to_list()
                    test_gaze_idx = X_test.index[te_g].to_list()

                    if len(val_gaze_idx) > 0:
                        val_gaze_idx = X_val.loc[val_gaze_idx].sample(frac=1.0, random_state=int(seed) + 101).index.to_list()
                    if len(test_gaze_idx) > 0:
                        test_gaze_idx = X_test.loc[test_gaze_idx].sample(frac=1.0, random_state=int(seed) + 202).index.to_list()

                    def _swap_from_source(
                        X_src: pd.DataFrame,
                        src_gaze_idx: list,
                        n_take: int,
                        X_train_local: pd.DataFrame,
                        rng_seed: int,
                    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
                        if n_take <= 0 or len(src_gaze_idx) == 0:
                            return X_train_local, X_src, 0

                        take_idx = src_gaze_idx[:n_take]
                        take_rows = X_src.loc[take_idx]
                        X_src = X_src.drop(index=take_idx)

                        tr_mask = _boolish_series_to_bool_mask(X_train_local["has_eyetracker"])
                        train_no_gaze_idx = X_train_local.index[~tr_mask].to_list()

                        if len(train_no_gaze_idx) >= n_take:
                            swap_idx = (
                                X_train_local.loc[train_no_gaze_idx]
                                .sample(n=n_take, random_state=int(rng_seed))
                                .index.to_list()
                            )
                            swap_rows = X_train_local.loc[swap_idx]
                            X_train_local = X_train_local.drop(index=swap_idx)
                            X_src = pd.concat([X_src, swap_rows], axis=0)

                        X_train_local = pd.concat([X_train_local, take_rows], axis=0)
                        return X_train_local, X_src, len(take_idx)

                    # Pull from val first, then test
                    take_val = min(need, len(val_gaze_idx))
                    X_train, X_val, got_val = _swap_from_source(X_val, val_gaze_idx, take_val, X_train, int(seed) + 303)
                    need -= got_val

                    if need > 0:
                        take_test = min(need, len(test_gaze_idx))
                        X_train, X_test, got_test = _swap_from_source(X_test, test_gaze_idx, take_test, X_train, int(seed) + 404)
                        need -= got_test

    # ----------------------------
    # 3) Save splits
    # ----------------------------
    if save_splits:
        X_train.to_pickle(train_path)
        X_val.to_pickle(val_path)
        X_test.to_pickle(test_path)

    return X_train, X_val, X_test


def _print_filtered_dataset_summary(args, df: pd.DataFrame) -> None:
    print("\n=== Effective Dataset (after all filters, before split) ===")
    print(f"Comparisons file : {args.comparisons}")
    print(f"Cities requested : {args.cities}")
    print(f"Gaze mode        : {getattr(args, 'gaze_mode', 'disable')}  (rows kept depend on has_eyetracker + eyetracker_filter)")
    print(f"Ties enabled     : {args.ties}  (ties=False removes score==0 rows)")
    print(f"Final row count  : {len(df):,}")

    if "score" in df.columns:
        print("\nScore distribution (post-filtering):")
        score_counts = df["score"].value_counts().sort_index()
        total_rows = len(df)
        for s, c in score_counts.items():
            print(f"  score={int(s):>2}: {int(c):>6,} ({(100.0 * float(c) / max(1, total_rows)):5.2f}%)")

    if "has_eyetracker" in df.columns:
        print("\nEyetracker availability (post-filtering):")
        et_counts = df["has_eyetracker"].value_counts(dropna=False)
        for k, v in et_counts.items():
            print(f"  {str(k):>5}: {int(v):>6,} ({(100.0 * float(v) / max(1, len(df))):5.2f}%)")

    print("\nExample rows (post-filtering):")
    print(df.head(3))
    print("========================================================\n")

def _print_image_overlap_stats(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    left_col: str = "image_l",
    right_col: str = "image_r",
    dataset_col: str | None = "dataset",
) -> None:
    def _unique_images(df: pd.DataFrame) -> set[str]:
        if len(df) == 0:
            return set()

        cols = [c for c in [left_col, right_col, dataset_col] if c is not None and c in df.columns]
        df2 = df[cols].copy()
        df2[left_col] = df2[left_col].astype(str)
        df2[right_col] = df2[right_col].astype(str)

        if dataset_col is not None and dataset_col in df2.columns:
            ds = df2[dataset_col].astype(str)
            left = ds + "/" + df2[left_col]
            right = ds + "/" + df2[right_col]
        else:
            left = df2[left_col]
            right = df2[right_col]

        return set(pd.concat([left, right], ignore_index=True).tolist())

    def _pct(n: int, d: int) -> float:
        return 0.0 if d == 0 else (100.0 * n / d)

    tr = _unique_images(X_train)
    va = _unique_images(X_val)
    te = _unique_images(X_test)

    tr_va = tr & va
    tr_te = tr & te
    va_te = va & te

    print(f"Images : train={len(tr)} | val={len(va)} | test={len(te)}")

    print(f"train∩val : {len(tr_va)} | % of train={_pct(len(tr_va), len(tr)):.2f}% | % of val={_pct(len(tr_va), len(va)):.2f}%")
    print(f"train∩test: {len(tr_te)} | % of train={_pct(len(tr_te), len(tr)):.2f}% | % of test={_pct(len(tr_te), len(te)):.2f}%")
    print(f"val∩test  : {len(va_te)} | % of val={_pct(len(va_te), len(va)):.2f}% | % of test={_pct(len(va_te), len(te)):.2f}%")



def _print_has_eyetracker_by_split(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> None:
    def _counts(df_: pd.DataFrame) -> tuple[int, int, float]:
        s = _boolish_series_to_bool_mask(df_["has_eyetracker"])
        n_true = int(s.sum())
        n_total = int(len(df_))
        return n_true, n_total, (n_true / max(1, n_total))

    if "has_eyetracker" not in X_train.columns:
        print("\nColumn 'has_eyetracker' not found in comparisons_df.")
        return

    tr_n, tr_tot, tr_rate = _counts(X_train)
    va_n, va_tot, va_rate = _counts(X_val)
    te_n, te_tot, te_rate = _counts(X_test)

    print("\nhas_eyetracker per split:")
    print(f"  Train: {tr_n}/{tr_tot} = {tr_rate:.2%}")
    print(f"  Val  : {va_n}/{va_tot} = {va_rate:.2%}")
    print(f"  Test : {te_n}/{te_tot} = {te_rate:.2%}")


def _print_split_sizes(df: pd.DataFrame, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> None:
    total = max(1, len(df))
    print("=== Splits (on the filtered dataset above) ===")
    print(f"- Train: {len(X_train):,}  [{len(X_train)/total:.2%}]")
    print(f"- Val  : {len(X_val):,}  [{len(X_val)/total:.2%}]")
    print(f"- Test : {len(X_test):,}  [{len(X_test)/total:.2%}]")
    print("========================================================\n")


def _print_label_distribution_by_split(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> None:
    print("=== Label distribution per split (score, after filtering) ===")
    for part_name, df_part in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        counts = df_part["score"].value_counts().sort_index()
        total_part = max(1, len(df_part))
        print(f"- {part_name}: {len(df_part):,} samples")
        for cls_val, cls_count in counts.items():
            pct = 100.0 * float(cls_count) / float(total_part)
            print(f"    score={int(cls_val):>2d}: {int(cls_count):>6,} ({pct:5.2f}%)")
    print("============================================================")


def _compute_and_attach_class_weights(args, X_train: pd.DataFrame) -> None:
    args.class_weights = compute_class_weights_from_df(
        X_train["score_classification"],
        use_ties=args.ties,
        enable_weights=args.use_class_weights,
    )

    if args.use_class_weights and args.class_weights is not None:
        cw = args.class_weights.detach().cpu().numpy().tolist()
        print(f"Class weights: ON  (computed from Train split) → {cw}")
    else:
        print("Class weights: OFF")
    print()


def _parse_gpu_ids(args) -> list[int]:
    raw = str(getattr(args, "gpu_ids", "") or "")
    ids = [int(x) for x in raw.split(",") if x.strip() != ""]
    return ids


def _select_device(args) -> tuple[torch.device, list[int]]:
    if not args.cuda:
        return torch.device("cpu"), []

    if not torch.cuda.is_available():
        raise AssertionError("ERROR: --cuda was passed but CUDA is not available.")

    if getattr(args, "multi_gpu", False):
        gpu_ids = _parse_gpu_ids(args)
        if len(gpu_ids) < 2:
            raise AssertionError("--multi_gpu requires at least 2 GPU ids, e.g. --gpu_ids 0,1")
        return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids

    cuda_id = int(getattr(args, "cuda_id", 0))
    return torch.device(f"cuda:{cuda_id}"), []


def _load_state_dict_safely(net: torch.nn.Module, state: dict, strict: bool = True) -> None:
    is_dp = isinstance(net, torch.nn.DataParallel)

    has_module_prefix = any(k.startswith("module.") for k in state.keys())
    if (not is_dp) and has_module_prefix:
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if is_dp and (not has_module_prefix):
        state = {f"module.{k}": v for k, v in state.items()}

    net.load_state_dict(state, strict=bool(strict))


# -------------------------------------------------------------------------------------------------
# Backbone / transforms / model construction
# -------------------------------------------------------------------------------------------------

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

CNN_SPECS = {
    "input_size": (3, 224, 224),
    "crop_pct": 0.875,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
    "interpolation": "bilinear",
}


# -------------------------------------------------------------------------------------------------
# Gaze grid resolver
# -------------------------------------------------------------------------------------------------

def _resolve_gaze_grid(args, is_cnn_backbone: bool, backbone_model, model_specs: dict) -> tuple[int, int]:
    """
    Resolve the supervision grid used by align-mode KL and (optionally) the data pipeline output.

    If args.gaze_map_size != "auto", the grid is forced to (S, S).
    Otherwise:
      - CNN backbones use a fixed 14x14
      - ViTs infer the patch grid from the backbone + specs
    """
    if str(getattr(args, "gaze_map_size", "auto")).lower() != "auto":
        forced = int(getattr(args, "gaze_map_size"))
        return forced, forced

    if is_cnn_backbone:
        return 14, 14

    grid_h, grid_w = infer_vit_grid_size(backbone_model, model_specs)
    return int(grid_h), int(grid_w)

# -------------------------------------------------------------------------------------------------
# Backbone specs + transforms
# -------------------------------------------------------------------------------------------------

def _build_transforms_and_specs(args):
    """
    Resolve the backbone configuration and build preprocessing pipelines.

    Central gaze policy:
      - build_gaze_config(...) defines all gaze dependencies (load/inject/KL/hook needs)
      - transforms only receive enable_gaze and gaze_output
      - args.gaze_map_size_int only matters when gaze maps are actually loaded
    """

    backbone_name = str(args.backbone).lower().strip()
    is_cnn_backbone = backbone_name in set(CNN_BACKBONES)

    # Backbone resolution
    if is_cnn_backbone:
        backbone_model = None
        model_specs = {
            "alias": args.backbone,
            "timm_id": None,
            **CNN_SPECS,
            "img_size": int(CNN_SPECS["input_size"][-1]),
        }
    else:
        backbone_model, model_specs = resolve_backbone(args.backbone, pretrained=True, strict=True)

    # Expected eval crop resolution (square)
    out_size = int(model_specs.get("img_size", model_specs["input_size"][-1]))

    # Attention/KL grid size (used by align-style KL and align-style gaze maps)
    grid_h, grid_w = _resolve_gaze_grid(args, is_cnn_backbone, backbone_model, model_specs)
    args.gaze_grid_size = (int(grid_h), int(grid_w))
    model_specs["patch_grid_size"] = tuple(args.gaze_grid_size)
    if (not is_cnn_backbone) and backbone_model is not None:
        patch_embed = getattr(backbone_model, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", getattr(backbone_model, "patch_size", None))
        num_patches = getattr(patch_embed, "num_patches", None)
        if patch_size is not None:
            model_specs["patch_size"] = tuple(patch_size) if isinstance(patch_size, (tuple, list)) else (int(patch_size), int(patch_size))
        if num_patches is not None:
            model_specs["num_patches"] = int(num_patches)

    # Central gaze policy object (single source of truth)
    gaze_cfg = build_gaze_config(args, is_cnn_backbone=is_cnn_backbone, out_size=out_size)

    # gaze_output is only meaningful when gaze is loaded; kept stable for transform signatures
    gaze_output = str(getattr(gaze_cfg, "gaze_output", "align")).lower().strip()
    if gaze_output not in ("align", "guide"):
        gaze_output = "align"

    # Select gaze folder size only when gaze maps are loaded
    if bool(getattr(gaze_cfg, "load_gaze", False)) and gaze_output == "align" and int(grid_h) != int(grid_w):
        raise RuntimeError(
            "The current gaze-map folder convention is square, but the resolved ViT patch grid is "
            f"{(int(grid_h), int(grid_w))}. Use a square input/patch grid or extend the gaze loader "
            "to address rectangular map folders explicitly."
        )

    if bool(getattr(gaze_cfg, "load_gaze", False)) and (gaze_output == "guide"):
        args.gaze_map_size_int = int(out_size)
    else:
        args.gaze_map_size_int = int(grid_h)

    preprocessing = build_train_eval_preprocessing(
        specs=model_specs,
        augment=getattr(args, "augment", "none"),
        ties=bool(getattr(args, "ties", True)),
        gaze_grid_size=args.gaze_grid_size,
        enable_gaze=bool(getattr(gaze_cfg, "load_gaze", False)),
        gaze_output=gaze_output,
    )
    train_tfms = preprocessing["train"]
    eval_tfms = preprocessing["eval"]
    train_meta = preprocessing["train_meta"]
    eval_meta = preprocessing["eval_meta"]

    # Metadata for logging/printing
    args.expected_img_size = int(out_size)
    attn_w = float(getattr(args, "attn_w", 0.0) or 0.0)

    args.transforms_meta = {
        "backbone": args.backbone,
        "backbone_family": "cnn" if is_cnn_backbone else "transformer",
        "model_specs": dict(model_specs),
        "gaze": {
            "mode": str(getattr(gaze_cfg, "mode", "disable")),
            "load_gaze": bool(getattr(gaze_cfg, "load_gaze", False)),
            "inject": bool(getattr(gaze_cfg, "inject", False)),
            "compute_kl": bool(getattr(gaze_cfg, "compute_kl", False)),
            "use_kl_in_loss": bool(getattr(gaze_cfg, "use_kl_in_loss", False)),
            "need_attn_maps": bool(getattr(gaze_cfg, "need_attn_maps", False)),
            "gaze_output": gaze_output,
            "grid_size": tuple(args.gaze_grid_size),
            "out_size": int(out_size),
            "attn_w": float(attn_w),
        },
        "eval": dict(eval_meta),
        "train": dict(train_meta),
        "train_transform_class": train_tfms.__class__.__name__,
        "eval_transform_class": eval_tfms.__class__.__name__,
    }

    return backbone_model, train_tfms, eval_tfms, is_cnn_backbone


def _build_dataloaders(args, logger, X_train, X_val, X_test, train_tfms, eval_tfms):

    cfg = getattr(args, "gaze_cfg", None)
    use_gaze = bool(getattr(cfg, "load_gaze", False))

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
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    return train_loader, val_loader, test_loader


def _build_model(args, backbone_model, is_cnn_backbone: bool) -> torch.nn.Module:
    backbone_name = str(args.backbone).lower().strip()

    if (not is_cnn_backbone) and (backbone_name not in set([b.lower() for b in TRANSFORMER_BACKBONES])):
        known = TRANSFORMER_BACKBONES + CNN_BACKBONES
        raise ValueError(f"Invalid backbone '{args.backbone}'. Available: {known}")

    gaze_cfg = getattr(args, "gaze_cfg", None)

    need_attn_maps = bool(getattr(gaze_cfg, "need_attn_maps", False)) if gaze_cfg is not None else False
    use_gaze_inj = bool(getattr(gaze_cfg, "inject", False)) if gaze_cfg is not None else False
    use_kl_in_loss = bool(getattr(gaze_cfg, "use_kl_in_loss", False)) if gaze_cfg is not None else False

    gaze_grid = getattr(args, "gaze_grid_size", (14, 14))
    if isinstance(gaze_grid, (list, tuple)) and len(gaze_grid) == 2:
        gaze_grid_hw = (int(gaze_grid[0]), int(gaze_grid[1]))
    else:
        g = int(gaze_grid)
        gaze_grid_hw = (g, g)

    if is_cnn_backbone:
        from nets.cnn import CNN as Net
        from torchvision import models

        cnn_factory = {
            "alex": models.alexnet,
            "vgg": models.vgg19,
            "dense": models.densenet121,
            "resnet": models.resnet50,
        }

        flatten_spatial = (str(getattr(args, "cnn_pool", "gap")).lower().strip() == "flatten")

        net = Net(
            backbone=cnn_factory[backbone_name],
            model=args.model,
            finetune=args.finetune,
            num_classes=3 if args.ties else 2,
            flatten_spatial=flatten_spatial,
            flat_dim_override=None,
            gaze_grid_size=int(gaze_grid_hw[0]),
        )
        return net

    from nets.transformer import Transformer as Net
    from nets.egvit import EGViTConfig
    from nets.gaze_guidance import GuideGuidanceConfig

    guidance_cfg = GuideGuidanceConfig(
        enabled=bool(use_gaze_inj),
        bottleneck_dim=int(getattr(args, "guidance_bottleneck_dim", 20)),
        gaze_hidden_dim=int(getattr(args, "guidance_gaze_hidden_dim", 30)),
        conv_hidden_channels=int(getattr(args, "guidance_conv_hidden_channels", 64)),
        drop_prob=float(getattr(args, "guidance_drop_prob", 0.0)),
        strength=float(getattr(args, "guidance_strength", 1.0)),
        train_only=bool(getattr(args, "guide_train_only", False)),
    )

    use_egvit = False
    if gaze_cfg is not None:
        use_egvit = bool(getattr(gaze_cfg, "egvit", False)) or (str(getattr(gaze_cfg, "mode", "")).lower().strip() == "egvit")
    else:
        use_egvit = (str(getattr(args, "gaze_mode", "disable")).lower().strip() == "egvit")

    focus_hw = getattr(args, "egvit_focus_hw", (3, 3))
    if isinstance(focus_hw, (list, tuple)) and len(focus_hw) == 2:
        focus_hw = (int(focus_hw[0]), int(focus_hw[1]))
    else:
        focus_hw = (3, 3)

    egvit_cfg = EGViTConfig(
        enabled=bool(use_egvit),
        mask_type=str(getattr(args, "egvit_mask_type", "separated")),
        keep_ratio=float(getattr(args, "egvit_keep_ratio", 0.25)),
        focus_hw=tuple(focus_hw),
        drop_prob=float(getattr(args, "egvit_drop_prob", 0.0)),
        train_only=bool(getattr(args, "egvit_train_only", True)),
    )

    net = Net(
        backbone=backbone_model,
        model=args.model,
        pooling=getattr(args, "pooling", "patch_mean"),
        pool_k=getattr(args, "pool_k", 10),
        num_classes=3 if args.ties else 2,
        finetune=args.finetune,
        num_ft_blocks=args.num_ft_blocks,
        rank_dropout=args.rank_dropout,
        cross_dropout=args.cross_dropout,
        use_attn_hook=bool(need_attn_maps),
        return_attn=bool(need_attn_maps),
        attention_mode=args.attention_mode,
        attn_layer=int(getattr(args, "attn_layer", -1)),
        attn_out_hw=tuple(gaze_grid_hw),
        use_gaze_injection=bool(use_gaze_inj),
        guidance_cfg=guidance_cfg,
        use_egvit_masking=bool(use_egvit),
        egvit_cfg=egvit_cfg,
    )

    net.attn_grad = bool(use_kl_in_loss)

    if need_attn_maps:
        try:
            from dataclasses import replace
            if hasattr(net, "attn_cfg"):
                net.attn_cfg = replace(net.attn_cfg, out_hw=tuple(gaze_grid_hw))
        except Exception as e:
            raise RuntimeError(f"Failed to set attention output size to gaze_grid_size={gaze_grid_hw}: {e}")

    return net


def _maybe_wrap_dataparallel(args, net: torch.nn.Module, gpu_ids: list[int]) -> torch.nn.Module:
    if args.cuda and getattr(args, "multi_gpu", False):
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
        print(f"[DataParallel] Using GPUs: {gpu_ids} (primary cuda:{gpu_ids[0]})")
    return net


def _maybe_resume(args, net: torch.nn.Module, device: torch.device) -> None:
    if not getattr(args, "resume", False):
        return

    checkpoint_name = os.path.join(args.model_dir, f"{args.resume_checkpoint}")
    print("\nResuming training.")
    print("Loading model:", checkpoint_name)

    state = torch.load(checkpoint_name, map_location=device)
    _load_state_dict_safely(net, state, strict=True)
    print()


def _cleanup_between_trials(args, net, train_loader, val_loader, test_loader) -> None:
    net_for_hooks = net.module if isinstance(net, torch.nn.DataParallel) else net
    if hasattr(net_for_hooks, "remove_attention_hooks"):
        net_for_hooks.remove_attention_hooks()

    try:
        del net
        del train_loader, val_loader, test_loader
    except Exception:
        pass

    gc.collect()
    if args.cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()

def scale_lr_and_eta_min_by_unfrozen_blocks(
    args,
    *,
    lr_01: float | None = None,
    lr_other: float = 2e-5,
    eta_other: float = 5e-6,
    scale_eta_min: bool = True,
) -> tuple[float, float]:
    base_lr = float(getattr(args, "base_lr", 0.0))
    eta_min = float(getattr(args, "eta_min", 0.0))
    finetune = bool(getattr(args, "finetune", False))
    n = int(getattr(args, "num_ft_blocks", 0))

    if lr_01 is None:
        lr_01 = base_lr

    if (not finetune) or (n <= 1):
        new_lr = float(lr_01)
        if scale_eta_min and base_lr > 0.0:
            scale = new_lr / base_lr
            new_eta = eta_min * scale
        else:
            new_eta = eta_min
    else:
        new_lr = float(lr_other)
        new_eta = float(eta_other)

    return new_lr, new_eta
