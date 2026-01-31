#!/usr/bin/env python3
"""
test.py

Evaluate a trained checkpoint on a comparisons pickle.

Configuration sources (priority order):
  1) --wandb_config PATH
  2) --wandb_run_id RUN_ID  (loads wandb/<run>/files/config.json if present, else wandb-metadata.json["args"])
  3) manual CLI flags

This script prints a structured run report so users can confirm:
  - where hyperparameters came from
  - what data filtering was applied
  - what model was instantiated
  - which checkpoint was loaded
  - what evaluation settings were used
"""

import argparse
import glob
import json
import os
import pickle
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as tv_models

from scripts.test_script import test
from data import ComparisonsDataset
from train_main_utils import build_eval_transforms

from train_utils import resolve_backbone, infer_vit_grid_size


warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


# -----------------------------
# Reporting helpers
# -----------------------------
def _hr(char: str = "=", n: int = 88) -> str:
    return char * n


def _fmt_bool(v: Any) -> str:
    if isinstance(v, bool):
        return "ON" if v else "OFF"
    return str(v)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _print_kv(title: str, kv: Dict[str, Any], indent: int = 2):
    pad = " " * indent
    print(f"{title}")
    for k in sorted(kv.keys()):
        print(f"{pad}{k}: {kv[k]}")


def _safe_len(x: Any) -> Optional[int]:
    try:
        return len(x)
    except Exception:
        return None

def _normalize_gaze_mode(args) -> str:
    gaze_mode = str(getattr(args, "gaze_mode", getattr(args, "gaze", "off"))).lower().strip()
    if gaze_mode not in ("off", "align", "guide", "align+guide"):
        gaze_mode = "off"
    args.gaze_mode = gaze_mode
    args.gaze = gaze_mode  # unified alias used across modules
    return gaze_mode


def _resolve_gaze_flags(args):
    gaze_mode = _normalize_gaze_mode(args)
    w_kl = float(getattr(args, "attn_w", 0.0) or 0.0)

    use_gaze_any = (gaze_mode != "off")
    use_gaze_kl = (gaze_mode in ("align", "align+guide")) and (w_kl > 0.0)
    use_gaze_inj = (gaze_mode in ("guide", "align+guide"))

    return gaze_mode, use_gaze_any, use_gaze_kl, use_gaze_inj

# -----------------------------
# W&B run discovery + parsing
# -----------------------------
def _find_wandb_run_files(run_id: str, wandb_dir: str = "wandb") -> dict:
    pattern = os.path.join(wandb_dir, f"run-*-{run_id}")
    matches = sorted(glob.glob(pattern))

    if not matches:
        pattern2 = os.path.join(wandb_dir, f"run-*-{run_id}*")
        matches = sorted(glob.glob(pattern2))

    if not matches:
        raise FileNotFoundError(
            f"Could not find a local W&B run directory for run id '{run_id}' under '{wandb_dir}/'."
        )

    run_dir = matches[-1]
    files_dir = os.path.join(run_dir, "files")

    config_json = os.path.join(files_dir, "config.json")
    if not os.path.exists(config_json):
        config_json = None

    metadata_json = os.path.join(files_dir, "wandb-metadata.json")
    if not os.path.exists(metadata_json):
        metadata_json = None

    summary_json = os.path.join(files_dir, "wandb-summary.json")
    if not os.path.exists(summary_json):
        summary_json = None

    return {
        "run_dir": run_dir,
        "files_dir": files_dir,
        "config_json": config_json,
        "metadata_json": metadata_json,
        "summary_json": summary_json,
    }


def _load_wandb_config_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cfg = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and "value" in v:
            cfg[k] = v["value"]
        else:
            cfg[k] = v
    return cfg


def _apply_wandb_config_to_args(args, cfg: dict):
    mapping = {
        "architecture_backbone": "backbone",
        "architecture_model": "model",
        "finetune_backbone": "finetune",
        "num_ft_blocks": "num_ft_blocks",
        "ties": "ties",
        "gaze_mode": "gaze",
        "rank_dropout": "rank_dropout",
        "cross_dropout": "cross_dropout",
        "attn_w": "attn_w",
        "rank_w": "rank_w",
        "ties_w": "ties_w",
        "rank_margin": "ranking_margin",
        "rank_margin_ties": "ranking_margin_ties",
    }

    for src_key, dst_attr in mapping.items():
        if src_key in cfg and cfg[src_key] is not None:
            setattr(args, dst_attr, cfg[src_key])

    for b in ["ties", "finetune"]:
        v = getattr(args, b, False)
        if isinstance(v, str):
            setattr(args, b, v.lower() in ("1", "true", "yes", "y", "t"))

    if getattr(args, "gaze", None) is None:
        args.gaze = "use"


def _load_metadata_args_list(metadata_path: str) -> List[str]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    args_list = meta.get("args", [])
    if not isinstance(args_list, list):
        raise ValueError("wandb-metadata.json 'args' is not a list.")
    return args_list


def _apply_train_cli_args_to_test_args(args, train_cli_args: List[str]):
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)

    p.add_argument("--model", type=str)
    p.add_argument("--backbone", type=str)

    p.add_argument("--finetune", nargs="?", const=True, type=str2bool)

    p.add_argument("--num_ft_blocks", type=int)
    p.add_argument("--rank_dropout", type=float)
    p.add_argument("--cross_dropout", type=float)

    p.add_argument("--ties", nargs="?", const=True, type=str2bool)

    p.add_argument("--gaze_mode", type=str)
    p.add_argument("--attn_w", type=float)

    p.add_argument("--rank_w", type=float)
    p.add_argument("--ties_w", type=float)
    p.add_argument("--ranking_margin", type=float)
    p.add_argument("--ranking_margin_ties", type=float)

    p.add_argument("--pooling", type=str)
    p.add_argument("--pool_k", type=int)
    p.add_argument("--use_seg", nargs="?", const=True, type=str2bool)
    p.add_argument("--full_accuracy", nargs="?", const=True, type=str2bool)

    p.add_argument("--gaze_root", type=str)
    p.add_argument("--gaze_subdir_fmt", type=str)
    p.add_argument("--gaze_map_size", type=str)

    p.add_argument("--attention_mode", type=str)
    p.add_argument("--attn_topk", type=int)

    p.add_argument("--use_class_weights", nargs="?", const=True, type=str2bool)
    p.add_argument("--label_smoothing", type=float)

    p.add_argument("--cnn_pool", type=str)  # only relevant for CNN backbones


    known, _unknown = p.parse_known_args(train_cli_args)

    for k, v in vars(known).items():
        if v is None:
            continue
        setattr(args, k, v)

    gm = str(getattr(args, "gaze_mode", getattr(args, "gaze", "off"))).lower().strip()
    
    # Legacy value mapping for older metadata
    if gm in ("use", "on", "true", "1"):
        gm = "align"
    elif gm == "only":
        gm = "align"
        if getattr(args, "eyetracker_only", None) is None:
            args.eyetracker_only = True
    
    if gm not in ("off", "align", "guide", "align+guide"):
        gm = "off"
    
    args.gaze_mode = gm
    args.gaze = gm


def _load_summary_json(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# -----------------------------
# Data loading
# -----------------------------
def read_data(args) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and filter comparisons dataframe.

    Filters:
      - --cities (if not "all")
      - --gaze only  (requires has_eyetracker True)
      - --ties off   (drops score==0 and maps {-1,+1}->{0,1})
      - --ties on    (maps {-1,0,+1}->{0,1,2})
    """
    t0 = time.time()
    try:
        df = pickle.load(open(args.comparisons, "rb"))
    except Exception:
        df = pd.read_pickle(args.comparisons)

    load_sec = time.time() - t0

    expected_cols = [
        "score",
        "image_l", "image_r",
        "dataset",
        "has_eyetracker",
        "npy_file_l", "npy_file_r",
        "survey_id", "trial_id",
    ]
    existing = [c for c in expected_cols if c in df.columns]
    df = df[existing].copy()

    before_rows = len(df)

    if "image_l" in df.columns:
        df["image_l"] = df["image_l"].astype(str).apply(lambda x: x if x.lower().endswith(".jpg") else f"{x}.jpg")
    if "image_r" in df.columns:
        df["image_r"] = df["image_r"].astype(str).apply(lambda x: x if x.lower().endswith(".jpg") else f"{x}.jpg")

    selected_cities = None
    if "dataset" in df.columns and args.cities.lower() != "all":
        selected_cities = [c.strip() for c in args.cities.split(",") if c.strip()]
        df = df[df["dataset"].isin(selected_cities)].copy()

    gaze_mode = args.gaze
    gaze_only_kept = None
    if "has_eyetracker" in df.columns:
        df["has_eyetracker"] = (
            df["has_eyetracker"]
            .replace({"True": True, "False": False, "true": True, "false": False})
            .fillna(False)
            .astype(bool)
        )
        if bool(getattr(args, "eyetracker_only", False)):
            df = df[df["has_eyetracker"].astype(bool)]


    ties_mode = bool(args.ties)
    dropped_ties = 0
    if not ties_mode:
        if "score" in df.columns:
            dropped_ties = int((df["score"] == 0).sum())
        df = df[df["score"] != 0].copy()
        df["score_classification"] = df["score"].replace({-1: 0, +1: 1})
    else:
        df["score_classification"] = df["score"] + 1

    after_rows = len(df)

    stats = {
        "load_seconds": round(load_sec, 3),
        "rows_before_filters": before_rows,
        "rows_after_filters": after_rows,
        "cities_filter": selected_cities if selected_cities is not None else "all",
        "gaze_mode": gaze_mode,
        "ties_mode": "ON" if ties_mode else "OFF",
        "dropped_ties_rows": dropped_ties,
        "gaze_only_rows_before_filter": gaze_only_kept,
        "columns_present": existing,
    }
    return df, stats


# -----------------------------
# Model construction
# -----------------------------
def build_model(args):
    TRANSFORMER_BACKBONES = [
        # --- The "Power 5" ---
        "dinov3_vitb16",
        "beitv2_base_patch16_224",
        "deit3_base_patch16_224",
        "siglip_base_patch16_224",
        "vit_base_patch16_clip_224",
    
        # --- Modern High-Performance Transformers ---
        "dinov2_base",        # NEW: DINOv2 (no registers)
        "dinov2_reg_base",    # DINOv2 + registers
        "eva02_base",
    
        # --- Legacy / Canonical Transformers ---
        "vit_base_patch16_224",  # NEW: Original ViT-B/16 (21k -> 1k)
        "vit_base_dino",
        "vit_small",
        "deit_base",
        "deit_small",
        "deit_tiny",
        "deit_base_distilled",
    ]
    CNN_BACKBONES = {"alex", "vgg", "dense", "resnet"}

    if args.backbone in TRANSFORMER_BACKBONES:
        from nets.transformer import Transformer as Net

        backbone_model, model_specs = resolve_backbone(args.backbone, pretrained=True, strict=True)
        use_gaze_loss = (
            args.model == "rsscnn"
            and args.gaze != "off"
            and float(getattr(args, "attn_w", 0.0)) > 0.0
        )
        
        attn_kwargs = {}


        # -----------------------------
        # Gaze mode -> injection flag
        # -----------------------------
        gaze_mode = str(getattr(args, "gaze_mode", getattr(args, "gaze", "off"))).lower().strip()
        if gaze_mode not in ("off", "align", "guide", "align+guide"):
            gaze_mode = "off"
        use_gaze_inj = gaze_mode in ("guide", "align+guide")
        
        # -----------------------------
        # Attention maps policy
        # -----------------------------
        # losses.py computes KL diagnostics in RSSCNN and expects attn_map keys.
        need_attn_maps = (str(getattr(args, "model", "")).lower().strip() == "rsscnn")
        
        use_attn_hook = bool(need_attn_maps)
        return_attn = bool(need_attn_maps)
        
        ms = int(getattr(args, "gaze_map_size_int", 14))
        attn_out_hw = tuple(getattr(args, "gaze_grid_size", (ms, ms)))
        
        # Transformer expects attention_mode in {"last","rollout","topk"}
        attn_mode = str(getattr(args, "attention_mode", "last")).lower().strip()
        if attn_mode == "cls":
            attn_mode = "last"
        
        attn_topk = getattr(args, "attn_topk", None)
        if attn_topk is not None:
            attn_topk = int(attn_topk)
        
        # -----------------------------
        # Instantiate wrapper (match transformer.py signature)
        # -----------------------------
        net = Net(
            backbone=backbone_model,
            model=str(getattr(args, "model", "rsscnn")).lower().strip(),
        
            pooling=str(getattr(args, "pooling", "cls")).lower().strip(),
            pool_k=int(getattr(args, "pool_k", 10)),
        
            num_classes=3 if bool(getattr(args, "ties", False)) else 2,
        
            finetune=bool(getattr(args, "finetune", False)),
            num_ft_blocks=int(getattr(args, "num_ft_blocks", 1)),
        
            rank_dropout=float(getattr(args, "rank_dropout", 0.0) or 0.0),
            cross_dropout=float(getattr(args, "cross_dropout", 0.0) or 0.0),
        
            use_attn_hook=use_attn_hook,
            return_attn=return_attn,
            attn_out_hw=attn_out_hw,
            attention_mode=attn_mode,
            attn_topk=attn_topk,
        
            use_gaze_injection=bool(use_gaze_inj),
        )
        
        net.attn_grad = bool(need_attn_maps)
        return net

    if args.backbone in CNN_BACKBONES:
        from nets.cnn import CNN as Net
    
        cnn_backbones = {
            "alex": tv_models.alexnet,
            "vgg": tv_models.vgg19,
            "dense": tv_models.densenet121,
            "resnet": tv_models.resnet50,
        }
    
        flatten_spatial = (getattr(args, "cnn_pool", "gap") == "flatten")
    
        return Net(
            backbone=cnn_backbones[args.backbone],
            model=args.model,
            finetune=args.finetune,
            num_classes=3 if args.ties else 2,
            flatten_spatial=flatten_spatial,
        )

    raise ValueError(f"Unknown backbone: {args.backbone}")


def _load_checkpoint(net: torch.nn.Module, ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load checkpoint into the model and report key compatibility information.
    """
    t0 = time.time()
    obj = torch.load(ckpt_path, map_location=device)

    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    def _strip_prefix(sd, prefix: str):
        if isinstance(sd, dict) and len(sd) > 0 and all(k.startswith(prefix) for k in sd.keys()):
            return {k[len(prefix):]: v for k, v in sd.items()}
        return sd

    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "_orig_mod.")

    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        sample_m = missing[:12] if missing else []
        sample_u = unexpected[:12] if unexpected else []
        raise RuntimeError(
            "Checkpoint/model mismatch during evaluation.\n"
            f"  ckpt: {ckpt_path}\n"
            f"  missing_keys (sample): {sample_m}\n"
            f"  unexpected_keys (sample): {sample_u}\n"
            "This usually means you changed one of: model head (rcnn/sscnn/rsscnn), "
            "ties on/off (2 vs 3 classes), backbone, pooling mode, or finetune block config.\n"
            "Fix by instantiating the exact same architecture/config as training, "
            "or evaluate the correct checkpoint."
        )

    load_sec = time.time() - t0
    return {
        "checkpoint_path": ckpt_path,
        "checkpoint_tensors": len(state_dict) if isinstance(state_dict, dict) else None,
        "load_seconds": round(load_sec, 3),
        "missing_keys_count": 0,
        "unexpected_keys_count": 0,
    }
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Checkpoint evaluation", allow_abbrev=False)

    p.add_argument("--log_console", nargs="?", const=True, default=True, type=str2bool)
    p.add_argument("--comparisons", type=str, required=True, help="Pickle file with comparisons dataframe.")
    p.add_argument("--dataset", type=str, required=True, help="Images root directory (e.g., images/).")
    p.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path or filename under --model_dir.")

    p.add_argument("--wandb_config", type=str, default=None, help="Path to wandb run files/config.json.")
    p.add_argument("--wandb_run_id", type=str, default=None, help="W&B run id (e.g., bz6cgldz) to auto-load config locally.")
    p.add_argument("--wandb_dir", type=str, default="wandb", help="Local W&B directory (default: wandb/).")

    p.add_argument("--cuda", nargs="?", const=True, default=False, type=str2bool, help="Enable CUDA if available.")
    p.add_argument("--cuda_id", type=int, default=0, help="CUDA device id.")
    
    p.add_argument("--batch_size", type=int, default=128, help="Evaluation batch size.")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")

    p.add_argument("--cities", type=str, default="all", help='all or comma-separated dataset values (e.g., "berlin,paris").')
    
    p.add_argument("--ties", nargs="?", const=True, default=False, type=str2bool, help="Enable ties (3-class).")
    p.add_argument(
        "--gaze_mode",
        type=str,
        default="off",
        choices=["off", "align", "guide", "align+guide"],
        help="off | align (KL) | guide (injection) | align+guide (both)",
    )
    p.add_argument("--gaze_root", type=str, default="Eyetracker_attention_maps", help="Folder for .npy gaze maps.")
    p.add_argument("--use_seg", nargs="?", const=True, default=False, type=str2bool, help="Use *_seg.jpg images.")

    p.add_argument("--model", choices=["rcnn", "sscnn", "rsscnn"], default="rcnn", help="Head type used in training.")
    p.add_argument("--backbone", type=str, default="vit_base_dino", help="Backbone used in training.")
    
    p.add_argument("--finetune", nargs="?", const=True, default=False, type=str2bool, help="If backbone was finetuned.")
    p.add_argument("--num_ft_blocks", type=int, default=1, help="Transformer blocks unfrozen (if finetune).")
    
    p.add_argument("--rank_dropout", type=float, default=0.3, help="Ranking head dropout (Transformer).")
    p.add_argument("--cross_dropout", type=float, default=0.3, help="Classification head dropout (Transformer).")

    p.add_argument("--full_accuracy", nargs="?", const=True, default=False, type=str2bool, help="Use margin-based accuracy for ranking.")
    
    p.add_argument("--ranking_margin", type=float, default=0.3, help="Margin for non-ties ranking.")
    p.add_argument("--ranking_margin_ties", type=float, default=None, help="Margin for ties loss (if used).")
    p.add_argument("--rank_w", type=float, default=1.0, help="Ranking loss weight.")
    p.add_argument("--ties_w", type=float, default=1.0, help="Ties loss weight.")
    p.add_argument("--attn_w", type=float, default=0.0, help="Gaze KL weight.")

    p.add_argument("--model_dir", type=str, default="models/", help="Used if --checkpoint is not an absolute path.")
    p.add_argument("--notes", type=str, default="", help="Prefix for outputs/saved/* filename.")
    p.add_argument("--seed", type=int, default=7, help="Random seed.")
    
    #p.add_argument("--gaze_map_size", default="auto", help="Gaze folder size: 'auto' or integer (e.g., 14, 16).")


    p.add_argument("--cnn_pool",type=str,default="flatten",choices=["gap", "flatten"],help="CNN feature pooling: gap (global average pool) or flatten (flatten spatial grid).",)


    # --- ADDED Missing Pooling Arguments ---
    p.add_argument(
    "--pooling",
    type=str,
    default="cls",
    choices=[
        "cls",
        "mean",
        "patch_mean",
        "reg_mean",
        "prefix_mean",
        "cls_reg_concat",
        "cls_reg_add",
        "concat",
        "topk",
        "max",
        "cls_max_concat",
    ],
    help="Feature pooling strategy (must match training).",
)

    p.add_argument("--pool_k", type=int, default=10, help="Number of patches to keep when using --pooling topk")
    p.add_argument(
        "--attention_mode",
        type=str,
        default="last",
        choices=["last", "rollout", "topk"],
        help="Attention map extraction mode (must match training when using gaze loss).",
    )
    p.add_argument(
        "--attn_topk",
        type=int,
        default=None,
        help="Top-k patches for attention_mode=topk (must match training).",
    )
    
    return p


def _resolve_config_source(args) -> Tuple[str, Dict[str, Any]]:
    """
    Apply W&B config to args and return (source_label, source_details).
    """
    source_details: Dict[str, Any] = {}
    if args.wandb_config:
        cfg = _load_wandb_config_json(args.wandb_config)
        _apply_wandb_config_to_args(args, cfg)
        source_details["wandb_config"] = args.wandb_config
        return "wandb_config", source_details

    if args.wandb_run_id:
        run_files = _find_wandb_run_files(args.wandb_run_id, args.wandb_dir)
        source_details.update(run_files)

        if run_files["config_json"]:
            cfg = _load_wandb_config_json(run_files["config_json"])
            _apply_wandb_config_to_args(args, cfg)
            return "wandb_run_id:config.json", source_details

        if run_files["metadata_json"]:
            train_cli_args = _load_metadata_args_list(run_files["metadata_json"])
            _apply_train_cli_args_to_test_args(args, train_cli_args)
            return "wandb_run_id:wandb-metadata.json(args)", source_details

        raise FileNotFoundError(
            f"Found run dir '{run_files['run_dir']}' but neither config.json nor wandb-metadata.json exists in files/."
        )

    return "manual_cli", source_details


def main():
    # =============================================================================================== #
    # (STEP 0) WALL-CLOCK START & HEADER
    # =============================================================================================== #
    start_wall = time.time()
    print(_hr())
    print("CHECKPOINT EVALUATION")
    print(f"Start time: {_now()}")
    print(_hr())


    # =============================================================================================== #
    # (STEP 1) ARGUMENT PARSING & CONFIG RESOLUTION
    # =============================================================================================== #
    args = parse_args().parse_args()

    # Determine whether config comes from CLI, W&B, or summary JSON
    config_source, config_details = _resolve_config_source(args)

    # Ensure tie margin is always defined
    if args.ranking_margin_ties is None:
        args.ranking_margin_ties = args.ranking_margin
    _normalize_gaze_mode(args)

    # =============================================================================================== #
    # (STEP 2) REPRODUCIBILITY & DEVICE SETUP
    # =============================================================================================== #
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        f"cuda:{args.cuda_id}"
        if (args.cuda and torch.cuda.is_available())
        else "cpu"
    )


    # =============================================================================================== #
    # (STEP 3) PRINT EFFECTIVE CONFIGURATION
    # =============================================================================================== #
    print("CONFIGURATION")
    print(_hr("-"))

    _print_kv("Config source:", {"source": config_source})
    if config_details:
        _print_kv(
            "Config details:",
            {k: v for k, v in config_details.items() if v is not None}
        )
    print()


    # =============================================================================================== #
    # (STEP 4) EVALUATION INPUT PATHS & RUNTIME SETTINGS
    # =============================================================================================== #
    print("EVALUATION INPUTS")
    print(_hr("-"))

    _print_kv("Paths:", {
        "comparisons": args.comparisons,
        "dataset_root": args.dataset,
        "checkpoint": args.checkpoint,
        "model_dir": args.model_dir,
        "gaze_root": args.gaze_root,
    })

    _print_kv("Device:", {
        "device": str(device),
        "cuda_requested": _fmt_bool(args.cuda),
        "cuda_available": _fmt_bool(torch.cuda.is_available()),
        "cuda_id": args.cuda_id,
    })

    _print_kv("Loader:", {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    })
    print()


    # =============================================================================================== #
    # (STEP 5) MODEL CONFIGURATION SUMMARY
    # =============================================================================================== #
    print("MODEL SETUP")
    print(_hr("-"))

    _print_kv("Architecture:", {
        "model_head": args.model,
        "backbone": args.backbone,
        "finetune": _fmt_bool(args.finetune),
        "num_ft_blocks": args.num_ft_blocks,
        "rank_dropout": args.rank_dropout,
        "cross_dropout": args.cross_dropout,
        "use_seg": _fmt_bool(args.use_seg),
    })

    _print_kv("Modes:", {
        "ties": _fmt_bool(args.ties),
        "gaze": args.gaze,
        "attn_w": args.attn_w,
        "full_accuracy": _fmt_bool(args.full_accuracy),
    })

    _print_kv("Loss parameters:", {
        "rank_w": args.rank_w,
        "ties_w": args.ties_w,
        "ranking_margin": args.ranking_margin,
        "ranking_margin_ties": args.ranking_margin_ties,
    })
    print()


    # =============================================================================================== #
    # (STEP 6) DATASET LOADING & FILTERING SUMMARY
    # =============================================================================================== #
    print("DATASET LOADING")
    print(_hr("-"))

    df, df_stats = read_data(args)
    _print_kv("Filtering summary:", df_stats)

    if "dataset" in df.columns:
        _print_kv("Rows per dataset:", df["dataset"].value_counts().to_dict())

    if "score" in df.columns:
        _print_kv(
            "Score distribution:",
            df["score"].value_counts().sort_index().to_dict()
        )

    if "has_eyetracker" in df.columns:
        _print_kv(
            "Eyetracker availability:",
            {str(k): v for k, v in df["has_eyetracker"].value_counts().to_dict().items()}
        )
    print()


    # =============================================================================================== #
    # (STEP 7) DATASET & DATALOADER CONSTRUCTION  (mirrors train.py)
    # =============================================================================================== #
    
    CNN_BACKBONES = {"alex", "vgg", "dense", "resnet"}
    is_cnn_backbone = str(args.backbone).lower().strip() in CNN_BACKBONES
    
    # Fixed preprocessing for CNNs (same as train.py)
    SPECS = {
        "input_size": (3, 224, 224),
        "crop_pct": 0.875,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "interpolation": "bilinear",
    }
    
    # 1) Resolve model specs + infer token grid (for gaze alignment)
    if is_cnn_backbone:
        model_specs = {
            "alias": args.backbone,
            "timm_id": None,
            **SPECS,
            "img_size": int(SPECS["input_size"][-1]),
        }
        grid_h, grid_w = 14, 14
    else:
        # No need to download weights just to get patch grid; pretrained=False is enough for grid inference
        backbone_tmp, model_specs = resolve_backbone(args.backbone, pretrained=False, strict=True)
        grid_h, grid_w = infer_vit_grid_size(backbone_tmp, model_specs)
        del backbone_tmp
    
    # Optional override to match existing gaze folder layout
    if str(getattr(args, "gaze_map_size", "auto")).lower() != "auto":
        forced = int(args.gaze_map_size)
        grid_h, grid_w = forced, forced
    
    args.gaze_grid_size = (int(grid_h), int(grid_w))
    args.gaze_map_size_int = int(grid_h)
    
    # 2) Build eval transforms (timm-style resize->center-crop->normalize; aligns gaze if enabled)
    use_gaze = (str(args.gaze).lower().strip() != "off")
    
    eval_tfms, eval_meta = build_eval_transforms(
        model_specs,
        gaze_grid_size=args.gaze_grid_size,
        enable_gaze=use_gaze,
    )
    
    # 3) Dataset (note: map_size selects gaze_root/<map_size>x<map_size>/...)
    dataset = ComparisonsDataset(
        dataframe=df,
        root_dir=args.dataset,
        transform=eval_tfms,
        gaze_root=args.gaze_root,
        use_gaze=use_gaze,
        use_seg=args.use_seg,
        logger=None,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


    # =============================================================================================== #
    # (STEP 8) MODEL INSTANTIATION & CHECKPOINT LOADING
    # =============================================================================================== #
    print("CHECKPOINT LOADING")
    print(_hr("-"))

    net = build_model(args).to(device)

    ckpt_path = (
        args.checkpoint
        if os.path.isabs(args.checkpoint)
        else os.path.join(args.model_dir, args.checkpoint)
    )

    ckpt_report = _load_checkpoint(net, ckpt_path, device)
    _print_kv("Checkpoint report:", ckpt_report)
    print()


    # =============================================================================================== #
    # (STEP 9) OPTIONAL W&B SUMMARY INSPECTION
    # =============================================================================================== #
    if args.wandb_run_id and config_details.get("summary_json"):
        summary = _load_summary_json(config_details["summary_json"])
        if summary:
            print("W&B RUN SUMMARY (local)")
            print(_hr("-"))

            keys = [
                "max_accuracy_validation",
                "best_val_acc",
                "final_val_acc",
                "final_test_acc",
                "epoch",
            ]
            _print_kv(
                "Selected metrics:",
                {k: summary.get(k) for k in keys if k in summary}
            )
            print()


    # =============================================================================================== #
    # (STEP 10) MODEL EVALUATION
    # =============================================================================================== #
    print("EVALUATION")
    print(_hr("-"))
    print(f"Evaluation started: {_now()}")

    t_eval0 = time.time()
    test(device, net, dataloader, args, logger=None)
    t_eval = time.time() - t_eval0

    print(f"Evaluation finished: {_now()}")
    print(f"Evaluation duration (seconds): {t_eval:.3f}")
    print()


    # =============================================================================================== #
    # (STEP 11) FINAL TIMING & CLEAN EXIT
    # =============================================================================================== #
    total_sec = time.time() - start_wall
    print(_hr())
    print(f"Done. Total wall time (seconds): {total_sec:.3f}")
    print(_hr())

if __name__ == "__main__":
    main()
