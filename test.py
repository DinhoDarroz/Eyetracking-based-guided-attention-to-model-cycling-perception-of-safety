#!/usr/bin/env python3
"""
test.py

Evaluate a trained checkpoint on a comparisons pickle.

Configuration sources (priority order):
  1) models/<run_id>/ best/last checkpoint auto-discovery
  2) local W&B config.json/config.yaml or wandb-metadata.json["args"]
  3) explicit CLI overrides

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
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as tv_models

from scripts.test_script import test
from data import ComparisonsDataset
from train_main_utils import build_eval_transforms
from train_main_utils import _build_model as build_train_model
from train_main_utils import _build_transforms_and_specs

from backbone_registry import infer_vit_grid_size, resolve_backbone
from gaze_policy import build_gaze_config, normalize_gaze_mode



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


def _as_numpy_1d(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)

    def _item_to_float(v: Any) -> float:
        if torch.is_tensor(v):
            return float(v.detach().cpu().view(-1)[0].item())
        return float(v)

    if isinstance(values, pd.Series):
        return values.map(_item_to_float).to_numpy(dtype=float)
    if torch.is_tensor(values):
        return values.detach().cpu().view(-1).numpy()
    return np.asarray(values, dtype=float).reshape(-1)


def _softmax_positive(logit_left: np.ndarray, logit_right: np.ndarray) -> np.ndarray:
    logits = np.stack([logit_left, logit_right], axis=1)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs[:, 1]


def _add_roc_curve(curves: List[Tuple[str, np.ndarray, np.ndarray]], name: str, y_true: np.ndarray, y_score: np.ndarray) -> None:
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask].astype(int)
    y_score = y_score[mask]

    if y_true.size == 0 or len(set(y_true.tolist())) < 2:
        print(f"[ROC] Skipping {name}: need both positive and negative examples.")
        return

    fpr, tpr, _thresholds = roc_curve(y_true, y_score)
    curves.append((f"{name} AUC={auc(fpr, tpr):.4f}", fpr, tpr))


def _plot_roc_when_ties_off(results_df: pd.DataFrame, args) -> None:
    if bool(getattr(args, "ties", False)):
        print("[ROC] Skipping ROC plot because ties are ON.")
        return
    if results_df is None or results_df.empty:
        print("[ROC] Skipping ROC plot because there are no evaluation results.")
        return

    curves: List[Tuple[str, np.ndarray, np.ndarray]] = []

    if args.model in ("ranking", "multitask", "multitask_gaze") and {"label_r", "rank_left", "rank_right"}.issubset(results_df.columns):
        label_r = _as_numpy_1d(results_df["label_r"])
        rank_left = _as_numpy_1d(results_df["rank_left"])
        rank_right = _as_numpy_1d(results_df["rank_right"])
        non_tie = label_r != 0
        # Match utils.accuracy.RankAUC: positive class is "left is preferred".
        rank_target = (label_r[non_tie] == -1).astype(int)
        rank_score = rank_left[non_tie] - rank_right[non_tie]
        _add_roc_curve(curves, "Ranking", rank_target, rank_score)

    if args.model in ("classification", "multitask", "multitask_gaze") and {"label_c", "logits_l", "logits_r"}.issubset(results_df.columns):
        label_c = _as_numpy_1d(results_df["label_c"]).astype(int)
        logits_l = _as_numpy_1d(results_df["logits_l"])
        logits_r = _as_numpy_1d(results_df["logits_r"])
        # Match utils.accuracy.ClassificationAUC: positive class is class 1.
        class_score = _softmax_positive(logits_l, logits_r)
        _add_roc_curve(curves, "Classification", label_c, class_score)

    if not curves:
        print("[ROC] Skipping ROC plot because no compatible scores were found.")
        return

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, fpr, tpr in curves:
        ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (ties off)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    ckpt_base = Path(getattr(args, "checkpoint", "model")).name
    plot_name = f"{getattr(args, 'notes', '')}_{ckpt_base}_roc.png".lstrip("_")
    plot_path = Path("outputs") / "saved" / plot_name
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"[ROC] Saved ROC plot: {plot_path}")
    plt.show()

def _normalize_and_attach_gaze_mode(args) -> str:
    raw = getattr(args, "gaze_mode", getattr(args, "gaze", None))
    raw_s = str(raw).lower().strip()

    # Backward-compat for older runs / metadata
    if raw_s in ("use", "on", "true", "1"):
        raw_s = "align"
    if raw_s == "only":
        raw_s = "align"
        if getattr(args, "eyetracker_only", None) is None:
            args.eyetracker_only = True

    args.gaze_mode = normalize_gaze_mode(raw_s)
    args.gaze = args.gaze_mode  # legacy alias for older code paths
    return args.gaze_mode


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

    config_yaml = os.path.join(files_dir, "config.yaml")
    if not os.path.exists(config_yaml):
        config_yaml = None

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
        "config_yaml": config_yaml,
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


def _load_wandb_config_yaml(path: str) -> dict:
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot read {path}: PyYAML is not installed. "
            "Install pyyaml or rely on wandb-metadata.json args."
        ) from exc

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
        "backbone": "backbone",
        "model": "model",
        "finetune_backbone": "finetune",
        "finetune": "finetune",
        "num_ft_layers": "num_ft_layers",
        "num_ft_blocks": "num_ft_layers",
        "ties": "ties",
        "gaze_mode": "gaze_mode",
        "eyetracker_filter": "eyetracker_filter",
        "rank_dropout": "rank_dropout",
        "cross_dropout": "cross_dropout",
        "attn_w": "attn_w",
        "attn_layer": "attn_layer",
        "attention_mode": "attention_mode",
        "attn_topk": "attn_topk",
        "rank_w": "rank_w",
        "ties_w": "ties_w",
        "rank_margin": "ranking_margin",
        "rank_margin_ties": "ranking_margin_ties",
        "ranking_margin": "ranking_margin",
        "ranking_margin_ties": "ranking_margin_ties",
        "pooling": "pooling",
        "pool_k": "pool_k",
        "cnn_pool": "cnn_pool",
        "use_seg": "use_seg",
        "full_accuracy": "full_accuracy",
        "gaze_root": "gaze_root",
        "gaze_map_size": "gaze_map_size",
        "guidance_drop_prob": "guidance_drop_prob",
        "guidance_strength": "guidance_strength",
        "guidance_bottleneck_dim": "guidance_bottleneck_dim",
        "guidance_gaze_hidden_dim": "guidance_gaze_hidden_dim",
        "guidance_conv_hidden_channels": "guidance_conv_hidden_channels",
        "guide_train_only": "guide_train_only",
        "egvit_mask_type": "egvit_mask_type",
        "egvit_keep_ratio": "egvit_keep_ratio",
        "egvit_focus_hw": "egvit_focus_hw",
        "egvit_drop_prob": "egvit_drop_prob",
        "egvit_train_only": "egvit_train_only",
        "comparisons": "comparisons",
        "dataset": "dataset",
        "cities": "cities",
        "batch_size": "batch_size",
        "seed": "seed",
    }

    for src_key, dst_attr in mapping.items():
        if src_key in cfg and cfg[src_key] is not None:
            if dst_attr in getattr(args, "_cli_overrides", set()):
                continue
            setattr(args, dst_attr, cfg[src_key])

    for b in ["ties", "finetune", "use_seg", "full_accuracy", "guide_train_only", "egvit_train_only"]:
        v = getattr(args, b, False)
        if isinstance(v, str):
            setattr(args, b, v.lower() in ("1", "true", "yes", "y", "t"))


def _run_dir_mtime(run_dir: Path) -> float:
    pts = list(run_dir.glob("*.pt"))
    files = pts or list(run_dir.iterdir())
    return max((p.stat().st_mtime for p in files), default=run_dir.stat().st_mtime)


def _select_latest_run_dir(model_dir: str) -> Path:
    root = Path(model_dir)
    candidates = [
        p for p in root.iterdir()
        if p.is_dir() and p.name != ".ipynb_checkpoints" and list(p.glob("*.pt"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No trained run folders with .pt files found under '{model_dir}'.")
    return max(candidates, key=_run_dir_mtime)


def _read_run_info(run_dir: Path) -> Dict[str, Any]:
    info_path = run_dir / "run_info.json"
    if not info_path.exists():
        return {}
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _checkpoint_sort_key(path: Path) -> Tuple[int, float]:
    numbers = [int(x) for x in re.findall(r"\d+", path.stem)]
    epoch = numbers[0] if numbers else -1
    return epoch, path.stat().st_mtime


def _select_checkpoint_from_run(run_dir: Path, preference: str = "best") -> Path:
    preference = str(preference or "best").lower().strip()
    patterns = ["best_model_*.pt", "best*.pt"] if preference == "best" else ["last_model_*.pt", "last*.pt"]
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(run_dir.glob(pattern))

    if not matches and preference == "best":
        return _select_checkpoint_from_run(run_dir, "last")
    if not matches and preference == "last":
        return _select_checkpoint_from_run(run_dir, "best")
    if not matches:
        raise FileNotFoundError(f"No checkpoint files found in '{run_dir}'.")

    return sorted(set(matches), key=_checkpoint_sort_key)[-1]


def _resolve_run_and_checkpoint(args) -> Dict[str, Any]:
    details: Dict[str, Any] = {}

    run_dir: Optional[Path] = None
    if args.run_id:
        run_dir = Path(args.model_dir) / args.run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder not found: {run_dir}")
    elif args.checkpoint is None:
        run_dir = _select_latest_run_dir(args.model_dir)
    else:
        ckpt_parent = Path(args.checkpoint).parent
        if str(ckpt_parent) == ".":
            ckpt_parent = Path(args.model_dir) / ckpt_parent
        if (ckpt_parent / "run_info.json").exists():
            run_dir = ckpt_parent

    if run_dir is not None:
        info = _read_run_info(run_dir)
        args.run_id = args.run_id or info.get("run_id") or run_dir.name
        args.wandb_run_id = args.wandb_run_id or args.run_id
        if args.checkpoint is None:
            args.checkpoint = str(_select_checkpoint_from_run(run_dir, args.checkpoint_kind))
        elif not os.path.isabs(args.checkpoint) and not os.path.exists(args.checkpoint):
            run_candidate = run_dir / args.checkpoint
            if run_candidate.exists():
                args.checkpoint = str(run_candidate)
        details.update({
            "run_dir": str(run_dir),
            "run_id": args.run_id,
            "run_name": info.get("run_name"),
            "checkpoint_kind": args.checkpoint_kind,
        })

    if args.checkpoint is None:
        raise ValueError("No checkpoint selected. Provide --run_id, --checkpoint, or keep trained runs under --model_dir.")

    if not os.path.isabs(args.checkpoint) and not os.path.exists(args.checkpoint):
        candidate = os.path.join(args.model_dir, args.checkpoint)
        if os.path.exists(candidate):
            args.checkpoint = candidate

    return details

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
    p.add_argument("--comparisons", type=str)
    p.add_argument("--dataset", type=str)
    p.add_argument("--cities", type=str)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--seed", type=int)

    p.add_argument("--finetune", nargs="?", const=True, type=str2bool)

    p.add_argument("--num_ft_layers", type=int)
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
    p.add_argument("--eyetracker_filter", type=str)

    p.add_argument("--gaze_root", type=str)
    p.add_argument("--gaze_subdir_fmt", type=str)
    p.add_argument("--gaze_map_size", type=str)

    p.add_argument("--attention_mode", type=str)
    p.add_argument("--attn_layer", type=int)
    p.add_argument("--attn_topk", type=int)

    p.add_argument("--guidance_drop_prob", type=float)
    p.add_argument("--guidance_strength", type=float)
    p.add_argument("--guidance_bottleneck_dim", type=int)
    p.add_argument("--guidance_gaze_hidden_dim", type=int)
    p.add_argument("--guidance_conv_hidden_channels", type=int)
    p.add_argument("--guide_train_only", nargs="?", const=True, type=str2bool)
    p.add_argument("--egvit_mask_type", type=str)
    p.add_argument("--egvit_keep_ratio", type=float)
    p.add_argument("--egvit_focus_hw", type=int, nargs=2)
    p.add_argument("--egvit_drop_prob", type=float)
    p.add_argument("--egvit_train_only", nargs="?", const=True, type=str2bool)

    p.add_argument("--use_class_weights", nargs="?", const=True, type=str2bool)
    p.add_argument("--label_smoothing", type=float)

    p.add_argument("--cnn_pool", type=str)  # only relevant for CNN backbones


    known, _unknown = p.parse_known_args(train_cli_args)

    for k, v in vars(known).items():
        if v is None:
            continue
        if k in getattr(args, "_cli_overrides", set()):
            continue
        setattr(args, k, v)

    raw = getattr(args, "gaze_mode", getattr(args, "gaze", None))
    raw_s = str(raw).lower().strip()
    
    # Backward-compat for older metadata values
    if raw_s in ("use", "on", "true", "1"):
        raw_s = "align"
    if raw_s == "only":
        raw_s = "align"
        if getattr(args, "eyetracker_only", None) is None:
            args.eyetracker_only = True
    
    args.gaze_mode = normalize_gaze_mode(raw_s)
    args.gaze = args.gaze_mode


def _normalize_finetune_layer_args(args) -> None:
    new_value = getattr(args, "num_ft_layers", None)
    legacy_value = getattr(args, "num_ft_blocks", None)
    if new_value is None:
        new_value = 1 if legacy_value is None else legacy_value
    args.num_ft_layers = int(new_value)
    if hasattr(args, "num_ft_blocks"):
        delattr(args, "num_ft_blocks")


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
        if str(getattr(args, "eyetracker_filter", "all")).lower().strip() == "only":
            gaze_only_kept = int(df["has_eyetracker"].sum())
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
        "dinov2_base",
        "dinov2_reg_base",
        "eva02_base",

        # --- Legacy / Canonical Transformers ---
        "vit_base_patch16_224",
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
        out_size = int(model_specs.get("img_size", model_specs["input_size"][-1]))

        gaze_cfg = getattr(args, "gaze_cfg", None)
        if gaze_cfg is None:
            gaze_cfg = build_gaze_config(args, is_cnn_backbone=False, out_size=out_size)

        use_attn_hook = bool(getattr(gaze_cfg, "need_attn_maps", False))
        return_attn = bool(getattr(gaze_cfg, "need_attn_maps", False))

        grid = getattr(args, "gaze_grid_size", None)
        if grid is None:
            ms = int(getattr(args, "gaze_map_size_int", 14))
            grid = (ms, ms)
        attn_out_hw = tuple(int(x) for x in grid)

        attn_mode = str(getattr(args, "attention_mode", "last")).lower().strip()
        if attn_mode == "cls":
            attn_mode = "last"

        attn_topk = getattr(args, "attn_topk", None)
        if attn_topk is not None:
            attn_topk = int(attn_topk)

        net = Net(
            backbone=backbone_model,
            model=str(getattr(args, "model", "multitask_gaze")).lower().strip(),
            pooling=str(getattr(args, "pooling", "cls")).lower().strip(),
            pool_k=int(getattr(args, "pool_k", 10)),
            num_classes=3 if bool(getattr(args, "ties", False)) else 2,
            finetune=bool(getattr(args, "finetune", False)),
            num_ft_layers=int(getattr(args, "num_ft_layers", getattr(args, "num_ft_blocks", 1))),
            rank_dropout=float(getattr(args, "rank_dropout", 0.0) or 0.0),
            cross_dropout=float(getattr(args, "cross_dropout", 0.0) or 0.0),
            use_attn_hook=use_attn_hook,
            return_attn=return_attn,
            attn_out_hw=attn_out_hw,
            attention_mode=attn_mode,
            attn_topk=attn_topk,
            use_gaze_injection=bool(getattr(gaze_cfg, "inject", False)),
        )

        net.attn_grad = bool(getattr(gaze_cfg, "need_attn_maps", False))
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
            "This usually means you changed one of: model head (ranking/classification/multitask/multitask_gaze), "
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
    p.add_argument("--comparisons", type=str, default=None, help="Pickle file with comparisons dataframe. Defaults to the training run value when available.")
    p.add_argument("--dataset", type=str, default=None, help="Images root directory. Defaults to the training run value when available.")
    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path. Defaults to the selected run's best checkpoint.")
    p.add_argument("--checkpoint_kind", type=str, default="best", choices=["best", "last"], help="Which checkpoint to auto-select from the run folder.")
    p.add_argument("--run_id", "--wandb_run_id", dest="run_id", type=str, default=None, help="Run id matching models/<run_id>. If omitted, the newest run folder is used.")

    p.add_argument("--wandb_config", type=str, default=None, help="Path to wandb run files/config.json.")
    p.add_argument("--wandb_dir", type=str, default="wandb", help="Local W&B directory (default: wandb/).")

    p.add_argument("--cuda", nargs="?", const=True, default=False, type=str2bool, help="Enable CUDA if available.")
    p.add_argument("--cuda_id", type=int, default=0, help="CUDA device id.")
    
    p.add_argument("--batch_size", type=int, default=128, help="Evaluation batch size.")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")

    p.add_argument("--cities", type=str, default="all", help='all or comma-separated dataset values (e.g., "berlin,paris").')
    p.add_argument("--eyetracker_filter", type=str, default="all", choices=["all", "only"], help="Match train.py eyetracker filtering.")
    
    p.add_argument("--ties", nargs="?", const=True, default=False, type=str2bool, help="Enable ties (3-class).")
    p.add_argument(
        "--gaze_mode",
        type=str,
        default="disable",
        help="disable | diag | guide | align | align+gaze (legacy accepted: off, align+guide, use, only)",
    )

    p.add_argument("--gaze_root", type=str, default="Eyetracker_attention_maps", help="Folder for .npy gaze maps.")
    p.add_argument("--use_seg", nargs="?", const=True, default=False, type=str2bool, help="Use *_seg.jpg images.")

    p.add_argument("--model", choices=["ranking", "classification", "multitask", "multitask_gaze"], default="ranking", help="Head type used in training.")
    p.add_argument("--backbone", type=str, default="vit_base_dino", help="Backbone used in training.")
    
    p.add_argument("--finetune", nargs="?", const=True, default=False, type=str2bool, help="If backbone was finetuned.")
    p.add_argument("--num_ft_layers", type=int, default=None, help="Last ViT encoder layers unfrozen (if finetune); earlier layers stay frozen.")
    p.add_argument("--num_ft_blocks", type=int, default=None, help=argparse.SUPPRESS)
    
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
    
    p.add_argument("--gaze_map_size", default="auto", help="Gaze folder size: 'auto' or integer (e.g., 14, 16).")


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
        default="raw",
        choices=["raw", "last", "rollout", "topk"],
        help="Attention map extraction mode (must match training when using gaze loss).",
    )
    p.add_argument("--attn_layer", type=int, default=-1, help="Transformer block index used when --attention_mode=raw.")
    p.add_argument(
        "--attn_topk",
        type=int,
        default=None,
        help="Top-k patches for attention_mode=topk (must match training).",
    )

    p.add_argument("--guidance_drop_prob", type=float, default=0.0)
    p.add_argument("--guidance_strength", type=float, default=1.0)
    p.add_argument("--guidance_bottleneck_dim", type=int, default=20)
    p.add_argument("--guidance_gaze_hidden_dim", type=int, default=30)
    p.add_argument("--guidance_conv_hidden_channels", type=int, default=64)
    p.add_argument("--guide_train_only", nargs="?", const=True, default=True, type=str2bool)
    p.add_argument("--egvit_mask_type", type=str, default="separated", choices=["separated", "focused"])
    p.add_argument("--egvit_keep_ratio", type=float, default=0.25)
    p.add_argument("--egvit_focus_hw", type=int, nargs=2, default=(7, 7))
    p.add_argument("--egvit_drop_prob", type=float, default=0.0)
    p.add_argument("--egvit_train_only", nargs="?", const=True, default=True, type=str2bool)
    
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

        if run_files["config_yaml"]:
            try:
                cfg = _load_wandb_config_yaml(run_files["config_yaml"])
                _apply_wandb_config_to_args(args, cfg)
                return "wandb_run_id:config.yaml", source_details
            except RuntimeError as exc:
                source_details["config_yaml_error"] = str(exc)

        if run_files["metadata_json"]:
            train_cli_args = _load_metadata_args_list(run_files["metadata_json"])
            _apply_train_cli_args_to_test_args(args, train_cli_args)
            return "wandb_run_id:wandb-metadata.json(args)", source_details

        raise FileNotFoundError(
            f"Found run dir '{run_files['run_dir']}' but no config.json, config.yaml, or wandb-metadata.json exists in files/."
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
    cli_overrides = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            name = token.split("=", 1)[0].lstrip("-").replace("-", "_")
            if name == "wandb_run_id":
                name = "run_id"
            cli_overrides.add(name)
    args._cli_overrides = cli_overrides
    args.wandb_run_id = args.run_id

    run_details = _resolve_run_and_checkpoint(args)

    # Determine whether config comes from CLI, W&B, or summary JSON
    config_source, config_details = _resolve_config_source(args)
    _normalize_finetune_layer_args(args)
    config_details = {**run_details, **config_details}

    if args.comparisons is None:
        raise ValueError(
            "No comparisons file configured. Pass --comparisons or keep local W&B metadata "
            "for the selected run so test.py can recover the training value."
        )
    if args.dataset is None:
        raise ValueError(
            "No dataset root configured. Pass --dataset or keep local W&B metadata "
            "for the selected run so test.py can recover the training value."
        )

    # Ensure tie margin is always defined
    if args.ranking_margin_ties is None:
        args.ranking_margin_ties = args.ranking_margin
    _normalize_and_attach_gaze_mode(args)

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
        "num_ft_layers": args.num_ft_layers,
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
    backbone_model, _train_tfms, eval_tfms, is_cnn_backbone = (
        _build_transforms_and_specs(args)
    )
    enable_gaze = bool(getattr(args.gaze_cfg, "load_gaze", False))
    
    dataset = ComparisonsDataset(
        dataframe=df,
        root_dir=args.dataset,
        transform=eval_tfms,
        gaze_root=args.gaze_root,
        use_gaze=enable_gaze,
        use_seg=args.use_seg,
        logger=None,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )


    # =============================================================================================== #
    # (STEP 8) MODEL INSTANTIATION & CHECKPOINT LOADING
    # =============================================================================================== #
    print("CHECKPOINT LOADING")
    print(_hr("-"))

    net = build_train_model(
        args=args,
        backbone_model=backbone_model,
        is_cnn_backbone=is_cnn_backbone,
    )
    net.to(device)

    ckpt_path = args.checkpoint

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
    results_df = test(device, net, dataloader, args, logger=None)
    t_eval = time.time() - t_eval0

    print(f"Evaluation finished: {_now()}")
    print(f"Evaluation duration (seconds): {t_eval:.3f}")
    _plot_roc_when_ties_off(results_df, args)
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
