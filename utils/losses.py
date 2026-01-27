"""
utils/losses.py

Loss building blocks and orchestration for pairwise subjective-safety learning.

This module centralizes:
  - Pairwise ranking losses (non-ties) where a preferred side must score higher
  - Tie losses where the model is encouraged to predict similar scores
  - Classification loss (CrossEntropy) for discrete label prediction
  - Optional gaze/attention alignment loss via symmetric KL divergence

It is designed to be called from the training loop with:
    total_loss = compute_loss(args, network_output_dict, labels)

Label conventions in this project:
  - labels["label_r"] is the raw ranking label in {-1, 0, +1}
        -1: left wins
         0: tie
        +1: right wins

Expected network_output_dict structure:
  - network_output_dict["left"]["output"]   : Tensor [B] or [B,1]
  - network_output_dict["right"]["output"]  : Tensor [B] or [B,1]
  - network_output_dict["logits"]["output"] : Tensor [B, C] (if classification head exists)
  - network_output_dict["left"]["attn_map"] : Tensor [B, H, W] (optional, for gaze KL)
  - network_output_dict["right"]["attn_map"]: Tensor [B, H, W] (optional, for gaze KL)

Expected labels dict structure:
  - labels["label_r"] : Tensor [B] in {-1,0,+1}
  - labels["label_c"] : Tensor [B] in {0..C-1} (if classification is used)
  - labels["gaze_l"] / labels["gaze_r"] : Tensor [B, H, W] (optional, gaze maps)
  - labels["has_eye_mask"] : Tensor [B] in {0,1} (optional, mask for gaze availability)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import sys
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "SmoothPairwiseRankingLoss",
    "TieHuberLoss",
    "MarginRankingLossWithTies",
    "compute_ranking_loss",
    "compute_loss_classification",
    "normalize_to_prob",
    "attention_kl_loss",
    "compute_loss",
]


# ====================================================================================== #
# Loss primitives
# ====================================================================================== #

class SmoothPairwiseRankingLoss(nn.Module):
    """
    Smooth pairwise ranking loss (RankNet-style / logistic).

    For a non-tie pair:
        diff = s_left - s_right
        y ∈ {-1, +1}
        loss = softplus(-y * diff) = log(1 + exp(-y * diff))

    Compared to hinge-based MarginRankingLoss:
      - smooth (no kink)
      - provides non-zero gradients even when predictions are correct
      - tends to be more robust for noisy, subjective labels

    Notes:
      - This loss assumes `target` is +1 when left should be larger than right,
        and -1 when right should be larger than left.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean",):
            raise ValueError("SmoothPairwiseRankingLoss currently supports only reduction='mean'.")
        self.reduction = reduction

    def forward(self, input_left: Tensor, input_right: Tensor, target: Tensor) -> Tensor:
        diff = input_left - input_right
        loss = F.softplus(-target * diff)
        return loss.mean()


class TieHuberLoss(nn.Module):
    """
    Robust symmetric loss around 0 to encourage ties.

    For tie pairs:
        diff = s_left - s_right ≈ 0

    Huber-like penalty around 0:
        if |diff| <= delta:
            0.5 * diff^2 / delta
        else:
            |diff| - 0.5 * delta

    If delta <= 0, it falls back to pure L1: |diff|.
    """

    def __init__(self, delta: float, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean",):
            raise ValueError("TieHuberLoss currently supports only reduction='mean'.")
        self.delta = float(delta)
        self.reduction = reduction

    def forward(self, input_left: Tensor, input_right: Tensor) -> Tensor:
        diff = input_left - input_right
        abs_diff = diff.abs()

        if self.delta <= 0.0:
            loss = abs_diff
        else:
            mask = abs_diff <= self.delta
            loss = torch.empty_like(abs_diff)
            loss[mask] = 0.5 * (diff[mask] ** 2) / self.delta
            loss[~mask] = abs_diff[~mask] - 0.5 * self.delta

        return loss.mean()


class MarginRankingLossWithTies(nn.Module):
    """
    Tie loss used when label == 0.

    Enforces |s_left - s_right| <= margin via hinge penalty:
        loss = relu(|diff| - margin)

    This is conceptually consistent with the non-tie hinge ranking loss, but symmetric.
    """

    def __init__(self, margin: float, reduction: str = "mean"):
        super().__init__()
        self.margin = float(margin)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be one of {'none','mean','sum'}")
        self.reduction = reduction

    def forward(self, input_left: Tensor, input_right: Tensor) -> Tensor:
        penalty = F.relu((input_left - input_right).abs() - self.margin)

        if self.reduction == "none":
            return penalty
        if self.reduction == "sum":
            return penalty.sum()
        return penalty.mean()


# ====================================================================================== #
# Small utilities
# ====================================================================================== #

def _as_1d_scores(x: Tensor) -> Tensor:
    """Ensure model outputs are [B]. Accepts [B] or [B,1]."""
    if x.dim() == 2 and x.size(1) == 1:
        return x.view(-1)
    if x.dim() == 1:
        return x
    # Defensive: flatten only if it collapses to a single value per sample.
    x_flat = x.view(x.size(0), -1)
    if x_flat.size(1) != 1:
        raise ValueError(f"Expected one score per sample. Got shape {tuple(x.shape)}.")
    return x_flat.view(-1)


# ====================================================================================== #
# Ranking loss: split non-ties vs ties
# ====================================================================================== #

def compute_ranking_loss(
    network_output_dict: dict,
    labels: dict,
    criterion_ranking: nn.Module,
    ties: bool = False,
    criterion_ties: Optional[nn.Module] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute (non-tie ranking loss, tie loss) for a batch.

    Non-ties:
      - Uses `criterion_ranking(output_left, output_right, label)` where label ∈ {-1,+1}

    Ties:
      - Only computed if ties=True
      - Uses `criterion_ties(output_left, output_right)` for samples where label == 0

    Returns:
        (loss_nonties, loss_ties) as two scalar tensors on the correct device/dtype.
    """
    if ties and criterion_ties is None:
        raise ValueError("ties=True requires a criterion_ties instance.")

    # 1) Extract model outputs
    output_left_raw: Tensor = network_output_dict["left"]["output"]
    output_right_raw: Tensor = network_output_dict["right"]["output"]
    output_left = _as_1d_scores(output_left_raw)
    output_right = _as_1d_scores(output_right_raw)

    # 2) Prepare ranking labels (legacy sign flip)
    label: Tensor = -1 * labels["label_r"]

    batch_size = int(label.size(0))
    
    # 3) Numerical guards (lightweight debug)
    if torch.isnan(output_left).any() or torch.isnan(output_right).any():
        print("[DEBUG compute_ranking_loss] NaN in ranking outputs!", file=sys.stderr)
    if torch.isinf(output_left).any() or torch.isinf(output_right).any():
        print("[DEBUG compute_ranking_loss] Inf in ranking outputs!", file=sys.stderr)

    # 4) Split ties / non-ties
    mask_nontie = (label != 0)
    mask_tie = (label == 0)
    n_nonties = int(mask_nontie.sum().item())
    n_ties = int(mask_tie.sum().item())
    """
    if (n_nonties == 0) or (ties and n_ties == 0):
        print(
            f"[DEBUG compute_ranking_loss] batch_size={batch_size}, "
            f"n_nonties={n_nonties}, n_ties={n_ties}",
            file=sys.stderr,
        )
    """
    # 5) Non-ties loss
    if n_nonties > 0:
        loss_nonties = criterion_ranking(
            output_left[mask_nontie],
            output_right[mask_nontie],
            label[mask_nontie],
        )
    else:
        loss_nonties = torch.tensor(0.0, device=output_left.device, dtype=output_left.dtype)

    # 6) Ties loss
    if ties:
        if n_ties > 0:
            loss_ties = criterion_ties(
                output_left[mask_tie],
                output_right[mask_tie],
            )
        else:
            loss_ties = torch.tensor(0.0, device=output_left.device, dtype=output_left.dtype)
    else:
        loss_ties = torch.tensor(0.0, device=output_left.device, dtype=output_left.dtype)

    # 7) Final NaN check
    if torch.isnan(loss_nonties) or torch.isnan(loss_ties):
        print(
            "[DEBUG compute_ranking_loss] NaN loss_nonties / loss_ties detected! "
            f"batch_size={batch_size}, n_nonties={n_nonties}, n_ties={n_ties}",
            file=sys.stderr,
        )

    return loss_nonties, loss_ties


# ====================================================================================== #
# Classification loss
# ====================================================================================== #

def compute_loss_classification(
    network_output_dict: dict,
    labels: dict,
    criterion_classification: nn.Module,
) -> Tensor:
    """
    Compute classification loss using logits and class labels.

    Expects:
      - network_output_dict["logits"]["output"] : [B, C]
      - labels["label_c"] : [B]
    """
    logits: Tensor = network_output_dict["logits"]["output"]
    y: Tensor = labels["label_c"]
    return criterion_classification(logits, y.long())


# ====================================================================================== #
# Attention / gaze alignment (KL)
# ====================================================================================== #

def normalize_to_prob(x: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Normalize a non-negative map into a probability distribution per sample.

    Input:  x [B, H, W] or [B, N]
    Output: p [B, H*W] with sum(p_i) = 1 for each sample.
    """
    if x.dim() == 2:
        flat = x
    else:
        flat = x.view(x.size(0), -1)

    flat = flat.clamp(min=eps)
    return flat / flat.sum(dim=1, keepdim=True).clamp(min=eps)


def attention_kl_loss(
    attn_left: Tensor,
    attn_right: Tensor,
    gaze_left: Tensor,
    gaze_right: Tensor,
    has_mask: Optional[Tensor],
    eps: float = 1e-8,
) -> Tensor:
    """
    Symmetric KL divergence between predicted attention maps and gaze maps.

    All maps are normalized to per-sample probability distributions before KL.

    Args:
        attn_left/attn_right: predicted attention maps [B,H,W] (not necessarily normalized)
        gaze_left/gaze_right: gaze probability maps [B,H,W] 
        has_mask: optional [B] indicating which samples have gaze data (1) vs missing (0)

    Returns:
        scalar KL loss
    """
    p_left = normalize_to_prob(gaze_left, eps=eps)
    p_right = normalize_to_prob(gaze_right, eps=eps)
    q_left = normalize_to_prob(attn_left, eps=eps)
    q_right = normalize_to_prob(attn_right, eps=eps)

    kl_left = (p_left * (torch.log(p_left + eps) - torch.log(q_left + eps))).sum(dim=1)
    kl_right = (p_right * (torch.log(p_right + eps) - torch.log(q_right + eps))).sum(dim=1)
    kl = 0.5 * (kl_left + kl_right)  # [B]

    if has_mask is not None:
        has_mask_f = has_mask.float()
        denom = has_mask_f.sum().clamp(min=1.0)
        return (kl * has_mask_f).sum() / denom

    return kl.mean()


# ====================================================================================== #
# Orchestrator: compute full loss per model type
# ====================================================================================== #


# ---------------------------------------------------------------------
# Assumes these helpers already exist in the same module:
#   - MarginRankingLossWithTies
#   - compute_ranking_loss
#   - compute_loss_classification
#   - attention_kl_loss
# ---------------------------------------------------------------------


def compute_loss(
    args,
    network_output_dict: dict,
    labels: dict,
    return_parts: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
    """
    Compute the training loss for the configured model type.

    Supported args.model:
      - "rcnn"   : ranking-only
      - "sscnn"  : classification-only
      - "rsscnn" : ranking + classification + optional gaze KL

    Returns:
      - return_parts=False: total_loss
      - return_parts=True : (total_loss, parts_dict)
    """
    # =================================================================================
    # 0) Return helper
    # =================================================================================
    def _ret(total: Tensor, parts: Optional[Dict[str, Any]] = None):
        if not return_parts:
            return total
        return total, (parts or {})

    model = getattr(args, "model", None)

    # =================================================================================
    # 1) Criteria
    # =================================================================================
    criterion_ranking = nn.MarginRankingLoss(
        reduction="mean",
        margin=float(args.ranking_margin),
    )

    # ---- classification criterion (optional class weights + label smoothing) ----
    class_weight_tensor: Optional[Tensor] = None
    if getattr(args, "use_class_weights", False) and ("logits" in network_output_dict):
        logits = network_output_dict["logits"]["output"]
        class_weight_tensor = torch.tensor(
            getattr(args, "class_weights"),
            dtype=torch.float,
            device=logits.device,
        )

    smoothing = float(getattr(args, "label_smoothing", 0.0) or 0.0)
    criterion_classification = nn.CrossEntropyLoss(
        weight=class_weight_tensor,
        label_smoothing=(smoothing if smoothing > 0 else 0.0),
    )

    # ---- ties criterion (optional) ----
    ties_enabled = bool(getattr(args, "ties", False))
    criterion_ties: Optional[nn.Module] = None
    if ties_enabled:
        criterion_ties = MarginRankingLossWithTies(
            margin=float(args.ranking_margin_ties),
            reduction="mean",
        )

    # =================================================================================
    # 2) RCNN: ranking-only
    # =================================================================================
    if model == "rcnn":
        loss_nonties, loss_ties = compute_ranking_loss(
            network_output_dict=network_output_dict,
            labels=labels,
            criterion_ranking=criterion_ranking,
            ties=ties_enabled,
            criterion_ties=criterion_ties,
        )

        rank_w = float(getattr(args, "rank_w", 1.0))
        ties_w = float(getattr(args, "ties_w", 1.0))

        loss_rank_combo = (rank_w * loss_nonties) + (ties_w * loss_ties)
        total = loss_rank_combo

        parts = {
            "loss_rank_nonties": loss_nonties.detach(),
            "loss_rank_ties": loss_ties.detach(),
            "loss_rank_combo": loss_rank_combo.detach(),
        }
        return _ret(total, parts)

    # =================================================================================
    # 3) SSCNN: classification-only
    # =================================================================================
    if model == "sscnn":
        loss_class = compute_loss_classification(
            network_output_dict=network_output_dict,
            labels=labels,
            criterion_classification=criterion_classification,
        )

        parts = {"loss_class": loss_class.detach()}
        return _ret(loss_class, parts)

    # =================================================================================
    # 4) RSSCNN: ranking + classification + optional gaze KL
    # =================================================================================
    if model == "rsscnn":
        # -----------------------------------------------------------------
        # 4.1) Classification loss
        # -----------------------------------------------------------------
        loss_class = compute_loss_classification(
            network_output_dict=network_output_dict,
            labels=labels,
            criterion_classification=criterion_classification,
        )

        # -----------------------------------------------------------------
        # 4.2) Ranking loss (non-ties + optional ties)
        # -----------------------------------------------------------------
        loss_nonties, loss_ties = compute_ranking_loss(
            network_output_dict=network_output_dict,
            labels=labels,
            criterion_ranking=criterion_ranking,
            ties=ties_enabled,
            criterion_ties=criterion_ties,
        )

        rank_w = float(getattr(args, "rank_w", 1.0))
        ties_w = float(getattr(args, "ties_w", 1.0))
        loss_rank_combo = (rank_w * loss_nonties) + (ties_w * loss_ties)

        # -----------------------------------------------------------------
        # 4.3) Gaze KL (optional)
        # -----------------------------------------------------------------
        w_kl = float(getattr(args, "attn_w", 0.0) or 0.0)
        gaze_mode = str(getattr(args, "gaze_mode", getattr(args, "gaze", "off"))).lower().strip()
        if gaze_mode not in ("off", "align", "guide", "align+guide"):
            gaze_mode = "off"

        use_gaze_kl = (gaze_mode in ("align", "align+guide"))

        gaze_any = False
        gaze_count = 0

        if (not use_gaze_kl) or (w_kl <= 0.0):
            loss_kl = loss_class.new_zeros(())
            w_kl_eff = 0.0
        else:
            if "has_eye_mask" not in labels:
                raise KeyError("labels['has_eye_mask'] missing.")
            if ("gaze_l" not in labels) or ("gaze_r" not in labels):
                raise KeyError("labels['gaze_l'] / labels['gaze_r'] missing.")
            if ("attn_map" not in network_output_dict["left"]) or ("attn_map" not in network_output_dict["right"]):
                raise KeyError("network_output_dict['left/right']['attn_map'] missing.")

            attn_l = network_output_dict["left"]["attn_map"]
            attn_r = network_output_dict["right"]["attn_map"]
            if (attn_l is None) or (attn_r is None):
                raise ValueError("Attention maps are None while gaze KL is enabled.")

            has_eye_mask = labels["has_eye_mask"]  # BoolTensor [B]
            gaze_count = int(has_eye_mask.long().sum().item())
            gaze_any = bool(has_eye_mask.any().item())

            if not gaze_any:
                loss_kl = loss_class.new_zeros(())
                w_kl_eff = 0.0
            else:
                loss_kl = attention_kl_loss(
                    attn_l,
                    attn_r,
                    labels["gaze_l"],
                    labels["gaze_r"],
                    has_mask=has_eye_mask,
                )
                w_kl_eff = w_kl

        # -----------------------------------------------------------------
        # 4.4) Total loss
        # -----------------------------------------------------------------
        loss_kl_weighted = w_kl_eff * loss_kl
        total = loss_class + loss_rank_combo + loss_kl_weighted
        
        parts = {
            # raw (keep graph)
            "loss_class_raw": loss_class,
            "loss_rank_combo_raw": loss_rank_combo,
            "loss_kl_raw": loss_kl,
            "loss_kl_weighted_raw": loss_kl_weighted,
        
            # detached (safe for logging)
            "loss_class": loss_class.detach(),
            "loss_rank_combo": loss_rank_combo.detach(),
            "loss_kl": loss_kl.detach(),
            "loss_kl_weighted": loss_kl_weighted.detach(),
        
            "w_kl_eff": float(w_kl_eff),
            "gaze_any": float(gaze_any),
            "gaze_count": gaze_count,

            "gaze_mode": str(gaze_mode),
            "use_gaze_kl": float(use_gaze_kl),

        }
        return _ret(total, parts)

    # =================================================================================
    # 5) Unknown model
    # =================================================================================
    raise ValueError(f"Unknown model type: {model!r}")