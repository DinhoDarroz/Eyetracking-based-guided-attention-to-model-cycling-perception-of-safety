import warnings
import sys
from typing import Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn import _reduction as _Reduction
import torch.nn.functional as F



__all__ = ['MarginRankingLossWithTies']


class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]

class SmoothPairwiseRankingLoss(nn.Module):
    """
    Smooth pairwise ranking loss (RankNet-style, logistic).

    label semantics (after your sign flip):
        -1 → right should have higher score
        +1 → left should have higher score

    For each non-tie pair:
        diff = s_left - s_right
        y ∈ {-1, +1}
        loss = softplus(-y * diff) = log(1 + exp(-y * diff))

    - Smooth (no hinge)
    - Non-zero gradients even when diff is large but in the right direction
    - Good for noisy, subjective pairwise preferences
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction != "mean":
            raise ValueError("SmoothPairwiseRankingLoss currently supports only reduction='mean'")
        self.reduction = reduction

    def forward(self, input_left: Tensor, input_right: Tensor, target: Tensor) -> Tensor:
        # target: +1 (left wins), -1 (right wins)
        diff = input_left - input_right  # [B]
        # softplus(-y * diff) = log(1 + exp(-y * diff))
        loss = F.softplus(-target * diff)
        return loss.mean()
    

class TieHuberLoss(nn.Module):
    """
    Symmetric Huber loss around 0 for ties.

    For tie pairs:
        diff = s_left - s_right
        We want diff ≈ 0, but robustly (not overly punishing outliers).

    Loss:
        if |diff| <= delta:
            0.5 * diff^2 / delta
        else:
            |diff| - 0.5 * delta

    where delta = margin_ties (a small positive number).
    If margin_ties == 0, we fall back to pure L1: |diff|.
    """

    def __init__(self, margin: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.delta = margin
        if reduction != "mean":
            raise ValueError("TieHuberLoss currently supports only reduction='mean'")
        self.reduction = reduction

    def forward(self, input_left: Tensor, input_right: Tensor) -> Tensor:
        diff = input_left - input_right  # [B]
        abs_diff = torch.abs(diff)

        if self.delta <= 0.0:
            # Pure L1 if delta == 0
            loss = abs_diff
        else:
            # Huber around 0
            mask = abs_diff <= self.delta
            loss = torch.empty_like(abs_diff)
            # quadratic region
            loss[mask] = 0.5 * (diff[mask] ** 2) / self.delta
            # linear region
            loss[~mask] = abs_diff[~mask] - 0.5 * self.delta

        return loss.mean()

class MarginRankingLossWithTies(_Loss):
    r"""

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(*)`, same shape as the input.

    Examples::
        >>> loss = MarginRankingLossWithTies(margin=1)
        >>> input1 = torch.randn(3, requires_grad=True)
        >>> input2 = torch.randn(3, requires_grad=True)
        >>> target = torch.randn(3).sign()
        >>> output = loss(input1, input2, target)
        >>> output.backward()
    """
    __constants__ = ['margin', 'reduction']
    margin: float

    def __init__(self, margin: float = 0., size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.margin = margin

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        ties_loss_valid = torch.abs(input1 - input2) - self.margin
        zeros = torch.zeros_like(ties_loss_valid)
        loss = torch.max(ties_loss_valid, zeros)

        if self.reduction == 'mean':
            avg_loss = loss.mean()
        else:
            raise Exception("Reduction type not valid. Currently, only allows for 'mean'.")
        return avg_loss


def compute_ranking_loss(
    network_output_dict,
    labels,
    criterion_ranking,
    ties: bool = False,
    criterion_ties=None,
):
    """
    Compute ranking loss with optional tie loss.

    label semantics:
        -1 → left wins
         0 → tie
        +1 → right wins
    """

    # -------------------------------------------------------------------------
    # 0. Sanity checks
    # -------------------------------------------------------------------------
    if ties and criterion_ties is None:
        raise Exception('If including ties, criterion (loss) for ties must be included')

    # -------------------------------------------------------------------------
    # 1. Extract model outputs (left / right) and flatten
    # -------------------------------------------------------------------------
    output_left_raw = network_output_dict['left']['output']
    output_right_raw = network_output_dict['right']['output']

    # Make sure they are 1D [batch_size]
    output_left = output_left_raw.view(output_left_raw.size(0))
    output_right = output_right_raw.view(output_right_raw.size(0))

    # -------------------------------------------------------------------------
    # 2. Prepare labels: -1 (right), 0 (tie), +1 (left)
    # -------------------------------------------------------------------------
    
    label = -1 * labels['label_r']

    batch_size = label.size(0)

    # -------------------------------------------------------------------------
    # 3. Basic numerical debug checks
    # -------------------------------------------------------------------------
    if torch.isnan(output_left).any() or torch.isnan(output_right).any():
        print("[DEBUG ranking_loss] NaN in outputs!")
    if torch.isinf(output_left).any() or torch.isinf(output_right).any():
        print("[DEBUG ranking_loss] Inf in outputs!")

    # -------------------------------------------------------------------------
    # 4. Split into non-ties and ties
    # -------------------------------------------------------------------------
    index_mask_nontie = (label != 0)
    index_mask_tie = (label == 0)

    n_nonties = index_mask_nontie.sum().item()
    n_ties = index_mask_tie.sum().item()

    # This tells you when the batch is "missing" a label type
    if (n_nonties == 0) or (ties and n_ties == 0):
        print(
            f"[DEBUG compute_ranking_loss] batch_size={batch_size}, "
            f"n_nonties={n_nonties}, n_ties={n_ties}"
        )

    # -------------------------------------------------------------------------
    # 5. Non-ties loss
    #    - If the batch has no non-ties, this contributes 0.
    # -------------------------------------------------------------------------
    if n_nonties > 0:
        loss_nonties = criterion_ranking(
            output_left[index_mask_nontie],
            output_right[index_mask_nontie],
            label[index_mask_nontie],
        )
    else:
        loss_nonties = torch.tensor(
            0.0, device=output_left.device, dtype=output_left.dtype
        )

    # -------------------------------------------------------------------------
    # 6. Ties loss (optional)
    #    - Only computed if `ties=True`.
    #    - If the batch has no ties, this contributes 0.
    # -------------------------------------------------------------------------
    if ties:
        if n_ties > 0:
            loss_ties = criterion_ties(
                output_left[index_mask_tie],
                output_right[index_mask_tie],
            )
        else:
            loss_ties = torch.tensor(
                0.0, device=output_left.device, dtype=output_left.dtype
            )
    else:
        loss_ties = torch.tensor(
            0.0, device=output_left.device, dtype=output_left.dtype
        )


    # -------------------------------------------------------------------------
    # 7. Final NaN check
    # -------------------------------------------------------------------------

    if torch.isnan(loss_nonties) or torch.isnan(loss_ties):
        print(
            "[DEBUG compute_ranking_loss] NaN loss_nonties / loss_ties detected! "
            f"batch_size={batch_size}, n_nonties={n_nonties}, n_ties={n_ties}"
        )

    return loss_nonties, loss_ties


def compute_loss_classification(network_output_dict, labels, criterion_classification):
    """
    Computes the classification loss between network outputs and labels using a loss criterion

        Parameters:
             network_output_dict (dict): Network output
             labels (nn.Tensor): Ground truth class labels
             criterion_classification (nn.Loss): Loss function (criterion)

        Returns:
            loss_class (nn.Tensor): Loss values
    """

    # Forward pass data output
    logits = network_output_dict['logits']['output']

    # Ground truth label data
    label = labels['label_c']

    # Classification loss
    loss_class = criterion_classification(logits, label.long())
    return loss_class


def normalize_to_prob(x, eps=1e-8):
    # x: [B,14,14] -> [B,196], each sums to 1
    B = x.shape[0]
    flat = x.reshape(B, -1)
    flat = flat.clamp(min=eps)
    flat = flat / flat.sum(dim=1, keepdim=True).clamp(min=eps)
    return flat

def attention_kl_loss(attn_left, attn_right, gaze_left, gaze_right, has_mask):
    # attn_*: [B,14,14] (unnormalized); gaze_*: [B,14,14] (already sums to 1 but we re-normalize safely)
    p_left  = normalize_to_prob(gaze_left)
    p_right = normalize_to_prob(gaze_right)
    q_left  = normalize_to_prob(attn_left)
    q_right = normalize_to_prob(attn_right)

    eps = 1e-8
    kl_left  = (p_left  * (torch.log(p_left  + eps) - torch.log(q_left  + eps))).sum(dim=1)
    kl_right = (p_right * (torch.log(p_right + eps) - torch.log(q_right + eps))).sum(dim=1)
    kl = 0.5 * (kl_left + kl_right)

    if has_mask is not None:
        has_mask = has_mask.float()
        denom = has_mask.sum().clamp(min=1.0)
        kl = (kl * has_mask).sum() / denom
    else:
        kl = kl.mean()
    return kl
    
def compute_loss(args, network_output_dict, labels):
    
    # ============================================================
    # 1) non-tie ranking loss (hinge / margin ranking)
    #    Used for pairs where label_r ∈ {+1, -1}
    # ============================================================
    criterion_ranking = nn.MarginRankingLoss(
        reduction='mean',
        margin=args.ranking_margin,
    )
    
    
    # ============================================================
    # 2) classification loss setup
    #    - optional class weighting
    #    - optional label smoothing
    # ============================================================
    
    # ---- 2.1 class weights (optional) ----
    class_weight_tensor = None
    
    # only compute class weights if:
    #   (a) user enabled class_weights, and
    #   (b) logits exist (i.e., model is sscnn or rsscnn)
    if getattr(args, "use_class_weights", False) and ("logits" in network_output_dict):
        logits = network_output_dict["logits"]["output"]
        device = logits.device
    
        # convert Python list → tensor on correct device
        class_weight_tensor = torch.tensor(
            args.class_weights,
            dtype=torch.float,
            device=device,
        )
    
    # ---- 2.2 label smoothing (optional) ----
    smoothing = getattr(args, "label_smoothing", 0.0)
    
    # final classification criterion
    criterion_classification = nn.CrossEntropyLoss(
        weight=class_weight_tensor,                       # None or tensor
        label_smoothing=(smoothing if smoothing > 0 else 0.0),
    )
    
    
    # ============================================================
    # 3) tie ranking loss (only used if ties=True)
    #    MarginRankingLossWithTies enforces |diff| < margin
    # ============================================================
    if args.ties:
        criterion_ties = MarginRankingLossWithTies(
            reduction='mean',
            margin=args.ranking_margin_ties,
        )
    else:
        criterion_ties = None

    # ======================================================================
    # MODEL: ranking-only (rcnn)  --- USE SMOOTH LOSS
    # ======================================================================
    if args.model == 'rcnn':
        # compute non-ties + ties losses exactly like in rsscnn
        loss_nonties, loss_ties = compute_ranking_loss(
            network_output_dict,
            labels,
            criterion_ranking,
            ties=args.ties,
            criterion_ties=criterion_ties,
        )
    
        # combine them with λ_R and λ_tie
        loss_rank_combo = args.rank_w * loss_nonties + args.ties_w * loss_ties
    
        return loss_rank_combo


    # ======================================================================
    # MODEL: classification-only (sscnn)
    # ======================================================================
    elif args.model == 'sscnn':
        return compute_loss_classification(
            network_output_dict,
            labels,
            criterion_classification,
        )

    # ======================================================================
    # MODEL: classification + ranking (rsscnn)
    # ======================================================================
    elif args.model == 'rsscnn':
        # weights: keep classification at 1.0, separate λ_R and λ_tie
        w_class   = 1.0
        lambda_R  = args.rank_w
        lambda_tie = args.ties_w
        w_kl      = args.attn_w

        # classification
        loss_class = compute_loss_classification(
            network_output_dict,
            labels,
            criterion_classification,
        )

        # ranking (split into non-ties and ties)
        loss_nonties, loss_ties = compute_ranking_loss(
            network_output_dict,
            labels,
            criterion_ranking,
            ties=args.ties,
            criterion_ties=criterion_ties,
        )

        # combine ranking terms like in the paper:
        #   L_R = λ_R * L_R̂(non-ties) + λ_1 * L_1(ties)
        loss_rank_combo = lambda_R * loss_nonties + lambda_tie * loss_ties

        # gaze KL (disabled when gaze='off' or attn_w == 0)
        if getattr(args, 'gaze', 'use') == 'off' or w_kl == 0:
            loss_kl = loss_rank_combo * 0.0  # same dtype/device, no-op
            w_kl = 0.0
        else:
            loss_kl = attention_kl_loss(
                network_output_dict['left']['attn_map'],
                network_output_dict['right']['attn_map'],
                labels['gaze_l'],
                labels['gaze_r'],
                has_mask=labels['has_eye_mask'],
            )

        return w_class * loss_class + loss_rank_combo + w_kl * loss_kl

    else:
        raise ValueError(f"Unknown model type: {args.model}")


if __name__ == '__main__':

    torch.manual_seed(8)

    loss = MarginRankingLossWithTies(margin=1)
    input1 = torch.randn(3, requires_grad=True)
    print(input1)
    input2 = torch.randn(3, requires_grad=True)
    print(input2)
    print(input1 - input2)
    output = loss(input1, input2)
    output.backward()

    print(output)
