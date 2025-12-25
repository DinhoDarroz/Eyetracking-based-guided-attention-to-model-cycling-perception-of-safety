"""Transformer-based pairwise model for ranking and classification.

This module wraps a ViT/DeiT-style backbone with lightweight heads that
support three training modes:

- ``rcnn``   : ranking loss only
- ``sscnn``  : classification loss only
- ``rsscnn`` : ranking + classification (+ optional attention KL)

The implementation keeps responsibilities narrow:
    * Backbone preparation (freezing / partial unfreezing)
    * Head construction (ranking + cross-branch classification)
    * Optional extraction of attention maps for gaze alignment

The forward API matches the expectations of :mod:`scripts.train_script` and
``utils.losses``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _to_cls_feats(feats: torch.Tensor) -> torch.Tensor:
    """Normalize backbone outputs to CLS embeddings.

    Accepts either ``[B, C]`` or ``[B, T, C]`` tensors (CLS at index 0). Some
    backbones return dictionaries; in that case common keys are inspected.

    Returns
    -------
    torch.Tensor
        The CLS embedding with shape ``[B, C]``.
    """

    if isinstance(feats, dict):
        for key in ("x", "last_hidden_state", "feat", "features", "tokens"):
            if key in feats:
                feats = feats[key]
                break

    if feats.dim() == 2:
        return feats
    if feats.dim() == 3:
        return feats[:, 0]

    raise ValueError(f"Unexpected features shape: {feats.shape}")


def _count_trainable(parameters) -> Tuple[int, int]:
    """Return a tuple with parameter counts.

    Returns
    -------
    (int, int)
        ``(total_params, trainable_params)``, useful for logging model size.
    """

    total = sum(p.numel() for p in parameters)
    trainable = sum(p.numel() for p in parameters if p.requires_grad)
    return total, trainable


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------


class Transformer(nn.Module):
    """Pairwise transformer wrapper.

    Args:
        backbone: ViT/DeiT-style backbone instance (already created).
        model: One of ``{"rcnn", "sscnn", "rsscnn"}``.
        num_classes: Number of classification classes (2 when ties disabled,
            3 when ties enabled).
        finetune: Whether to train backbone parameters.
        num_ft_blocks: If finetuning, number of *last* transformer blocks to
            unfreeze (LayerNorm is also unfrozen).
        rank_dropout: Dropout probability inside the ranking head.
        cross_dropout: Dropout probability inside the classification head.
        use_attn_hook: If True, registers a hook on the last transformer block
            attention to extract CLS->patch maps.
        return_attn: If False, attention maps are not computed/returned.
        attention_mode: Strategy for converting attention to a spatial map.
            - ``last``   : CLS->patch attention from the last block only.
            - ``rollout``: Rollout across all blocks (augmented with identity).
            - ``topk``   : Last-block attention masked to top-k tokens.
        topk: When ``attention_mode='topk'``, keep only the largest ``k`` CLS->patch
            weights (per sample) before normalizing. Ignored otherwise.
    """

    def __init__(
        self,
        backbone: nn.Module,
        model: str,
        num_classes: int = 2,
        finetune: bool = False,
        num_ft_blocks: int = 1,
        rank_dropout: float = 0.3,
        cross_dropout: float = 0.3,
        use_attn_hook: bool = False,
        return_attn: bool = True,
        attention_mode: str = "last",
        topk: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.model = model
        self.backbone = backbone
        self.transformer = backbone  # public alias used elsewhere

        self.rank_dropout = rank_dropout
        self.cross_dropout = cross_dropout
        self.return_attn = return_attn
        self.attention_mode = attention_mode
        self.topk = topk
        self.attn_grad = False  # toggled in train.py when gaze KL is active
        self._last_attn: Optional[torch.Tensor] = None
        self._attn_stack: List[torch.Tensor] = []

        # 1) Backbone setup
        self._freeze_backbone()
        if finetune:
            self._unfreeze_last_blocks(num_ft_blocks)

        # 2) Feature dimension discovery
        self.feat_dim = self._infer_feature_dim()

        # 3) Heads
        self.feat_norm = nn.LayerNorm(self.feat_dim)
        self.pair_norm = nn.LayerNorm(self.feat_dim * 2)

        # ------------------------------------------------------------------
        # Ranking head:
        #   feat_dim -> 4096 -> 1
        # ------------------------------------------------------------------
        self.rank_fc_1 = nn.Linear(self.feat_dim, 4096)
        self.rank_relu = nn.ReLU()
        self.rank_drop = nn.Dropout(self.rank_dropout)
        self.rank_fc_out = nn.Linear(4096, 1)
        
        # ------------------------------------------------------------------
        # Cross-branch classification head:
        #   [feat_L || feat_R] -> 512 -> 512 -> num_classes
        # ------------------------------------------------------------------
        self.cross_fc_1 = nn.Linear(self.feat_dim * 2, 512)
        self.cross_relu_1 = nn.ReLU()
        self.cross_drop_1 = nn.Dropout(self.cross_dropout)

        self.cross_fc_2 = nn.Linear(512, 512)
        self.cross_relu_2 = nn.ReLU()
        self.cross_drop_2 = nn.Dropout(self.cross_dropout)

        self.cross_fc_3 = nn.Linear(512, num_classes)

        # 4) Optional attention hook (only if caller cares about gaze loss)
        if use_attn_hook:
            self._register_attn_capture()

        total, trainable = _count_trainable(self.parameters())
        print(f"[Transformer] params total={total:,} trainable={trainable:,}")

    # ------------------------------------------------------------------
    # Backbone management
    # ------------------------------------------------------------------
    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters (feature extractor mode)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_last_blocks(self, num_ft_blocks: int) -> None:
        """Unfreeze the last ``num_ft_blocks`` transformer blocks and final norm."""

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            total_blocks = len(blocks)
            n_unfreeze = max(1, min(num_ft_blocks, total_blocks))
            for block in blocks[-n_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True

        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def _infer_feature_dim(self) -> int:
        if hasattr(self.backbone, "num_features"):
            return int(self.backbone.num_features)
        if hasattr(self.backbone, "embed_dim"):
            return int(self.backbone.embed_dim)
        if hasattr(self.backbone, "head") and hasattr(self.backbone.head, "in_features"):
            return int(self.backbone.head.in_features)
        raise AttributeError(
            "Cannot infer feature dimension from backbone. "
            "Expected `num_features`, `embed_dim`, or `head.in_features`."
        )

    # ------------------------------------------------------------------
    # Attention capture
    # ------------------------------------------------------------------
    def _register_attn_capture(self) -> None:
        """Attach hooks to every transformer block to capture raw attention."""
        vt = self.backbone
        if not hasattr(vt, "blocks") or len(vt.blocks) == 0:
            return

        def hook_block(attn_module):
            original_forward = attn_module.forward

            def forward_with_capture(x, *args, **kwargs):
                try:
                    batch, tokens, dim = x.shape
                    qkv = attn_module.qkv(x)  # (B, T, 3*D)
                    qkv = qkv.reshape(batch, tokens, 3, attn_module.num_heads, dim // attn_module.num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
                    q, k, _v = qkv[0], qkv[1], qkv[2]

                    # Raw attention weights [B, H, T, T]
                    attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                    attn = attn.softmax(dim=-1)  # [B, H, T, T]

                    attn_to_store = attn if (self.training and self.attn_grad) else attn.detach()
                    self._attn_stack.append(attn_to_store)
                    self._last_attn = attn_to_store  # convenience for legacy usage
                except Exception:
                    self._attn_stack.append(None)  # placeholder to keep depth aligned
                    self._last_attn = None

                return original_forward(x, *args, **kwargs)

            attn_module.forward = forward_with_capture

        for block in vt.blocks:
            attn_module = getattr(block, "attn", None)
            if attn_module is not None:
                hook_block(attn_module)

    def _reset_attention_cache(self) -> None:
        """Clear cached attentions before each forward branch."""
        self._attn_stack = []
        self._last_attn = None

    def _rollout_attention(self, batch_size: int, device, dtype) -> torch.Tensor:
        """Compute rollout over stacked attentions. Falls back to uniform."""

        if not self._attn_stack:
            return torch.full((batch_size, 14, 14), 1.0 / (14 * 14), device=device, dtype=dtype)

        # Keep only valid tensors, discard failed captures
        attns: List[torch.Tensor] = [a for a in self._attn_stack if isinstance(a, torch.Tensor)]
        if not attns:
            return torch.full((batch_size, 14, 14), 1.0 / (14 * 14), device=device, dtype=dtype)

        result = torch.eye(attns[0].size(-1), device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        for attn in attns:
            attn_mean = attn.mean(dim=1)  # [B, T, T]
            attn_aug = attn_mean + torch.eye(attn_mean.size(-1), device=device, dtype=dtype)
            attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            result = result @ attn_aug

        cls_to_patches = result[:, 0, 1:]
        return self._tokens_to_map(cls_to_patches, batch_size, device, dtype)

    def _tokens_to_map(self, cls_to_patches: torch.Tensor, batch_size: int, device, dtype) -> torch.Tensor:
        """Convert CLS-to-patch weights to a normalized 14x14 map."""
        num_patches = cls_to_patches.size(1)
        grid = int(math.sqrt(num_patches))
        if grid * grid == num_patches:
            attn_map = cls_to_patches.view(batch_size, 1, grid, grid)
        else:
            attn_map = cls_to_patches.view(batch_size, 1, 1, num_patches)
        attn_map = F.interpolate(attn_map, size=(14, 14), mode="bilinear", align_corners=False)

        # Normalize to sum to 1 per sample (probability-like map)
        flat = attn_map.view(batch_size, -1)
        flat = flat / flat.sum(dim=1, keepdim=True).clamp(min=1e-6)
        attn_map = flat.view(batch_size, 1, 14, 14)
        return attn_map.squeeze(1)

    def _cls_attention_map(self, batch_size: int, device, dtype) -> torch.Tensor:
        """Return a [B,14,14] attention map (uniform fallback).

        Respects ``attention_mode``:
        - ``rollout`` uses all captured layers.
        - ``last`` uses the most recent layer.
        - ``topk`` sparsifies the last layer before mapping.
        """

        if self.attention_mode == "rollout":
            return self._rollout_attention(batch_size, device, dtype)

        if self._last_attn is None or self._last_attn.dim() != 4:
            return torch.full((batch_size, 14, 14), 1.0 / (14 * 14), device=device, dtype=dtype)

        attn = self._last_attn.mean(dim=1)  # [B, T, T]
        cls_to_all = attn[:, 0]             # [B, T]
        cls_to_patches = cls_to_all[:, 1:]  # drop CLS->CLS

        if self.attention_mode == "topk" and self.topk and self.topk > 0:
            # Mask all but top-k tokens per sample
            values, indices = cls_to_patches.topk(k=min(self.topk, cls_to_patches.size(1)), dim=1)
            mask = torch.zeros_like(cls_to_patches)
            mask.scatter_(1, indices, values)
            cls_to_patches = mask

        return self._tokens_to_map(cls_to_patches, batch_size, device, dtype)

    # ------------------------------------------------------------------
    # Branch + fusion
    # ------------------------------------------------------------------
    def _forward_branch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for a single branch.

        Returns
        -------
        (torch.Tensor, torch.Tensor, Optional[torch.Tensor])
            Tuple of (CLS features, ranking score [B,1], optional attention map [B,14,14]).
        """
        self._reset_attention_cache()

        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone(x)

        cls = _to_cls_feats(feats)
        cls = self.feat_norm(cls)

        hidden = self.rank_fc_1(cls)
        hidden = self.rank_relu(hidden)
        hidden = self.rank_drop(hidden)
        score = self.rank_fc_out(hidden)  # [B,1]

        attn_map = None
        if self.return_attn:
            attn_map = self._cls_attention_map(batch_size=cls.size(0), device=cls.device, dtype=cls.dtype)

        return cls, score, attn_map

    def _fusion_logits(self, feats_left: torch.Tensor, feats_right: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([feats_left, feats_right], dim=-1)
        pair = self.pair_norm(pair)

        hidden = self.cross_fc_1(pair)
        hidden = self.cross_relu_1(hidden)
        hidden = self.cross_drop_1(hidden)

        hidden = self.cross_fc_2(hidden)
        hidden = self.cross_relu_2(hidden)
        hidden = self.cross_drop_2(hidden)

        return self.cross_fc_3(hidden)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def partial_eval(self) -> None:
        """Placeholder to mirror CNN API; currently a no-op."""

    def forward(self, left_batch: torch.Tensor, right_batch: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        left_feats, left_score, left_attn = self._forward_branch(left_batch)
        right_feats, right_score, right_attn = self._forward_branch(right_batch)

        if self.model == "rcnn":
            return {
                "left": {"output": left_score, "attn_map": left_attn},
                "right": {"output": right_score, "attn_map": right_attn},
            }

        if self.model == "sscnn":
            logits = self._fusion_logits(left_feats, right_feats)
            return {"logits": {"output": logits}}

        if self.model == "rsscnn":
            logits = self._fusion_logits(left_feats, right_feats)
            return {
                "left": {"output": left_score, "attn_map": left_attn},
                "right": {"output": right_score, "attn_map": right_attn},
                "logits": {"output": logits},
            }

        raise ValueError(f"Invalid model type: {self.model}")


if __name__ == "__main__":
    # Lightweight smoke test (uses timm if available)
    try:
        import timm

        backbone = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=False,
            num_classes=0,
        )

        net = Transformer(
            backbone=backbone,
            model="rsscnn",
            num_classes=3,
            finetune=False,
            return_attn=True,
            use_attn_hook=False,
        )

        x_l = torch.randn(2, 3, 224, 224)
        x_r = torch.randn(2, 3, 224, 224)
        out = net(x_l, x_r)
        print("Forward keys:", out.keys())
        if "logits" in out:
            print(" logits:", out["logits"]["output"].shape)
    except Exception as exc:
        print(f"Smoke test skipped: {exc}")
