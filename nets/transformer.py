"""
Transformer-based Siamese Network for Subjective Cycling Safety (pairwise comparisons).

This module provides a single model wrapper, `Transformer`, that:
  - Wraps a Vision Transformer–style backbone (timm / HF-like variants).
  - Produces ranking scores per image (RCNN head).
  - Optionally produces fused classification logits for a pair (SSCNN / RSSCNN heads).
  - Optionally extracts attention-derived spatial maps for gaze / saliency supervision.

Attention extraction supports CLI modes:
  - last    : CLS→patch attention from the last transformer block
  - rollout : attention rollout across blocks (identity-augmented)
  - topk    : last-block CLS→patch attention, sparsified to top-k tokens

The code is written to be robust across modern ViT-family backbones that expose
`forward_features()` and/or an internal block list at `backbone.blocks`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrDict = Union[torch.Tensor, Dict[str, torch.Tensor]]


class Transformer(nn.Module):
    """
    Siamese wrapper for transformer-style vision backbones.

    Output contract (matches existing training code):
      - model == "rcnn":
          {"left": {"output": score_l, "attn_map": attn_l},
           "right":{"output": score_r, "attn_map": attn_r}}
      - model == "sscnn":
          {"logits": {"output": logits}}
      - model == "rsscnn":
          {"left": {...}, "right": {...}, "logits": {...}}

    Notes:
      - The attention map is optional and computed only when `return_attn=True`.
      - Attention capture uses lightweight hooks that reconstruct attention from QK.
      - The `attn_grad` flag controls whether captured attention participates in gradients.
    """

    # ---------------------------------------------------------------------
    # Construction
    # ---------------------------------------------------------------------
    def __init__(
        self,
        backbone: nn.Module,
        model: str,
        pooling: str = "cls",
        pool_k: int = 10,
        num_classes: int = 2,
        finetune: bool = False,
        num_ft_blocks: int = 1,
        rank_dropout: float = 0.3,
        cross_dropout: float = 0.3,
        use_attn_hook: bool = False,
        return_attn: bool = True,
        attention_mode: str = "last",
        attn_topk: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.model = str(model).lower().strip()
        self.backbone = backbone

        # Pooling configuration
        self.pooling = str(pooling).lower().strip()
        self.pool_k = int(pool_k)

        # Attention configuration
        self.return_attn = bool(return_attn)
        self.attention_mode = str(attention_mode).lower().strip()
        self.attn_topk = attn_topk if attn_topk is None else int(attn_topk)

        # Training-time control for attention gradients
        self.attn_grad = False

        # Internal attention caches (populated only if hooks are enabled)
        self._attn_stack: List[Optional[torch.Tensor]] = []
        self._last_attn: Optional[torch.Tensor] = None

        # ------------------------------------------------------------------
        # Backbone trainability policy
        # ------------------------------------------------------------------
        self._freeze_backbone()
        if bool(finetune):
            self._unfreeze_last_blocks(num_ft_blocks=int(num_ft_blocks))

        # ------------------------------------------------------------------
        # Backbone output inspection (feature dim + prefix token count)
        # ------------------------------------------------------------------
        self.feat_dim, self.num_prefix_tokens = self._inspect_backbone_structure()

        # Pooling mode may change output feature dimensionality
        if self.pooling == "concat":
            self.feat_dim *= 2

        # ------------------------------------------------------------------
        # Heads (ranking and optional pairwise fusion classification)
        # ------------------------------------------------------------------
        self.feat_norm = nn.LayerNorm(self.feat_dim)
        self.pair_norm = nn.LayerNorm(self.feat_dim * 2)

        # Ranking head (per-image)
        self.rank_fc_1 = nn.Linear(self.feat_dim, 4096)
        self.rank_relu = nn.ReLU()
        self.rank_drop = nn.Dropout(float(rank_dropout))
        self.rank_fc_out = nn.Linear(4096, 1)

        # Fusion head (pairwise classification)
        self.cross_fc_1 = nn.Linear(self.feat_dim * 2, 512)
        self.cross_relu_1 = nn.ReLU()
        self.cross_drop_1 = nn.Dropout(float(cross_dropout))

        self.cross_fc_2 = nn.Linear(512, 512)
        self.cross_relu_2 = nn.ReLU()
        self.cross_drop_2 = nn.Dropout(float(cross_dropout))

        self.cross_fc_3 = nn.Linear(512, int(num_classes))

        # ------------------------------------------------------------------
        # Attention hook registration
        # ------------------------------------------------------------------
        if bool(use_attn_hook):
            self._register_attention_hooks()

    # ---------------------------------------------------------------------
    # Backbone trainability helpers
    # ---------------------------------------------------------------------
    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_last_blocks(self, num_ft_blocks: int) -> None:
        """
        Unfreeze the last N blocks of the backbone when the backbone exposes a
        block container. Also unfreezes final normalization layers when present.
        """
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            blocks = getattr(self.backbone, "stages", None)

        if blocks is not None and len(blocks) > 0:
            n = max(1, min(int(num_ft_blocks), len(blocks)))
            for blk in blocks[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True

        if hasattr(self.backbone, "norm"):
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        if hasattr(self.backbone, "head"):
            for p in self.backbone.head.parameters():
                p.requires_grad = True

    # ---------------------------------------------------------------------
    # Backbone output inspection
    # ---------------------------------------------------------------------
    def _inspect_backbone_structure(self) -> Tuple[int, int]:
        """
        Infer:
          - feature dimension (embedding dim for token features)
          - number of prefix tokens (CLS and register tokens)

        The inspection uses a single dummy forward pass at 224x224 to avoid
        relying on backbone-specific attributes (embed_dim, num_features, etc.).
        """
        device = next(self.backbone.parameters()).device
        dummy = torch.zeros(1, 3, 224, 224, device=device)

        with torch.no_grad():
            feats = self._forward_backbone(dummy)

        feats = self._unwrap_backbone_output(feats)

        # Token features: [B, T, C]
        if isinstance(feats, torch.Tensor) and feats.dim() == 3:
            _, T, C = feats.shape

            # For patch16 models at 224x224, expected patch count is 14*14=196.
            expected_patches = (224 // 16) ** 2
            prefix = max(0, T - expected_patches)
            return int(C), int(prefix)

        # Conv-like features: [B, C, H, W] or already pooled [B, C]
        if isinstance(feats, torch.Tensor) and feats.dim() == 4:
            return int(feats.shape[1]), 0
        if isinstance(feats, torch.Tensor) and feats.dim() == 2:
            return int(feats.shape[1]), 0

        raise ValueError(f"Unexpected backbone output type/shape: {type(feats)} / {getattr(feats, 'shape', None)}")

    # ---------------------------------------------------------------------
    # Backbone forward wrappers
    # ---------------------------------------------------------------------
    def _forward_backbone(self, x: torch.Tensor) -> TensorOrDict:
        """
        Forward into the backbone, preferring forward_features when available.
        """
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)
        return self.backbone(x)

    @staticmethod
    def _unwrap_backbone_output(out: TensorOrDict) -> torch.Tensor:
        """
        Unwrap common dict-like backbone outputs to a tensor.
        """
        if isinstance(out, dict):
            for key in ("x", "last_hidden_state", "feat", "features", "tokens", "tensor"):
                if key in out:
                    return out[key]
            return next(iter(out.values()))
        return out

    # ---------------------------------------------------------------------
    # Feature pooling
    # ---------------------------------------------------------------------
    def _extract_features(self, feats: TensorOrDict) -> torch.Tensor:
        """
        Convert raw backbone outputs into a single feature vector per image.

        Supported backbone outputs:
          - [B, T, C] token features (ViT-like)
          - [B, C, H, W] conv features
          - [B, C] already pooled
          - dict-like wrappers around the above
        """
        feats = self._unwrap_backbone_output(feats)

        # Conv-style feature map -> global average pool
        if feats.dim() == 4:
            return feats.mean(dim=(-2, -1))

        # Already pooled
        if feats.dim() == 2:
            return feats

        # Token features
        if feats.dim() != 3:
            raise ValueError(f"Unsupported feature shape for pooling: {feats.shape}")

        B, T, C = feats.shape
        num_prefix = int(self.num_prefix_tokens)

        if num_prefix > 0:
            cls_token = feats[:, 0]               # CLS always first
            patch_tokens = feats[:, num_prefix:]  # patches after all prefix tokens (CLS + regs)
        else:
            patch_tokens = feats
            cls_token = patch_tokens.mean(dim=1)

        if self.pooling == "cls":
            return cls_token

        if self.pooling == "mean":
            return patch_tokens.mean(dim=1)

        if self.pooling == "concat":
            return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)

        if self.pooling == "topk":
            k = min(int(self.pool_k), patch_tokens.size(1))
            norms = patch_tokens.norm(dim=-1)                 # [B, P]
            _, idx = norms.topk(k, dim=1)                     # [B, k]
            idx = idx.unsqueeze(-1).expand(-1, -1, C)          # [B, k, C]
            selected = torch.gather(patch_tokens, 1, idx)      # [B, k, C]
            return selected.mean(dim=1)

        raise ValueError(f"Unknown pooling mode: {self.pooling}")

    # ---------------------------------------------------------------------
    # Attention capture (hooks)
    # ---------------------------------------------------------------------
    def _register_attention_hooks(self) -> None:
        """
        Register attention capture hooks for ViT-like backbones.

        The hook reconstructs attention from QK for timm-style Attention modules
        that expose `.qkv`. The captured tensor is:
          - shape: [B, H, T, T]
          - value: softmax(QK^T * scale)

        When gradients are not required for attention-based losses, the attention
        tensor is detached for efficiency and memory stability.
        """
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None or not isinstance(blocks, (list, nn.ModuleList)) or len(blocks) == 0:
            return

        def wrap_attn_forward(attn_module: nn.Module) -> None:
            original_forward = attn_module.forward

            def forward_with_capture(x: torch.Tensor, *args, **kwargs):
                attn_tensor: Optional[torch.Tensor] = None

                try:
                    if x.dim() == 3 and hasattr(attn_module, "qkv"):
                        B, T, C = x.shape

                        if hasattr(attn_module, "num_heads"):
                            H = int(attn_module.num_heads)
                        elif hasattr(attn_module, "num_attention_heads"):
                            H = int(attn_module.num_attention_heads)
                        else:
                            H = max(1, C // 64)

                        head_dim = C // H

                        qkv = attn_module.qkv(x)
                        qkv = qkv.reshape(B, T, 3, H, head_dim).permute(2, 0, 3, 1, 4)
                        q, k = qkv[0], qkv[1]  # [B, H, T, Dh]

                        scale = head_dim ** -0.5
                        if hasattr(attn_module, "scale") and attn_module.scale is not None:
                            scale = float(attn_module.scale)

                        attn = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T]
                        attn = attn.softmax(dim=-1)

                        attn_tensor = attn if (self.training and bool(self.attn_grad)) else attn.detach()

                except Exception:
                    attn_tensor = None

                self._attn_stack.append(attn_tensor)
                self._last_attn = attn_tensor

                return original_forward(x, *args, **kwargs)

            attn_module.forward = forward_with_capture  # type: ignore[assignment]

        for blk in blocks:
            if hasattr(blk, "attn"):
                wrap_attn_forward(blk.attn)

    def _reset_attention_cache(self) -> None:
        self._attn_stack = []
        self._last_attn = None

    # ---------------------------------------------------------------------
    # Attention map utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _uniform_map(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return a uniform 14x14 map when attention is unavailable."""
        return torch.full((batch_size, 14, 14), 1.0 / 196.0, device=device, dtype=dtype)

    @staticmethod
    def _tokens_to_14x14(cls_to_patches: torch.Tensor) -> torch.Tensor:
        """
        Convert a CLS→patch vector into a normalized 14x14 map.

        The input is expected to be [B, P] where P is the number of patch tokens.
        If P is not a perfect square, the vector is treated as a 1xP strip and
        interpolated to 14x14.
        """
        B, P = cls_to_patches.shape
        grid = int(math.sqrt(P))

        if grid * grid == P:
            m = cls_to_patches.view(B, 1, grid, grid)
        else:
            m = cls_to_patches.view(B, 1, 1, P)

        m = F.interpolate(m, size=(14, 14), mode="bilinear", align_corners=False)
        flat = m.view(B, -1)
        flat = flat / flat.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return flat.view(B, 14, 14)

    def _cls_map_from_attention(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Build a CLS-based attention map from an attention tensor [B, H, T, T].
        """
        B = attn.size(0)
        attn_avg = attn.mean(dim=1)          # [B, T, T]
        cls_to_all = attn_avg[:, 0, :]       # [B, T]

        # Remove prefix tokens (CLS + registers). Patch tokens follow after prefix.
        p0 = int(self.num_prefix_tokens)
        cls_to_patches = cls_to_all[:, p0:]  # [B, P]

        if self.attention_mode == "topk" and self.attn_topk is not None:
            k = min(int(self.attn_topk), cls_to_patches.size(1))
            values, idx = cls_to_patches.topk(k=k, dim=1)
            mask = torch.zeros_like(cls_to_patches)
            mask.scatter_(1, idx, values)
            cls_to_patches = mask

        return self._tokens_to_14x14(cls_to_patches)

    def _rollout_map(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Compute attention rollout over the captured block attentions.

        Rollout definition:
          - For each block attention A (averaged over heads), use:
                A_hat = (A + I) / row_sum(A + I)
          - Multiply in order from early to late:
                R = A_hat_L @ ... @ A_hat_2 @ A_hat_1
          - Use CLS row of R and project to patch tokens.

        Captured attentions that are None are skipped. If no valid attentions are
        available, a uniform map is returned.
        """
        mats: List[torch.Tensor] = []
        for a in self._attn_stack:
            if a is None:
                continue
            mats.append(a.mean(dim=1))  # [B, T, T]

        if len(mats) == 0:
            # Batch size is not directly available here; infer from last cache if possible.
            last = self._last_attn
            if last is None:
                return None  # caller handles uniform fallback
            B = last.size(0)
            return self._uniform_map(B, device=device, dtype=dtype)

        B, T, _ = mats[0].shape
        I = torch.eye(T, device=mats[0].device, dtype=mats[0].dtype).unsqueeze(0).expand(B, -1, -1)

        # Identity-augmented, row-normalized attentions
        mats_hat = []
        for A in mats:
            A = A + I
            A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            mats_hat.append(A)

        # Rollout multiplication (early -> late)
        R = mats_hat[0]
        for A in mats_hat[1:]:
            R = A @ R

        cls_to_all = R[:, 0, :]            # [B, T]
        p0 = int(self.num_prefix_tokens)
        cls_to_patches = cls_to_all[:, p0:]  # [B, P]
        return self._tokens_to_14x14(cls_to_patches)

    def _get_attention_map(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Produce a 14x14 attention-derived map according to attention_mode.
        """
        if self.attention_mode not in ("last", "rollout", "topk"):
            return self._uniform_map(batch_size, device=device, dtype=dtype)

        if self.attention_mode == "rollout":
            m = self._rollout_map(device=device, dtype=dtype)
            if m is None:
                return self._uniform_map(batch_size, device=device, dtype=dtype)
            return m

        # last / topk modes operate on the last captured attention
        if self._last_attn is None:
            return self._uniform_map(batch_size, device=device, dtype=dtype)

        return self._cls_map_from_attention(self._last_attn)

    # ---------------------------------------------------------------------
    # Forward branches and fusion head
    # ---------------------------------------------------------------------
    def _forward_branch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward one side (left or right) through backbone + ranking head.
        """
        self._reset_attention_cache()

        feats = self._forward_backbone(x)
        pooled = self._extract_features(feats)
        pooled = self.feat_norm(pooled)

        hidden = self.rank_fc_1(pooled)
        hidden = self.rank_relu(hidden)
        hidden = self.rank_drop(hidden)
        score = self.rank_fc_out(hidden)

        attn_map = None
        if self.return_attn:
            attn_map = self._get_attention_map(
                batch_size=pooled.size(0),
                device=pooled.device,
                dtype=pooled.dtype,
            )

        return pooled, score, attn_map

    def _fusion_logits(self, feats_left: torch.Tensor, feats_right: torch.Tensor) -> torch.Tensor:
        """
        Fuse left/right feature vectors into classification logits.
        """
        pair = torch.cat([feats_left, feats_right], dim=-1)
        pair = self.pair_norm(pair)

        hidden = self.cross_fc_1(pair)
        hidden = self.cross_relu_1(hidden)
        hidden = self.cross_drop_1(hidden)

        hidden = self.cross_fc_2(hidden)
        hidden = self.cross_relu_2(hidden)
        hidden = self.cross_drop_2(hidden)

        return self.cross_fc_3(hidden)

    # ---------------------------------------------------------------------
    # Public forward
    # ---------------------------------------------------------------------
    def forward(self, left_batch: torch.Tensor, right_batch: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward two image batches through the Siamese wrapper.
        """
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
