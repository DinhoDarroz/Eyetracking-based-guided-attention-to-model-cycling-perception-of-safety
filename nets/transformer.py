# transformer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .transformer_utils import (
    AttentionConfig,
    AttentionRecorder,
    GuideGuidance,
    GuideGuidanceConfig,
    forward_backbone_tokens,
    infer_embed_dim,
    infer_num_prefix_tokens,
    pool_tokens,
    uniform_attention_map,
)


# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TransformerConfig:
    """
    Model wrapper configuration.

    model:
      - "rcnn"  : ranking-only (no pairwise class head)
      - "sscnn" : ranking + pairwise classification head
      - "rsscnn": ranking + pairwise classification head + (optional) attention map output

    pooling:
      Token pooling strategy used to produce a single vector per branch.
    """
    model: str = "rsscnn"
    pooling: str = "cls"
    pool_k: int = 10
    num_classes: int = 2

    finetune: bool = False
    num_ft_blocks: int = 1

    rank_dropout: float = 0.3
    cross_dropout: float = 0.3

    force_num_prefix_tokens: Optional[int] = None
    apply_token_norm: bool = False

    attention: AttentionConfig = AttentionConfig(
        enabled=False,
        return_attn=True,
        mode="last",
        topk=None,
        out_hw=(14, 14),
    )

    guidance: GuideGuidanceConfig = GuideGuidanceConfig(enabled=False)


# -------------------------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    Siamese wrapper around a transformer backbone.

    Outputs:
      {
        "left":  {"output": score_l, "attn_map": attn_l},
        "right": {"output": score_r, "attn_map": attn_r},
        "logits":{"output": logits}
      }

    Gaze support:
      - Guidance injection is active when guidance config is enabled and gaze maps are provided.
      - Attention maps are produced when attention recorder is enabled (typically for KL alignment loss).
    """

    def __init__(
        self,
        backbone: nn.Module,
        model: str = "rsscnn",
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
        force_num_prefix_tokens: Optional[int] = None,
        apply_token_norm: bool = False,
        attn_out_hw: Optional[Tuple[int, int]] = None,
        use_gaze_injection: bool = False,
        guidance_cfg: Optional[GuideGuidanceConfig] = None,
    ) -> None:
        super().__init__()

        # ----------------------------------------------------------------------------------
        # Step 1) Store backbone references 
        # ----------------------------------------------------------------------------------
        self.backbone = backbone
        self.transformer = backbone

        # ----------------------------------------------------------------------------------
        # Step 2) Build and validate config
        # ----------------------------------------------------------------------------------
        cfg = TransformerConfig(
            model=str(model).lower().strip(),
            pooling=str(pooling).lower().strip(),
            pool_k=int(pool_k),
            num_classes=int(num_classes),
            finetune=bool(finetune) and (int(num_ft_blocks) > 0),
            num_ft_blocks=int(max(0, num_ft_blocks)),
            rank_dropout=float(rank_dropout),
            cross_dropout=float(cross_dropout),
            force_num_prefix_tokens=force_num_prefix_tokens,
            apply_token_norm=bool(apply_token_norm),
            attention=AttentionConfig(
                enabled=bool(use_attn_hook),
                return_attn=bool(return_attn),
                mode=str(attention_mode).lower().strip(),
                topk=None if attn_topk is None else int(attn_topk),
                out_hw=tuple(attn_out_hw) if attn_out_hw is not None else (14, 14),
            ),
            guidance=(
                guidance_cfg
                if guidance_cfg is not None
                else GuideGuidanceConfig(enabled=bool(use_gaze_injection))
            ),
        )
        self.cfg = cfg

        if self.cfg.model not in ("rcnn", "sscnn", "rsscnn"):
            raise ValueError(f"Unknown model='{self.cfg.model}'. Expected: rcnn/sscnn/rsscnn.")

        allowed_poolings = {
            "cls",
            "mean",
            "patch_mean",
            "reg_mean",
            "prefix_mean",
            "max",
            "cls_max_concat",
            "cls_reg_concat",
            "cls_reg_add",
            "concat",
            "topk",
        }
        if self.cfg.pooling not in allowed_poolings:
            raise ValueError(f"Unknown pooling='{self.cfg.pooling}'. Expected one of: {sorted(allowed_poolings)}.")

        if self.cfg.attention.mode not in ("last", "rollout", "topk"):
            raise ValueError(f"Unknown attention_mode='{self.cfg.attention.mode}'. Expected: last/rollout/topk.")

        # ----------------------------------------------------------------------------------
        # Step 3) Gaze/attention runtime flags (controlled externally by training loop)
        # ----------------------------------------------------------------------------------
        self.gaze_backprop_enabled = True   # toggled by train_script _set_gaze_bp(...)
        self.gaze_requires_grad = False     # computed per forward (depends on finetune + training)

        # ----------------------------------------------------------------------------------
        # Step 4) Freeze/finetune backbone parameters
        # ----------------------------------------------------------------------------------
        self._freeze_backbone()
        if self.cfg.finetune:
            self._unfreeze_last_blocks(self.cfg.num_ft_blocks)

        # ----------------------------------------------------------------------------------
        # Step 5) Infer backbone token dimensions
        # ----------------------------------------------------------------------------------
        self.embed_dim = infer_embed_dim(self.backbone)
        self.num_prefix_tokens = infer_num_prefix_tokens(
            self.backbone,
            force=self.cfg.force_num_prefix_tokens,
        )

        # Optional token normalization (applied before pooling)
        self.apply_token_norm = bool(self.cfg.apply_token_norm)
        self.token_norm: Optional[nn.Module] = self.backbone.norm if hasattr(self.backbone, "norm") else None

        # ----------------------------------------------------------------------------------
        # Step 6) Feature dimensionality after pooling
        # ----------------------------------------------------------------------------------
        two_d_poolings = {"concat", "cls_reg_concat", "cls_max_concat"}
        self.feat_dim = (int(self.embed_dim) * 2) if (self.cfg.pooling in two_d_poolings) else int(self.embed_dim)

        # Normalization for pooled features and paired features
        self.feat_norm = nn.LayerNorm(self.feat_dim) if (self.cfg.pooling in two_d_poolings) else nn.Identity()
        self.pair_norm = nn.LayerNorm(self.feat_dim * 2)

        # ----------------------------------------------------------------------------------
        # Step 7) Heads 
        # ----------------------------------------------------------------------------------
        # Rank head: 2 hidden layers + dropout + scalar output
        self.rank_fc_1 = nn.Linear(self.feat_dim, 384)
        self.rank_fc_2 = nn.Linear(384, 162)
        self.rank_relu = nn.ReLU()
        self.rank_drop = nn.Dropout(float(self.cfg.rank_dropout))
        self.rank_fc_out = nn.Linear(162, 1)

        # Pairwise class head: 3 hidden layers + dropout + class logits
        self.cross_fc_1 = nn.Linear(self.feat_dim * 2, 512)
        self.cross_relu_1 = nn.ReLU()
        self.cross_drop_1 = nn.Dropout(float(self.cfg.cross_dropout))

        self.cross_fc_2 = nn.Linear(512, 512)
        self.cross_relu_2 = nn.ReLU()
        self.cross_drop_2 = nn.Dropout(float(self.cfg.cross_dropout))

        self.cross_fc_3 = nn.Linear(512, 256)
        self.cross_relu_3 = nn.ReLU()
        self.cross_drop_3 = nn.Dropout(float(self.cfg.cross_dropout))

        self.cross_fc_out = nn.Linear(256, int(self.cfg.num_classes))

        # ----------------------------------------------------------------------------------
        # Step 8) Attention recorder (align path)
        # ----------------------------------------------------------------------------------
        self.attn_cfg = self.cfg.attention
        self.attn_recorder: Optional[AttentionRecorder] = (
            AttentionRecorder(self.attn_cfg) if self.attn_cfg.enabled else None
        )
        if self.attn_recorder is not None:
            self.attn_recorder.attach(self.backbone)

        # ----------------------------------------------------------------------------------
        # Step 9) Guidance module (guide path)
        # ----------------------------------------------------------------------------------
        self.guidance_cfg = self.cfg.guidance
        self.guidance: Optional[GuideGuidance] = (
            GuideGuidance(token_dim=int(self.embed_dim), cfg=self.guidance_cfg)
            if self.guidance_cfg.enabled
            else None
        )

    # -------------------------------------------------------------------------------------------------
    # Backbone finetune management
    # -------------------------------------------------------------------------------------------------

    def _freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_last_blocks(self, num_ft_blocks: int) -> None:
        n_req = int(num_ft_blocks)
        if n_req <= 0:
            return

        def _unfreeze_param_attr(module: nn.Module, attr_name: str) -> None:
            if not hasattr(module, attr_name):
                return
            obj = getattr(module, attr_name)
            if isinstance(obj, torch.nn.Parameter):
                obj.requires_grad = True

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            blocks = getattr(self.backbone, "stages", None)

        if blocks is not None:
            n_total = len(blocks)
            n = max(0, min(n_req, n_total))
            for blk in list(blocks)[-n:]:
                for p in blk.parameters():
                    p.requires_grad = True

        norm = getattr(self.backbone, "norm", None)
        if isinstance(norm, nn.Module):
            for p in norm.parameters():
                p.requires_grad = True

        patch_embed = getattr(self.backbone, "patch_embed", None)
        if isinstance(patch_embed, nn.Module):
            for p in patch_embed.parameters():
                p.requires_grad = True

        _unfreeze_param_attr(self.backbone, "pos_embed")
        _unfreeze_param_attr(self.backbone, "cls_token")
        _unfreeze_param_attr(self.backbone, "dist_token")
        _unfreeze_param_attr(self.backbone, "reg_token")

    # -------------------------------------------------------------------------------------------------
    # Public controls
    # -------------------------------------------------------------------------------------------------

    def set_gaze_backprop(self, enabled: bool) -> None:
        """
        Controls whether attention tensors are kept with gradients (for KL backward) or detached.

        Training code sets this based on:
          use_gaze_kl AND (not use_nobp)
        """
        self.gaze_backprop_enabled = bool(enabled)
        if self.attn_recorder is not None:
            keep = bool(enabled) and bool(self.training) and bool(self.gaze_requires_grad)
            self.attn_recorder.set_keep_grad(keep)

    def remove_attention_hooks(self) -> None:
        if self.attn_recorder is not None:
            self.attn_recorder.detach(self.backbone)

    def train(self, mode: bool = True):
        """
        If the backbone is fully frozen, force backbone eval mode.
        """
        super().train(mode)
        backbone_has_grad = any(p.requires_grad for p in self.backbone.parameters())
        if not backbone_has_grad:
            self.backbone.eval()
        return self

    # -------------------------------------------------------------------------------------------------
    # Heads
    # -------------------------------------------------------------------------------------------------

    def _rank_score(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.rank_fc_1(pooled)
        x = self.rank_relu(x)
        x = self.rank_drop(x)

        x = self.rank_fc_2(x)
        x = self.rank_relu(x)
        x = self.rank_drop(x)

        return self.rank_fc_out(x)

    def _fusion_logits(self, left_vec: torch.Tensor, right_vec: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([left_vec, right_vec], dim=-1)
        pair = self.pair_norm(pair)

        x = self.cross_fc_1(pair)
        x = self.cross_relu_1(x)
        x = self.cross_drop_1(x)

        x = self.cross_fc_2(x)
        x = self.cross_relu_2(x)
        x = self.cross_drop_2(x)

        x = self.cross_fc_3(x)
        x = self.cross_relu_3(x)
        x = self.cross_drop_3(x)

        return self.cross_fc_out(x)

    # -------------------------------------------------------------------------------------------------
    # Forward helpers
    # -------------------------------------------------------------------------------------------------

    def _compute_attention_require_grad(self) -> bool:
        """
        Attention gradients are meaningful only when:
          - attention recording is enabled
          - attention maps are requested (return_attn)
          - model is in training mode
          - backbone has trainable parameters
        """
        if not (self.attn_cfg.enabled and self.attn_cfg.return_attn):
            return False
        if not self.training:
            return False
        backbone_has_grad = any(p.requires_grad for p in self.backbone.parameters())
        return bool(backbone_has_grad)

    def _forward_one(
        self,
        x: torch.Tensor,
        gaze_map: Optional[torch.Tensor],
        has_eye_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # --------------------------------------------------------------
        # Step A) Configure attention recorder gradient behavior
        # --------------------------------------------------------------
        self.gaze_requires_grad = self._compute_attention_require_grad()

        if self.attn_recorder is not None:
            self.attn_recorder.set_keep_grad(bool(self.gaze_requires_grad and self.gaze_backprop_enabled))
            self.attn_recorder.begin_capture()

        # --------------------------------------------------------------
        # Step B) Backbone forward (optionally guided)
        # --------------------------------------------------------------
        try:
            feats = forward_backbone_tokens(
                self.backbone,
                x,
                attention_recorder=self.attn_recorder,
                guidance=self.guidance,
                gaze_map=gaze_map,
                has_eye_mask=has_eye_mask,
                num_prefix_tokens=int(self.num_prefix_tokens),
            )
        finally:
            if self.attn_recorder is not None:
                self.attn_recorder.end_capture()

        # --------------------------------------------------------------
        # Step C) Pool tokens -> feature vector
        # --------------------------------------------------------------
        pooled = pool_tokens(
            feats,
            pooling=str(self.cfg.pooling),
            num_prefix_tokens=int(self.num_prefix_tokens),
            pool_k=int(self.cfg.pool_k),
            apply_token_norm=bool(self.apply_token_norm),
            token_norm=self.token_norm,
        )
        pooled = self.feat_norm(pooled)

        # --------------------------------------------------------------
        # Step D) Rank score head
        # --------------------------------------------------------------
        score = self._rank_score(pooled)

        # --------------------------------------------------------------
        # Step E) Attention map (optional)
        # --------------------------------------------------------------
        attn_map: Optional[torch.Tensor] = None
        used_uniform = False

        if self.attn_recorder is not None:
            attn_map, used_uniform = self.attn_recorder.attention_map_and_meta(
                feats_for_dtype=feats if feats.ndim >= 2 else pooled,
                num_prefix_tokens=int(self.num_prefix_tokens),
                out_hw=tuple(self.attn_cfg.out_hw),
            )

            if attn_map is None:
                b = int(x.shape[0])
                attn_map = uniform_attention_map(
                    b=b,
                    out_hw=tuple(self.attn_cfg.out_hw),
                    device=x.device,
                    dtype=pooled.dtype,
                )
                used_uniform = True

        self._last_branch_used_uniform = bool(used_uniform)
        return pooled, score, attn_map

    # -------------------------------------------------------------------------------------------------
    # Forward (divided into clear steps)
    # -------------------------------------------------------------------------------------------------

    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        gaze_left: Optional[torch.Tensor] = None,
        gaze_right: Optional[torch.Tensor] = None,
        has_eye_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # --------------------------------------------------------------
        # Step 1) Left branch forward
        # --------------------------------------------------------------
        pooled_l, score_l, attn_l = self._forward_one(x_left, gaze_left, has_eye_mask)
        used_uniform_l = bool(getattr(self, "_last_branch_used_uniform", False))

        # --------------------------------------------------------------
        # Step 2) Right branch forward
        # --------------------------------------------------------------
        pooled_r, score_r, attn_r = self._forward_one(x_right, gaze_right, has_eye_mask)
        used_uniform_r = bool(getattr(self, "_last_branch_used_uniform", False))

        # --------------------------------------------------------------
        # Step 3) Attention meta tracking (debugging / logging)
        # --------------------------------------------------------------
        self.last_attn_meta = {
            "left": {"attn_map_is_none": (attn_l is None), "used_uniform": used_uniform_l},
            "right": {"attn_map_is_none": (attn_r is None), "used_uniform": used_uniform_r},
        }

        # --------------------------------------------------------------
        # Step 4) Pairwise classification logits (model-dependent)
        # --------------------------------------------------------------
        if self.cfg.model in ("sscnn", "rsscnn"):
            logits = self._fusion_logits(pooled_l, pooled_r)
        else:
            b = int(x_left.shape[0])
            logits = torch.zeros((b, int(self.cfg.num_classes)), device=x_left.device, dtype=pooled_l.dtype)

        # --------------------------------------------------------------
        # Step 5) Package outputs (legacy-compatible)
        # --------------------------------------------------------------
        return {
            "left": {"output": score_l, "attn_map": attn_l},
            "right": {"output": score_r, "attn_map": attn_r},
            "logits": {"output": logits},
        }
