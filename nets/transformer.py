import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    """
    Pairwise model on top of a ViT/DeiT-style backbone.

    Modes (self.model):
      - 'rcnn'  : Ranking loss only
      - 'sscnn' : Classification loss only
      - 'rsscnn': Ranking + classification + (optional) attention KL

    Expected backbone behavior:
      - Either exposes .forward_features(x) -> [B, C] or [B, T, C]
      - Or __call__(x) -> [B, C] or [B, T, C]
      - CLS token is at index 0 if tokens are returned.

    Forward(left, right) returns (depending on self.model):

      'rcnn':
        {
          'left':  {'output': [B,1], 'attn_map': [B,14,14] or None},
          'right': {'output': [B,1], 'attn_map': [B,14,14] or None},
        }

      'sscnn':
        {
          'logits': {'output': [B,num_classes]},
        }

      'rsscnn':
        {
          'left':  {'output': [B,1], 'attn_map': [B,14,14] or None},
          'right': {'output': [B,1], 'attn_map': [B,14,14] or None},
          'logits': {'output': [B,num_classes]},
        }

    If return_attn=False, attn_map will be None and no attention extraction
    is performed (to save time and GPU resources when gaze is off).
    """

    def __init__(
        self,
        backbone,
        model: str,
        num_classes: int = 2,
        finetune: bool = False,
        num_ft_blocks: int = 1,   
        rank_dropout: float = 0.3,
        cross_dropout: float = 0.3,
        use_attn_hook: bool = False,
        return_attn: bool = True,
    ):

        super().__init__()
        self.model = model  # 'rcnn' | 'sscnn' | 'rsscnn'

        # Alias so train.py / test.py can detect "transformer" vs CNN
        self.transformer = backbone
        self.backbone = backbone

        self.rank_dropout = rank_dropout
        self.cross_dropout = cross_dropout

        # Flag to control whether we actually compute/return attn maps
        self.return_attn = return_attn

        # Internal attention storage & gradient flag
        self._last_attn = None      # will store [B,H,T,T]
        self.attn_grad = False      # overridden in train.py when using gaze+KL

        # ------------------------------------------------------------------
        # Freeze backbone or partially unfreeze last N blocks when finetuning
        # ------------------------------------------------------------------
        if not finetune:
            # Fully frozen backbone (linear-probe style)
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            # First freeze everything
            for p in self.backbone.parameters():
                p.requires_grad = False

            # Then unfreeze only the last num_ft_blocks transformer blocks + final norm
            if hasattr(self.backbone, "blocks") and len(self.backbone.blocks) > 0:
                total_blocks = len(self.backbone.blocks)
                # Clamp num_ft_blocks to valid range [1, total_blocks]
                n_unfreeze = max(1, min(num_ft_blocks, total_blocks))

                for block in self.backbone.blocks[-n_unfreeze:]:
                    for p in block.parameters():
                        p.requires_grad = True

            # Also unfreeze final LayerNorm if present
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True


        # ------------------------------------------------------------------
        # Infer feature dimension from backbone
        # ------------------------------------------------------------------
        if hasattr(self.backbone, "num_features"):
            feat_dim = self.backbone.num_features
        elif hasattr(self.backbone, "embed_dim"):
            feat_dim = self.backbone.embed_dim
        elif hasattr(self.backbone, "head") and hasattr(self.backbone.head, "in_features"):
            feat_dim = self.backbone.head.in_features
        else:
            raise AttributeError(
                "Cannot infer feature dim from backbone. "
                "Expected `num_features`, `embed_dim`, or `head.in_features`."
            )
        self.feat_dim = feat_dim

        # Optional normalizations (safe to keep; they don't change head sizes)
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.pair_norm = nn.LayerNorm(feat_dim * 2)

        # ------------------------------------------------------------------
        # Ranking head:
        #   feat_dim -> 4096 -> 1
        # ------------------------------------------------------------------
        self.rank_fc_1 = nn.Linear(feat_dim, 4096)
        self.rank_relu = nn.ReLU()
        self.rank_drop = nn.Dropout(self.rank_dropout)
        self.rank_fc_out = nn.Linear(4096, 1)
    
        # ------------------------------------------------------------------
        # Cross-branch classification head:
        #   [feat_L || feat_R] -> 512 -> 512 -> num_classes
        # ------------------------------------------------------------------
        self.cross_fc_1 = nn.Linear(feat_dim * 2, 512)
        self.cross_relu_1 = nn.ReLU()
        self.cross_drop_1 = nn.Dropout(self.cross_dropout)

        self.cross_fc_2 = nn.Linear(512, 512)
        self.cross_relu_2 = nn.ReLU()
        self.cross_drop_2 = nn.Dropout(self.cross_dropout)

        self.cross_fc_3 = nn.Linear(512, num_classes)

        """
        # ------------------------------------------------------------------
        # Cross-branch classification head (your requested architecture)
        #   [CLS_left || CLS_right] → 2*feat_dim
        #       → Linear(2*feat_dim → 256)
        #       → GELU
        #       → Dropout(cross_dropout)
        #       → Linear(256 → num_classes)
        # ------------------------------------------------------------------
        self.cross_fc_1 = nn.Linear(feat_dim * 2, 256)
        self.cross_gelu = nn.GELU()
        self.cross_drop = nn.Dropout(self.cross_dropout)
        self.cross_fc_out = nn.Linear(256, num_classes)
        """
        # ------------------------------------------------------------------
        # Attention capture: last block attention [B,H,T,T]
        # ------------------------------------------------------------------
        if use_attn_hook:
            self._patch_last_block_attention()

    # ----------------------------------------------------------------------
    # Hook last transformer block to capture attention maps
    # ----------------------------------------------------------------------
    def _patch_last_block_attention(self):
        vt = self.backbone
        if not hasattr(vt, "blocks") or len(vt.blocks) == 0:
            return

        last_block = vt.blocks[-1]
        if not hasattr(last_block, "attn"):
            return

        attn_module = last_block.attn
        orig_forward = attn_module.forward

        def forward_with_attn_capture(x, *args, **kwargs):
            # Capture attention weights for CLS->patch map
            try:
                B, N, C = x.shape
                qkv = attn_module.qkv(x)  # (B, N, 3*C)
                qkv = qkv.reshape(B, N, 3, attn_module.num_heads, C // attn_module.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
                q, k, v = qkv[0], qkv[1], qkv[2]

                attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                attn = attn.softmax(dim=-1)  # [B, H, T, T]

                if self.training and getattr(self, "attn_grad", False):
                    self._last_attn = attn
                else:
                    self._last_attn = attn.detach()
            except Exception:
                # If something changes in timm internals, just skip capture gracefully
                self._last_attn = None

            # Call original forward for actual output
            return orig_forward(x, *args, **kwargs)

        attn_module.forward = forward_with_attn_capture

    # ----------------------------------------------------------------------
    # Normalize backbone outputs to CLS embeddings [B, C]
    # ----------------------------------------------------------------------
    @staticmethod
    def _to_cls_feats(feats: torch.Tensor) -> torch.Tensor:
        # Some models return dicts
        if isinstance(feats, dict):
            if "x" in feats:
                feats = feats["x"]
            elif "last_hidden_state" in feats:
                feats = feats["last_hidden_state"]
            else:
                for k in ["feat", "features", "tokens"]:
                    if k in feats:
                        feats = feats[k]
                        break

        # [B, C] → already CLS
        if feats.dim() == 2:
            return feats
        # [B, T, C] → CLS at index 0
        elif feats.dim() == 3:
            return feats[:, 0]
        else:
            raise ValueError(f"Unexpected features shape: {feats.shape}")

    # ----------------------------------------------------------------------
    # Extract CLS → patches attention map and resize to [B,14,14]
    # ----------------------------------------------------------------------
    def _extract_cls_attention_map(self, batch_size: int, device, dtype):
        """
        Returns an attention map [B,14,14] suitable for gaze KL loss.

        If attention cannot be extracted, returns a uniform map over 14x14.
        """
        if self._last_attn is None or self._last_attn.dim() != 4:
            # Fallback: uniform attention
            return torch.full(
                (batch_size, 14, 14),
                1.0 / (14 * 14),
                device=device,
                dtype=dtype,
            )

        attn = self._last_attn  # [B, H, T, T]
        B = attn.shape[0]

        # Mean over heads → [B, T, T]
        attn = attn.mean(dim=1)

        # CLS → all tokens: [B, T]
        cls_to_all = attn[:, 0]  # use CLS token row

        # Drop CLS→CLS, keep CLS→patches: [B, T-1]
        cls_to_patches = cls_to_all[:, 1:]

        # Reshape to a grid if possible, then interpolate to 14x14
        num_patches = cls_to_patches.size(1)
        g = int(math.sqrt(num_patches))

        if g * g == num_patches:
            attn_map = cls_to_patches.view(B, 1, g, g)
        else:
            # treat as 1 × 1 × N "image" and interpolate anyway
            attn_map = cls_to_patches.view(B, 1, 1, num_patches)

        attn_map = F.interpolate(
            attn_map,
            size=(14, 14),
            mode="bilinear",
            align_corners=False,
        )  # [B,1,14,14]
        attn_map = attn_map.squeeze(1)  # [B,14,14]

        return attn_map

    # ----------------------------------------------------------------------
    # Single branch: x -> CLS feats, score [B,1], (optional) attn_map [B,14,14]
    # ----------------------------------------------------------------------
    def _forward_branch(self, x: torch.Tensor):
        # Backbone forward
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone(x)

        cls = self._to_cls_feats(feats)  # [B, C]
        cls = self.feat_norm(cls)
        B = cls.size(0)
        device = cls.device
        dtype = cls.dtype

        # Ranking head: feat_dim -> 4096 -> 1
        h = self.rank_fc_1(cls)
        h = self.rank_relu(h)
        h = self.rank_drop(h)
        score = self.rank_fc_out(h)  # [B,1]

        # Attention map [B,14,14], only if requested
        if self.return_attn:
            attn_map = self._extract_cls_attention_map(
                batch_size=B,
                device=device,
                dtype=dtype,
            )
        else:
            attn_map = None

        return cls, score, attn_map
    
    # ----------------------------------------------------------------------
    # Fusion head: from left & right CLS feats to classification logits
    # ----------------------------------------------------------------------
    def _fusion_logits_from_feats(self, feats_left, feats_right):
        pair = torch.cat([feats_left, feats_right], dim=-1)  # [B, 2*C]
        pair = self.pair_norm(pair)

        h = self.cross_fc_1(pair)
        h = self.cross_relu_1(h)
        h = self.cross_drop_1(h)

        h = self.cross_fc_2(h)
        h = self.cross_relu_2(h)
        h = self.cross_drop_2(h)

        logits = self.cross_fc_3(h)
        return logits

    """
    def _fusion_logits_from_feats(self, feats_left, feats_right):
        pair = torch.cat([feats_left, feats_right], dim=-1)  # [B, 2*feat_dim]
        pair = self.pair_norm(pair)
    
        h = self.cross_fc_1(pair)
        h = self.cross_gelu(h)
        h = self.cross_drop(h)
    
        logits = self.cross_fc_out(h)
        return logits
    """
    # ----------------------------------------------------------------------
    # Optional hook used in train.py (no-op but keeps API compatible)
    # ----------------------------------------------------------------------
    def partial_eval(self):
        """
        Placeholder to mirror the CNN API; currently a no-op.
        Can be extended for debugging / visualization if needed.
        """
        return

    # ----------------------------------------------------------------------
    # Main pairwise forward
    # ----------------------------------------------------------------------
    def forward(self, left_batch: torch.Tensor, right_batch: torch.Tensor):
        left_feats, left_score, left_attn = self._forward_branch(left_batch)
        right_feats, right_score, right_attn = self._forward_branch(right_batch)

        if self.model == "rcnn":
            # Ranking only
            return {
                "left": {"output": left_score, "attn_map": left_attn},
                "right": {"output": right_score, "attn_map": right_attn},
            }

        elif self.model == "sscnn":
            # Classification only
            logits = self._fusion_logits_from_feats(left_feats, right_feats)
            return {
                "logits": {"output": logits},
            }

        elif self.model == "rsscnn":
            # Classification + ranking
            logits = self._fusion_logits_from_feats(left_feats, right_feats)
            return {
                "left": {"output": left_score, "attn_map": left_attn},
                "right": {"output": right_score, "attn_map": right_attn},
                "logits": {"output": logits},
            }

        else:
            raise ValueError(f"Invalid model type: {self.model}")


if __name__ == "__main__":
    # Simple smoke test (requires timm for some backbones)
    try:
        import timm

        backbone = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=False,
            num_classes=0,
        )
    except Exception:
        backbone = None
        print("timm not available, skipping ViT test.")

    if backbone is not None:
        net = Transformer(
            backbone,
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
        if "left" in out:
            print(
                " left.output:", out["left"]["output"].shape,
                " left.attn_map:", None if out["left"]["attn_map"] is None else out["left"]["attn_map"].shape,
            )
        if "right" in out:
            print(
                " right.output:", out["right"]["output"].shape,
                " right.attn_map:", None if out["right"]["attn_map"] is None else out["right"]["attn_map"].shape,
            )
        if "logits" in out:
            print(" logits:", out["logits"]["output"].shape)
