import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Pairwise CNN model for ranking and/or classification.

    Supports two dense-head input conventions:
      - GAP features:        flat_dim = C
      - Flattened features: flat_dim = C * H * W  (VGG-style: 512*7*7 = 25088)

    Parameters
    ----------
    backbone : callable
        Torchvision constructor like torchvision.models.vgg19, resnet50, etc.
        Must accept weights=... (torchvision >= 0.13) or pretrained=...
    model : str
        'rcnn' | 'sscnn' | 'rsscnn'
    finetune : bool
        If False, backbone params are frozen.
    num_classes : int
        2 or 3 depending on ties setup.
    gaze_grid_size : int
        Output grid size for CAM-like attention maps.
    flatten_spatial : bool
        If True, dense heads see C*H*W (matches big VGG checkpoints).
    flat_dim_override : int | None
        If set, forces dense head input to exactly this dim (best way to match ckpts).
    dropout_p : float
        Dropout probability in heads.
    """

    def __init__(
        self,
        backbone,
        model: str,
        finetune: bool = False,
        num_classes: int = 3,
        gaze_grid_size: int = 14,
        flatten_spatial: bool = False,
        flat_dim_override: int | None = None,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.model = str(model)
        self.gaze_grid_size = int(gaze_grid_size)

        # -----------------------------
        # Backbone feature extractor (keeps spatial grid)
        # -----------------------------
        try:
            self.cnn = backbone(weights="DEFAULT").features
        except Exception:
            self.cnn = nn.Sequential(*list(backbone(weights="DEFAULT").children())[:-2])

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        if not finetune:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # -----------------------------
        # Infer feature map (C,H,W)
        # -----------------------------
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.cnn(dummy)
            if feat.dim() == 2:
                C = feat.size(1)
                H = W = 1
            else:
                _, C, H, W = feat.size()

        self.cnn_channels = int(C)
        self.cnn_h = int(H)
        self.cnn_w = int(W)

        # -----------------------------
        # Decide flatten vs GAP
        # -----------------------------
        if flat_dim_override is not None:
            self.flat_dim = int(flat_dim_override)
            self.flatten_spatial = (self.flat_dim != self.cnn_channels)
        else:
            self.flatten_spatial = bool(flatten_spatial)
            self.flat_dim = int(C * H * W) if self.flatten_spatial else int(C)

        # -----------------------------
        # Ranking head
        # -----------------------------
        self.rank_fc_1 = nn.Linear(self.flat_dim, 4096)
        self.rank_fc_out = nn.Linear(4096, 1)
        self.rank_relu = nn.ReLU()
        self.rank_drop = nn.Dropout(dropout_p)

        # -----------------------------
        # Cross-branch classification head
        # -----------------------------
        self.cross_fc_1 = nn.Linear(self.flat_dim * 2, 512)
        self.cross_relu_1 = nn.ReLU()
        self.cross_drop_1 = nn.Dropout(dropout_p)

        self.cross_fc_2 = nn.Linear(512, 512)
        self.cross_relu_2 = nn.ReLU()
        self.cross_drop_2 = nn.Dropout(dropout_p)

        self.cross_fc_3 = nn.Linear(512, int(num_classes))

    # -----------------------------
    # Backbone forward
    # -----------------------------
    def _forward_backbone(self, x: torch.Tensor):
        feat = self.cnn(x)

        if feat.dim() == 2:
            feat_map = feat.unsqueeze(-1).unsqueeze(-1)
            flat = feat
            return feat_map, flat

        feat_map = feat
        if self.flatten_spatial:
            flat = feat_map.flatten(1)
        else:
            pooled = self.global_pool(feat_map)
            flat = pooled.flatten(1)

        return feat_map, flat

    # -----------------------------
    # CAM-like attention (for gaze comparisons / KL)
    # -----------------------------
    def _extract_cam_like_attn(self, feat_map: torch.Tensor) -> torch.Tensor:
        cam = feat_map.mean(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.interpolate(
            cam,
            size=(self.gaze_grid_size, self.gaze_grid_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # [B,grid,grid]

        B = cam.shape[0]
        flat = cam.view(B, -1).clamp(min=1e-8)
        flat = flat / flat.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return flat.view(B, self.gaze_grid_size, self.gaze_grid_size)

    # -----------------------------
    # Single-branch ranking
    # -----------------------------
    def single_forward_ranking(self, batch: torch.Tensor):
        feat_map, flat = self._forward_backbone(batch)

        x = self.rank_fc_1(flat)
        x = self.rank_relu(x)
        x = self.rank_drop(x)
        score = self.rank_fc_out(x)

        attn_map = self._extract_cam_like_attn(feat_map)
        return {"output": score, "attn_map": attn_map}

    # -----------------------------
    # Fusion classification
    # -----------------------------
    def single_forward_fusion(self, batch_left: torch.Tensor, batch_right: torch.Tensor):
        feat_map_l, flat_l = self._forward_backbone(batch_left)
        feat_map_r, flat_r = self._forward_backbone(batch_right)

        x = torch.cat((flat_l, flat_r), dim=1)
        x = self.cross_fc_1(x)
        x = self.cross_relu_1(x)
        x = self.cross_drop_1(x)

        x = self.cross_fc_2(x)
        x = self.cross_relu_2(x)
        x = self.cross_drop_2(x)

        logits = self.cross_fc_3(x)
        return {"output": logits, "feat_map_left": feat_map_l, "feat_map_right": feat_map_r}

    # -----------------------------
    # Pairwise forward (project API)
    # -----------------------------
    def forward(self, left_batch: torch.Tensor, right_batch: torch.Tensor):
        if self.model == "rcnn":
            return {"left": self.single_forward_ranking(left_batch),
                    "right": self.single_forward_ranking(right_batch)}

        if self.model == "sscnn":
            fusion = self.single_forward_fusion(left_batch, right_batch)
            return {"logits": {"output": fusion["output"]}}

        if self.model == "rsscnn":
            left_out = self.single_forward_ranking(left_batch)
            right_out = self.single_forward_ranking(right_batch)
            fusion = self.single_forward_fusion(left_batch, right_batch)
            return {"left": left_out, "right": right_out, "logits": {"output": fusion["output"]}}

        raise ValueError(f"Unknown model type: {self.model}")


def infer_cnn_kwargs_from_state_dict(state_dict: dict) -> dict:
    """
    Infer dense-head shapes from an existing checkpoint state_dict.

    Returns kwargs usable in CNN(...):
      - flat_dim_override: rank_fc_1 input dim
      - num_classes:       cross_fc_3 output dim
    """
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    return {
        "flat_dim_override": int(sd["rank_fc_1.weight"].shape[1]),
        "num_classes": int(sd["cross_fc_3.weight"].shape[0]),
    }
