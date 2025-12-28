import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CNN(nn.Module):
    """
    Pairwise CNN model for ranking and/or classification.

    Modes (self.model):
      - 'rcnn'  : Ranking loss only
      - 'sscnn' : Classification loss only
      - 'rsscnn': Ranking + classification + (optional) attention KL
    """

    def __init__(self, backbone, model: str, finetune: bool = False, num_classes: int = 3):
        super().__init__()
        self.model = model  # 'rcnn' | 'sscnn' | 'rsscnn'

        # ------------------------------------------------------------------
        # Backbone: get convolutional feature extractor
        # ------------------------------------------------------------------
        # torchvision-style models (alexnet, vgg, densenet, resnet)
        try:
            # Models with .features (e.g., alexnet, vgg, densenet)
            self.cnn = backbone(weights='DEFAULT').features
        except AttributeError:
            # Models like resnet: use everything except the final pooling+fc
            self.cnn = nn.Sequential(*list(backbone(weights='DEFAULT').children())[:-1])

        # Optionally freeze backbone
        if not finetune:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------
        # Infer feature-map size with a dummy forward
        # ------------------------------------------------------------------
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self.cnn(dummy)  # [1, C, H, W] or [1, C]
            if feat.dim() == 2:
                # e.g., some backbones may already pool to [B, C]
                C = feat.size(1)
                H = W = 1
            else:
                _, C, H, W = feat.size()

        self.cnn_channels = C
        self.cnn_h = H
        self.cnn_w = W
        self.flat_dim = C * H * W

        # ------------------------------------------------------------------
        # Ranking head (single-branch)
        # ------------------------------------------------------------------
        self.rank_fc_1 = nn.Linear(self.flat_dim, 4096)
        self.rank_fc_out = nn.Linear(4096, 1)
        self.rank_relu = nn.ReLU()
        self.rank_drop = nn.Dropout(0.3)

        # ------------------------------------------------------------------
        # Cross-branch classification head (fusion of two branches)
        # ------------------------------------------------------------------
        self.cross_fc_1 = nn.Linear(self.flat_dim * 2, 512)
        self.cross_relu_1 = nn.ReLU()
        self.cross_drop_1 = nn.Dropout(0.3)

        self.cross_fc_2 = nn.Linear(512, 512)
        self.cross_relu_2 = nn.ReLU()
        self.cross_drop_2 = nn.Dropout(0.3)

        self.cross_fc_3 = nn.Linear(512, num_classes)

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _forward_backbone(self, x: torch.Tensor):
        """
        Run CNN backbone and return feature map [B,C,H,W] and flattened features [B,flat_dim].
        """
        feat = self.cnn(x)  # usually [B, C, H, W] or [B, C, 1, 1]
        if feat.dim() == 2:
            # [B, C] -> treat as [B, C, 1, 1]
            feat_map = feat.unsqueeze(-1).unsqueeze(-1)
        else:
            feat_map = feat

        B = feat_map.size(0)
        flat = feat_map.view(B, -1)
        return feat_map, flat

    def _extract_cam_like_attn(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Produce a coarse attention map [B,14,14] from convolutional features:
        - Global-average pool over channels -> [B, H, W]
        - Upsample to 14x14 (to match gaze maps)
        - Normalize to form a probability distribution per sample
        """
        # feat_map: [B,C,H,W]
        cam = feat_map.mean(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.interpolate(cam, size=(14, 14), mode='bilinear', align_corners=False)
        cam = cam.squeeze(1)  # [B,14,14]

        # L1 normalize per sample
        B = cam.shape[0]
        flat = cam.view(B, -1)
        flat = flat.clamp(min=1e-8)
        flat = flat / flat.sum(dim=1, keepdim=True).clamp(min=1e-8)
        cam = flat.view(B, 14, 14)
        return cam

    # ----------------------------------------------------------------------
    # Single-branch ranking forward
    # ----------------------------------------------------------------------
    def single_forward_ranking(self, batch: torch.Tensor):
        """
        Forward pass for ranking on a single branch.
        Returns:
          {
            'output':  [B,1],
            'attn_map': [B,14,14]  (CNN-based CAM approximation)
          }
        """
        feat_map, flat = self._forward_backbone(batch)  # [B,C,H,W], [B,flat_dim]

        # Ranking head
        x = self.rank_fc_1(flat)
        x = self.rank_relu(x)
        x = self.rank_drop(x)
        score = self.rank_fc_out(x)  # [B,1]

        # CAM-like attention map for KL loss (if used)
        attn_map = self._extract_cam_like_attn(feat_map)  # [B,14,14]

        return {'output': score, 'attn_map': attn_map}

    # ----------------------------------------------------------------------
    # Cross-branch fusion forward (classification)
    # ----------------------------------------------------------------------
    def single_forward_fusion(self, batch_left: torch.Tensor, batch_right: torch.Tensor):
        """
        Forward pass for classification using concatenated features.
        Returns:
          {
            'output': logits [B,num_classes],
            'features_left':  [B,flat_dim],
            'features_right': [B,flat_dim],
          }
        """
        feat_map_l, flat_l = self._forward_backbone(batch_left)
        feat_map_r, flat_r = self._forward_backbone(batch_right)

        x = torch.cat((flat_l, flat_r), dim=1)  # [B, 2*flat_dim]
        x = self.cross_fc_1(x)
        x = self.cross_relu_1(x)
        x = self.cross_drop_1(x)

        x = self.cross_fc_2(x)
        x = self.cross_relu_2(x)
        x = self.cross_drop_2(x)

        logits = self.cross_fc_3(x)

        return {
            'output': logits,
            'features_left': flat_l,
            'features_right': flat_r,
            'feat_map_left': feat_map_l,
            'feat_map_right': feat_map_r,
        }

    # ----------------------------------------------------------------------
    # Unified pairwise forward (API expected by train_script.py & losses.py)
    # ----------------------------------------------------------------------
    def forward(self, left_batch: torch.Tensor, right_batch: torch.Tensor):
        """
        Pairwise forward used throughout the training code.

        Returns (depending on self.model):

        - 'rcnn':
            {
              'left':  {'output': [B,1], 'attn_map': [B,14,14]},
              'right': {'output': [B,1], 'attn_map': [B,14,14]},
            }

        - 'sscnn':
            {
              'logits': {'output': [B,num_classes]},
            }

        - 'rsscnn':
            {
              'left':  {'output': [B,1], 'attn_map': [B,14,14]},
              'right': {'output': [B,1], 'attn_map': [B,14,14]},
              'logits': {'output': [B,num_classes]},
            }
        """
        if self.model == 'rcnn':
            left_out = self.single_forward_ranking(left_batch)
            right_out = self.single_forward_ranking(right_batch)
            return {
                'left': {
                    'output': left_out['output'],
                    'attn_map': left_out['attn_map'],
                },
                'right': {
                    'output': right_out['output'],
                    'attn_map': right_out['attn_map'],
                },
            }

        elif self.model == 'sscnn':
            fusion = self.single_forward_fusion(left_batch, right_batch)
            return {
                'logits': {
                    'output': fusion['output'],
                }
            }

        elif self.model == 'rsscnn':
            left_out = self.single_forward_ranking(left_batch)
            right_out = self.single_forward_ranking(right_batch)
            fusion = self.single_forward_fusion(left_batch, right_batch)
            return {
                'left': {
                    'output': left_out['output'],
                    'attn_map': left_out['attn_map'],
                },
                'right': {
                    'output': right_out['output'],
                    'attn_map': right_out['attn_map'],
                },
                'logits': {
                    'output': fusion['output'],
                }
            }

        else:
            raise ValueError(f"Unknown model type: {self.model}")


if __name__ == '__main__':
    # Simple smoke test
    net = CNN(backbone=models.resnet50, model='rsscnn', finetune=False, num_classes=3)
    x = torch.randn(2, 3, 224, 224)
    y = torch.randn(2, 3, 224, 224)
    out = net(x, y)
    for k, v in out.items():
        if isinstance(v, dict) and 'output' in v:
            print(k, 'output shape:', v['output'].shape)
        else:
            print(k, type(v))