# transformer_utils.py
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================================================
# Basic tensor helpers
# ================================================================================================

def _ensure_gaze_4d(gaze: torch.Tensor) -> torch.Tensor:
    if gaze.ndim == 4:
        return gaze
    if gaze.ndim == 3:
        return gaze.unsqueeze(1)
    if gaze.ndim == 2:
        return gaze.unsqueeze(1).unsqueeze(1)
    raise ValueError(f"Unsupported gaze tensor shape: {tuple(gaze.shape)}")


def uniform_attention_map(
    b: int,
    out_hw: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h, w = int(out_hw[0]), int(out_hw[1])
    m = torch.ones((b, 1, h, w), device=device, dtype=dtype)
    return m / float(h * w)


def _safe_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_backbone_input_hw(backbone: nn.Module) -> Tuple[int, int]:
    for cfg_name in ("pretrained_cfg", "default_cfg"):
        cfg = getattr(backbone, cfg_name, None)
        if isinstance(cfg, dict):
            inp = cfg.get("input_size", None)
            if isinstance(inp, (tuple, list)) and len(inp) == 3:
                return int(inp[1]), int(inp[2])
    return 224, 224


# ================================================================================================
# Backbone output normalization / inference helpers
# ================================================================================================

def _normalize_backbone_output(feats: Any) -> torch.Tensor:
    if torch.is_tensor(feats):
        return feats

    if isinstance(feats, dict):
        cls_k = None
        patch_k = None

        for ck in ("x_norm_clstoken", "clstoken", "cls_token", "x_clstoken"):
            v = feats.get(ck, None)
            if torch.is_tensor(v) and v.ndim == 2:
                cls_k = ck
                break

        for pk in ("x_norm_patchtokens", "patchtokens", "patch_tokens", "x_patchtokens"):
            v = feats.get(pk, None)
            if torch.is_tensor(v) and v.ndim == 3:
                patch_k = pk
                break

        if cls_k is not None and patch_k is not None:
            cls_tok = feats[cls_k].unsqueeze(1)
            patch_tok = feats[patch_k]
            return torch.cat([cls_tok, patch_tok], dim=1)

        candidate_keys = ("x", "tokens", "last_hidden_state", "feats", "features", "penultimate", "pre_logits", "logits")
        for k in candidate_keys:
            v = feats.get(k, None)
            if torch.is_tensor(v):
                return v

        for v in feats.values():
            if torch.is_tensor(v):
                return v

        raise TypeError(f"Backbone returned dict with no tensor values. Keys={list(feats.keys())}")

    if isinstance(feats, (tuple, list)):
        for v in feats:
            if torch.is_tensor(v) and v.ndim == 3:
                return v
        for v in feats:
            if torch.is_tensor(v) and v.ndim == 2:
                return v
        for v in feats:
            if torch.is_tensor(v):
                return v
        raise TypeError("Backbone returned tuple/list with no tensor entries.")

    raise TypeError(f"Unsupported backbone output type: {type(feats)}")


def infer_embed_dim(backbone: nn.Module) -> int:
    if hasattr(backbone, "embed_dim"):
        return int(getattr(backbone, "embed_dim"))
    if hasattr(backbone, "num_features"):
        return int(getattr(backbone, "num_features"))

    device = _safe_module_device(backbone)
    h, w = _get_backbone_input_hw(backbone)
    dummy = torch.zeros(1, 3, h, w, device=device)

    with torch.no_grad():
        feats = backbone.forward_features(dummy) if hasattr(backbone, "forward_features") else backbone(dummy)

    t = _normalize_backbone_output(feats)
    if t.ndim in (2, 3):
        return int(t.shape[-1])
    raise ValueError(f"Unexpected normalized backbone output shape: {tuple(t.shape)}")


def infer_num_prefix_tokens(backbone: nn.Module, force: Optional[int] = None) -> int:
    if force is not None:
        return int(force)
    npt = getattr(backbone, "num_prefix_tokens", None)
    if npt is not None:
        return int(npt)
    return 1


def infer_patch_grid(backbone: nn.Module, num_patches: Optional[int] = None) -> Tuple[int, int]:
    pe = getattr(backbone, "patch_embed", None)
    if pe is not None:
        gs = getattr(pe, "grid_size", None)
        if isinstance(gs, (tuple, list)) and len(gs) == 2:
            return int(gs[0]), int(gs[1])

        np = getattr(pe, "num_patches", None)
        if isinstance(np, int) and np > 0:
            g = int(math.isqrt(np))
            if g * g == int(np):
                return g, g

    if num_patches is not None and int(num_patches) > 0:
        g = int(math.isqrt(int(num_patches)))
        if g * g == int(num_patches):
            return g, g

    return 14, 14


# ================================================================================================
# Attention extraction (align mode)
# ================================================================================================

@dataclass(frozen=True)
class AttentionConfig:
    """
    mode:
      - "last":    last block CLS->patch attention (head-averaged)
      - "rollout": rollout across blocks (identity-augmented, row-normalized)
      - "topk":    "last" attention sparsified to top-k patches

    out_hw:
      attention map spatial output size (H,W), typically gaze grid size.
    """
    enabled: bool = False
    return_attn: bool = True
    mode: str = "last"                  # {"last","rollout","topk"}
    topk: Optional[int] = None
    out_hw: Tuple[int, int] = (14, 14)


class AttentionRecorder:
    """
    Monkeypatch-based recorder for timm-style ViT Attention modules.

    The hooked forward computes and stores attn_pre = softmax(qk^T * scale) before dropout.
    When args/kwargs are present, the original forward is called and attention is approximated
    from qkv(x) as a fallback.
    """
    def __init__(self, cfg: AttentionConfig) -> None:
        self.cfg = cfg

        self._attn_hooked: bool = False
        self._original_attn_forwards: Dict[int, Any] = {}
        self._hooked_modules: List[nn.Module] = []

        self._attn_mats: List[torch.Tensor] = []
        self._last_attn: Optional[torch.Tensor] = None

        self._active_attn_sink: Optional[List[torch.Tensor]] = None
        self._active_last_attn: Optional[torch.Tensor] = None

        self._keep_grad: bool = False
        self._fallback_calls: int = 0
        self._fallback_warned: int = 0

        self._last_used_uniform: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled)

    def set_keep_grad(self, enabled: bool) -> None:
        self._keep_grad = bool(enabled)

    def reset(self) -> None:
        self._attn_mats = []
        self._last_attn = None
        self._active_attn_sink = None
        self._active_last_attn = None
        self._last_used_uniform = False

    def attach(self, backbone: nn.Module) -> None:
        if self._attn_hooked or (not self.enabled):
            return

        hooked_any = False
        for m in backbone.modules():
            qkv = getattr(m, "qkv", None)
            proj = getattr(m, "proj", None)
            if not (isinstance(qkv, nn.Linear) and isinstance(proj, nn.Linear)):
                continue
            if not hasattr(m, "num_heads"):
                continue
            if not hasattr(m, "attn_drop"):
                continue
            if not hasattr(m, "proj_drop"):
                continue

            self._hook_attention_module(m)
            hooked_any = True

        self._attn_hooked = hooked_any
        if self.enabled and (not hooked_any):
            warnings.warn("AttentionConfig.enabled=True but no compatible attention modules were found/hooked.")

    def detach(self, backbone: nn.Module) -> None:
        if not self._original_attn_forwards:
            self._attn_hooked = False
            self.reset()
            return

        for m in backbone.modules():
            mid = id(m)
            if mid in self._original_attn_forwards:
                m.forward = self._original_attn_forwards[mid]

        self._original_attn_forwards.clear()
        self._hooked_modules.clear()
        self._attn_hooked = False
        self.reset()

    def begin_capture(self) -> None:
        self.reset()
        local_mats: List[torch.Tensor] = []
        self._active_attn_sink = local_mats
        self._active_last_attn = None

    def end_capture(self) -> None:
        self._attn_mats = [] if self._active_attn_sink is None else list(self._active_attn_sink)
        self._last_attn = self._active_last_attn
        self._active_attn_sink = None
        self._active_last_attn = None

    def _hook_attention_module(self, mod: nn.Module) -> None:
        mid = id(mod)
        if mid in self._original_attn_forwards:
            return

        orig_forward = mod.forward
        self._original_attn_forwards[mid] = orig_forward

        def _store_attn(attn_pre: torch.Tensor) -> None:
            attn_store = attn_pre if self._keep_grad else attn_pre.detach()
            self._active_last_attn = attn_store
            if (self.cfg.mode == "rollout") and (self._active_attn_sink is not None):
                self._active_attn_sink.append(attn_store)

        def _compute_attn_pre_from_x(x_in: torch.Tensor, _mod=mod) -> Optional[torch.Tensor]:
            if x_in.ndim != 3:
                return None

            b, n, c = x_in.shape
            num_heads = int(getattr(_mod, "num_heads", 0))
            if num_heads <= 0 or (c % num_heads) != 0:
                return None

            head_dim = c // num_heads
            qkv = _mod.qkv(x_in)
            qkv = qkv.reshape(b, n, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k = qkv[0], qkv[1]

            scale = getattr(_mod, "scale", head_dim ** -0.5)
            attn_logits = (q @ k.transpose(-2, -1)) * scale
            return attn_logits.softmax(dim=-1)

        def wrapped_forward(
            x: torch.Tensor,
            *args: Any,
            _mod=mod,
            _orig=orig_forward,
            **kwargs: Any,
        ):
            want_attn = bool(self.cfg.enabled and self.cfg.return_attn)
            if not want_attn:
                return _orig(x, *args, **kwargs)

            if args or kwargs:
                out = _orig(x, *args, **kwargs)

                self._fallback_calls += 1
                if self._fallback_warned < 5:
                    self._fallback_warned += 1
                    warnings.warn(
                        "Attention hook fallback triggered due to args/kwargs (mask/bias/etc). "
                        "Attention is approximated from qkv(x)."
                    )

                try:
                    attn_pre = _compute_attn_pre_from_x(x, _mod=_mod)
                    if attn_pre is not None:
                        _store_attn(attn_pre)
                except Exception:
                    pass

                return out

            try:
                if x.ndim != 3:
                    return _orig(x, *args, **kwargs)

                b, n, c = x.shape
                num_heads = int(getattr(_mod, "num_heads", 0))
                if num_heads <= 0 or (c % num_heads) != 0:
                    return _orig(x, *args, **kwargs)

                head_dim = c // num_heads

                qkv = _mod.qkv(x)
                qkv = qkv.reshape(b, n, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]

                scale = getattr(_mod, "scale", head_dim ** -0.5)
                attn_logits = (q @ k.transpose(-2, -1)) * scale
                attn_pre = attn_logits.softmax(dim=-1)

                attn_fwd = _mod.attn_drop(attn_pre) if hasattr(_mod, "attn_drop") else attn_pre
                _store_attn(attn_pre)

                out = (attn_fwd @ v).transpose(1, 2).reshape(b, n, c)
                out = _mod.proj(out) if hasattr(_mod, "proj") else out
                out = _mod.proj_drop(out) if hasattr(_mod, "proj_drop") else out
                return out
            except Exception:
                return _orig(x, *args, **kwargs)

        mod.forward = wrapped_forward
        self._hooked_modules.append(mod)

    @staticmethod
    def _patch_vector_to_map(
        patch_scores: torch.Tensor,
        out_hw: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        mode: str,
        topk: Optional[int],
    ) -> torch.Tensor:
        b, p = patch_scores.shape
        patch_scores = patch_scores.to(device=device, dtype=dtype)

        if mode == "topk":
            k = topk
            if k is None:
                k = max(1, int(0.10 * p))
            k = max(1, min(int(k), p))
            thr = patch_scores.topk(k, dim=1).values[:, -1].unsqueeze(1)
            patch_scores = torch.where(patch_scores >= thr, patch_scores, torch.zeros_like(patch_scores))
            s = patch_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
            patch_scores = patch_scores / s

        grid = int(math.isqrt(p))
        h, w = int(out_hw[0]), int(out_hw[1])

        if grid * grid == p:
            m = patch_scores.view(b, 1, grid, grid)
            return F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)

        m = patch_scores.view(b, 1, p, 1)
        m = F.interpolate(m, size=(h, 1), mode="bilinear", align_corners=False)
        m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
        return m

    def attention_map_and_meta(
        self,
        feats_for_dtype: torch.Tensor,
        num_prefix_tokens: int,
        out_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[torch.Tensor], bool]:
        if not (self.cfg.enabled and self.cfg.return_attn):
            return None, False

        out_hw_eff = self.cfg.out_hw if out_hw is None else tuple(out_hw)

        if self.cfg.mode == "rollout":
            m = self._attention_rollout_map(feats_for_dtype, num_prefix_tokens, out_hw_eff)
        else:
            m = self._attention_last_map(feats_for_dtype, num_prefix_tokens, out_hw_eff)

        used_uniform = False
        if m is None:
            b = int(feats_for_dtype.shape[0])
            m = uniform_attention_map(b=b, out_hw=out_hw_eff, device=feats_for_dtype.device, dtype=feats_for_dtype.dtype)
            used_uniform = True

        self._last_used_uniform = bool(used_uniform)
        return m, used_uniform

    def _attention_last_map(
        self,
        feats_for_dtype: torch.Tensor,
        num_prefix_tokens: int,
        out_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if self._last_attn is None:
            return None

        attn = self._last_attn.mean(dim=1)  # (B,N,N)
        if attn.shape[-1] <= int(num_prefix_tokens):
            return None

        patch_scores = attn[:, 0, int(num_prefix_tokens):]  # (B,P)
        patch_scores = patch_scores / patch_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)

        mode = "topk" if (self.cfg.mode == "topk") else "last"
        return self._patch_vector_to_map(
            patch_scores,
            out_hw=out_hw,
            device=feats_for_dtype.device,
            dtype=feats_for_dtype.dtype,
            mode=mode,
            topk=self.cfg.topk,
        )

    def _attention_rollout_map(
        self,
        feats_for_dtype: torch.Tensor,
        num_prefix_tokens: int,
        out_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if len(self._attn_mats) == 0:
            return None

        device = feats_for_dtype.device
        out_dtype = feats_for_dtype.dtype

        mats: List[torch.Tensor] = []
        for a in self._attn_mats:
            A = a.mean(dim=1)  # (B,N,N)
            if A.device != device:
                A = A.to(device)
            mats.append(A)

        b, n, _ = mats[0].shape
        I = torch.eye(n, device=device, dtype=mats[0].dtype).unsqueeze(0).expand(b, -1, -1)

        mats_hat: List[torch.Tensor] = []
        for A in mats:
            A = A + I
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            mats_hat.append(A)

        R = mats_hat[0]
        for A in mats_hat[1:]:
            R = R @ A

        if R.shape[-1] <= int(num_prefix_tokens):
            return None

        patch_scores = R[:, 0, int(num_prefix_tokens):]  # (B,P)
        patch_scores = patch_scores / patch_scores.sum(dim=1, keepdim=True).clamp_min(1e-12)

        if patch_scores.dtype != out_dtype:
            patch_scores = patch_scores.to(dtype=out_dtype)

        return self._patch_vector_to_map(
            patch_scores,
            out_hw=out_hw,
            device=device,
            dtype=out_dtype,
            mode="rollout",
            topk=None,
        )


# ================================================================================================
# Gaze token embedding 
# ================================================================================================

class GazeTokenEmbedder(nn.Module):
    """
    Converts a gaze map M_g into patch-aligned gaze tokens G.

    Input:
      gaze_map: (B,H,W) or (B,1,H,W)
      grid_hw: (Gh,Gw)

    Output:
      gaze_tokens: (B, P, dg), where P = Gh*Gw
    """
    def __init__(self, gaze_token_dim: int) -> None:
        super().__init__()
        self.gaze_token_dim = int(gaze_token_dim)
        self.proj = nn.Sequential(
            nn.Linear(1, self.gaze_token_dim),
            nn.GELU(),
        )

    def forward(self, gaze_map: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
        gh, gw = int(grid_hw[0]), int(grid_hw[1])
        g = _ensure_gaze_4d(gaze_map).float()  # (B,1,H,W)
        g = F.interpolate(g, size=(gh, gw), mode="bilinear", align_corners=False)
        g = g.clamp_min(0.0)

        g_flat = g.flatten(2).transpose(1, 2).contiguous()  # (B,P,1)

        g_max = g_flat.amax(dim=1, keepdim=True).clamp_min(1e-12)  # (B,1,1)
        g_flat = g_flat / g_max

        return self.proj(g_flat)  # (B,P,dg)


# ================================================================================================
# GII injector with per-layer parameters
# ================================================================================================

@dataclass(frozen=True)
class GuideGuidanceConfig:
    """
    bottleneck_dim: d' in the paper
    gaze_hidden_dim: dg (gaze token embedding dim)
    drop_prob: stochastic gaze disabling during training (p in {0,1})
    strength: scale applied to injected residual
    """
    enabled: bool = False
    bottleneck_dim: int = 128
    gaze_hidden_dim: int = 64
    conv_hidden_channels: int = 64
    drop_prob: float = 0.0
    strength: float = 1.0


class GIIInjectorLayer(nn.Module):
    """
    Computes bar_Z_l from tilde_Z_l and gaze tokens, matching the paper layout:
      - down-projection (d -> d')
      - gaze projection into d'
      - GFF spatial attention via concat(avg,max) -> conv -> sigmoid
      - reweight visual tokens and up-projection (d' -> d)
    """
    def __init__(self, token_dim: int, gaze_token_dim: int, cfg: GuideGuidanceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(token_dim)
        dg = int(gaze_token_dim)
        db = int(cfg.bottleneck_dim)
        ch = int(cfg.conv_hidden_channels)

        self.down = nn.Sequential(
            nn.Linear(d, db),
            nn.GELU(),
        )

        self.gaze_proj = nn.Sequential(
            nn.Linear(dg, db),
            nn.GELU(),
        )

        # GFF input is concat(avg_pool, max_pool) -> 2 channels
        self.gff = nn.Sequential(
            nn.Conv2d(2, ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, 1, kernel_size=1, padding=0),
        )

        self.up = nn.Linear(db, d)

    def forward(
        self,
        z_tilde: torch.Tensor,
        gaze_tokens: torch.Tensor,
        p_mask: torch.Tensor,
        num_prefix_tokens: int,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        if not self.cfg.enabled:
            return z_tilde.new_zeros(z_tilde.shape)

        if z_tilde.ndim != 3:
            raise ValueError(f"Expected z_tilde (B,N,D), got {tuple(z_tilde.shape)}")

        b, n, _ = z_tilde.shape
        t = int(num_prefix_tokens)
        if n <= t:
            return z_tilde.new_zeros(z_tilde.shape)

        gh, gw = int(grid_hw[0]), int(grid_hw[1])
        p = n - t
        if gh * gw != p:
            g = int(math.isqrt(p))
            gh, gw = g, g

        gaze_tokens = gaze_tokens.to(device=z_tilde.device, dtype=z_tilde.dtype)

        z_hat = self.down(z_tilde)          # (B,N,db)
        z_hat_c = z_hat[:, :t, :]           # (B,T,db)
        z_hat_v = z_hat[:, t:, :]           # (B,P,db)

        g_hat = self.gaze_proj(gaze_tokens) # (B,P,db)

        # p_mask is (B,1,1) and broadcasts over (B,P,db)
        z_vg = z_hat_v + (p_mask * g_hat)   # (B,P,db)

        avg_pool = z_vg.mean(dim=-1, keepdim=True)        # (B,P,1)
        max_pool = z_vg.amax(dim=-1, keepdim=True)        # (B,P,1)

        f = torch.cat([avg_pool, max_pool], dim=-1)       # (B,P,2)
        f = f.view(b, gh, gw, 2).permute(0, 3, 1, 2)      # (B,2,gh,gw)

        a = torch.sigmoid(self.gff(f))                    # (B,1,gh,gw)
        a = a.flatten(2).transpose(1, 2).contiguous()     # (B,P,1)

        z_hat_v_prime = z_hat_v * a                       # (B,P,db)

        z_hat_prime = torch.cat([z_hat_c, z_hat_v_prime], dim=1)  # (B,N,db)
        z_bar = self.up(z_hat_prime)                              # (B,N,D)
        
        p_mask_ = p_mask.to(device=z_bar.device, dtype=z_bar.dtype)  # (B,1,1)
        z_bar = z_bar * p_mask_                                       # (B,N,D)
        
        return float(self.cfg.strength) * z_bar


# ================================================================================================
# Guided forward that injects inside each transformer block (A)
# ================================================================================================

def _resolve_drop_path(blk: nn.Module, which: int) -> Optional[nn.Module]:
    if which == 1:
        return getattr(blk, "drop_path1", getattr(blk, "drop_path", None))
    return getattr(blk, "drop_path2", getattr(blk, "drop_path", None))


def _maybe_layer_scale(blk: nn.Module, which: int, x: torch.Tensor) -> torch.Tensor:
    ls = getattr(blk, "ls1", None) if which == 1 else getattr(blk, "ls2", None)
    if isinstance(ls, nn.Module):
        return ls(x)
    return x


def _gaze_presence_mask(
    b: int,
    has_eye_mask: Optional[torch.Tensor],
    drop_prob: float,
    training: bool,
    device: torch.device,
) -> torch.Tensor:
    if has_eye_mask is None:
        p = torch.ones((b,), device=device, dtype=torch.float32)
    else:
        m = has_eye_mask.to(device=device, dtype=torch.bool)
        p = m.float()

    if training and (float(drop_prob) > 0.0):
        drop = (torch.rand((b,), device=device) < float(drop_prob)).float()
        p = p * (1.0 - drop)

    return p.view(b, 1, 1)  # (B,1,1)


def forward_backbone_tokens(
    backbone: nn.Module,
    x: torch.Tensor,
    attention_recorder: Optional[Any] = None,
    gaze_embedder: Optional["GazeTokenEmbedder"] = None,
    gii_layers: Optional[nn.ModuleList] = None,
    gaze_map: Optional[torch.Tensor] = None,
    has_eye_mask: Optional[torch.Tensor] = None,
    num_prefix_tokens: int = 1,
    guidance_drop_prob: float = 0.0,
) -> torch.Tensor:
    """
    Guidance implementation that preserves the backbone's native forward_features behavior.

    - If guidance is disabled or no gaze is present in the batch: uses backbone.forward_features/backbone(x)
    - If guidance is enabled and gaze exists: registers per-block forward hooks that add GII residuals
      to the block outputs, then runs backbone.forward_features/backbone(x) normally.
    """
    blocks = getattr(backbone, "blocks", None)

    # Determine whether any sample in the batch has gaze available
    has_any_gaze = True
    if has_eye_mask is not None:
        has_any_gaze = bool(has_eye_mask.to(torch.bool).any().item())

    guidance_enabled = (
        (gii_layers is not None)
        and (gaze_embedder is not None)
        and (gaze_map is not None)
        and (blocks is not None)
        and (len(gii_layers) > 0)
        and has_any_gaze
    )

    # Fallback: native backbone forward
    if not guidance_enabled:
        feats = backbone.forward_features(x) if hasattr(backbone, "forward_features") else backbone(x)
        return _normalize_backbone_output(feats)

    b = int(x.shape[0])

    # Patch grid inferred from backbone config; works with DINOv3 and other timm ViTs
    grid_hw = infer_patch_grid(backbone, num_patches=None)

    # Per-sample mask p in {0,1} with optional stochastic dropping during training
    p_mask = _gaze_presence_mask(
        b=b,
        has_eye_mask=has_eye_mask,
        drop_prob=float(guidance_drop_prob),
        training=bool(backbone.training),
        device=x.device,
    )

    # Embed gaze once for the whole forward
    gaze_tokens = gaze_embedder(gaze_map, grid_hw=grid_hw)  # (B,P,dg)

    hooks: List[Any] = []

    def _make_block_hook(layer_idx: int):
        def _hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
            # Some implementations may return tuples; only tensor outputs are supported for injection.
            if not torch.is_tensor(output):
                return output

            # GII expects tokens shaped (B,N,D). Use block output as z_tilde surrogate.
            z_tilde = output

            z_bar = gii_layers[layer_idx](
                z_tilde=z_tilde,
                gaze_tokens=gaze_tokens,
                p_mask=p_mask,
                num_prefix_tokens=int(num_prefix_tokens),
                grid_hw=grid_hw,
            )
            return output + z_bar
        return _hook

    # Attach hooks for the layers that have injectors
    n_hook = min(len(blocks), len(gii_layers))
    for i in range(n_hook):
        hooks.append(blocks[i].register_forward_hook(_make_block_hook(i)))

    try:
        feats = backbone.forward_features(x) if hasattr(backbone, "forward_features") else backbone(x)
    finally:
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

    return _normalize_backbone_output(feats)


# ================================================================================================
# Token pooling 
# ================================================================================================

def pool_tokens(
    feats: torch.Tensor,
    pooling: str,
    num_prefix_tokens: int,
    pool_k: int,
    apply_token_norm: bool = False,
    token_norm: Optional[nn.Module] = None,
) -> torch.Tensor:
    pooling = str(pooling).lower().strip()
    t_pref = int(num_prefix_tokens)

    if feats.ndim == 2:
        pooled = feats
        if pooling in ("concat", "cls_reg_concat", "cls_max_concat"):
            pooled = torch.cat([pooled, pooled], dim=-1)
        return pooled

    if feats.ndim != 3:
        raise ValueError(f"Unexpected backbone output shape: {tuple(feats.shape)}")

    tokens = feats
    if apply_token_norm and (token_norm is not None):
        try:
            tokens = token_norm(tokens)
        except Exception:
            pass

    prefix = tokens[:, :t_pref, :]
    patches = tokens[:, t_pref:, :]

    cls = prefix[:, 0, :] if prefix.shape[1] >= 1 else tokens[:, 0, :]

    has_regs = (prefix.shape[1] > 1)
    regs = prefix[:, 1:, :] if has_regs else None

    has_patches = (patches.shape[1] > 0)
    patch_mean = patches.mean(dim=1) if has_patches else cls
    reg_mean = regs.mean(dim=1) if has_regs else cls

    if pooling == "cls":
        pooled = cls
    elif pooling == "max":
        pooled = patches.max(dim=1).values if has_patches else cls
    elif pooling == "cls_max_concat":
        patch_max = patches.max(dim=1).values if has_patches else cls
        pooled = torch.cat([cls, patch_max], dim=-1)
    elif pooling in ("mean", "patch_mean"):
        pooled = patch_mean
    elif pooling == "reg_mean":
        pooled = reg_mean
    elif pooling == "prefix_mean":
        pooled = prefix.mean(dim=1) if prefix.shape[1] > 0 else cls
    elif pooling == "cls_reg_concat":
        pooled = torch.cat([cls, reg_mean], dim=-1)
    elif pooling == "cls_reg_add":
        pooled = cls + reg_mean
    elif pooling == "concat":
        pooled = torch.cat([cls, patch_mean], dim=-1)
    elif pooling == "topk":
        if not has_patches:
            pooled = cls
        else:
            k = max(1, min(int(pool_k), int(patches.shape[1])))
            norms = patches.norm(dim=-1)
            idx = norms.topk(k, dim=1).indices
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
            selected = torch.gather(patches, dim=1, index=idx_exp)
            pooled = selected.mean(dim=1)
    else:
        raise ValueError(f"Unknown pooling mode: {pooling}")

    return pooled
