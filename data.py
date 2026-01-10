import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance, ImageDraw
import numpy as np  
from timeit import default_timer as timer
import re
from torchvision.transforms import functional as TF
import random, math
import torchvision.transforms.functional as F
import torch
import torchvision.transforms as transforms
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm
from typing import Optional, Tuple
import torch.nn.functional as nnF




class ComparisonsDataset(Dataset):
    """
    Pairwise image-comparison dataset with optional eyetracker gaze supervision.

    Design goals
    ------------
    - Keep the dataset "raw": return images as PIL and gaze as tensors in their native resolution.
    - Avoid hardcoding any gaze grid size (e.g., 14x14). Backbone-dependent grid sizing belongs in transforms.
    - Guarantee DataLoader collation stability by returning a sentinel dummy gaze tensor [1,1,1] when gaze is absent.
      Transforms can deterministically replace this dummy with [1, grid_h, grid_w] if needed.
    - Enforce "both sides or none" semantics for gaze supervision via `has_eyetracker` mask (ok_l AND ok_r).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        root_dir: str,
        transform=None,
        logger=None,
        gaze_root: Optional[str] = None,
        use_gaze: bool = True,
        use_seg: bool = False,
        map_size: int = 14,
        gaze_subdir_fmt: str = "{s}x{s}",
    ):
        """
        Args:
            dataframe: DataFrame with at least columns:
                - image_l, image_r, score, score_classification
                Optionally:
                - dataset (subfolder)
                - has_eyetracker (bool/int)
                - npy_file_l, npy_file_r (paths or filenames)
            root_dir: Base directory containing images (optionally with dataset/city subfolders).
            transform: Callable applied to the sample dict after loading (responsible for resize/crop/normalize).
            logger: Optional logger for profiling.
            gaze_root: Base directory for gaze npy files if paths are relative.
            use_gaze: Global toggle to enable/disable gaze loading.
            use_seg: If True, replaces ".jpg" with "_seg.jpg" and requires those files to exist.
            map_size: Default spatial dimension used for dummy gaze tensors (e.g., 14 -> [1,14,14]).
        """
        self.comparisons_frame = dataframe.reset_index(drop=True)
        self.root_dir = str(root_dir)
        self.transform = transform
        self.logger = logger
        self.gaze_root = gaze_root
        self.use_gaze = bool(use_gaze)
        self.use_seg = bool(use_seg)
    
        # Which on-disk gaze resolution folder to use (e.g., 14x14, 16x16)
        self.map_size = int(map_size)
        self.gaze_subdir_fmt = str(gaze_subdir_fmt)
    
        # Collation-safe dummy gaze. Keep it minimal; transforms will expand/resize as needed.
        # Using 1x1 avoids silently “pretending” gaze is 14x14 when it might not be.
        self._gaze_dummy = torch.zeros((1, 1, 1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.comparisons_frame)

    @staticmethod
    def _load_image(path: str) -> Image.Image:
        """Loads an RGB image as PIL."""
        return Image.open(path).convert("RGB")

    def _resolve_gaze_path(self, fname: Optional[str]) -> str:
        """
        Resolves gaze path.
    
        If fname is relative and gaze_root is provided, we look under:
            gaze_root / <subdir_for_map_size> / fname
        where <subdir_for_map_size> defaults to "{s}x{s}" -> "14x14", "16x16", ...
    
        If fname is absolute, keep it.
        """
        if not fname:
            return ""
    
        # Absolute paths bypass gaze_root/subdir logic
        if os.path.isabs(fname):
            return fname
    
        if self.gaze_root is None:
            return fname  # relative but no root given
    
        subdir = self.gaze_subdir_fmt.format(s=self.map_size)
        return os.path.join(self.gaze_root, subdir, fname)


    def _load_gaze_npy(self, fname: Optional[str]) -> Tuple[torch.Tensor, bool]:
        """
        Loads gaze heatmap from .npy and returns (tensor [C,H,W], found_flag).

        Supported formats
        -----------------
        - [H,W]     -> converted to [1,H,W]
        - [C,H,W]   -> kept as-is

        Failure policy
        --------------
        - Missing/invalid files return dummy [1,map_size,map_size] and found_flag=False.
        - No attempt to resize to a fixed grid; transforms should handle backbone-dependent resizing.
        """
        full_path = self._resolve_gaze_path(fname)
        if not full_path or (not os.path.exists(full_path)):
            return self._gaze_dummy.clone(), False

        try:
            arr = np.load(full_path)
            t = torch.from_numpy(arr).float()

            if t.ndim == 2:
                t = t.unsqueeze(0)  # [1,H,W]
            elif t.ndim == 3:
                # [C,H,W] as-is
                pass
            else:
                return self._gaze_dummy.clone(), False

            # Ensure contiguous for downstream resize ops
            t = t.contiguous()
            return t, True
        except Exception:
            return self._gaze_dummy.clone(), False

    def _build_image_paths(self, row: pd.Series) -> Tuple[str, str]:
        """
        Builds left/right image paths with optional dataset subfolder and optional *_seg.jpg replacement.
        """
        city = None
        if "dataset" in row.index and pd.notna(row["dataset"]):
            city = str(row["dataset"])

        if city:
            img_l = os.path.join(self.root_dir, city, str(row["image_l"]))
            img_r = os.path.join(self.root_dir, city, str(row["image_r"]))
        else:
            img_l = os.path.join(self.root_dir, str(row["image_l"]))
            img_r = os.path.join(self.root_dir, str(row["image_r"]))

        if self.use_seg:
            img_l_seg = re.sub(r"(?i)\.jpg$", "_seg.jpg", img_l)
            img_r_seg = re.sub(r"(?i)\.jpg$", "_seg.jpg", img_r)

            if not os.path.exists(img_l_seg):
                raise FileNotFoundError(f"[--use_seg] Segmented file missing: {img_l_seg}")
            if not os.path.exists(img_r_seg):
                raise FileNotFoundError(f"[--use_seg] Segmented file missing: {img_r_seg}")

            img_l, img_r = img_l_seg, img_r_seg

        return img_l, img_r

    def __getitem__(self, idx: int) -> dict:
        start = timer()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.comparisons_frame.iloc[int(idx)]

        # ---------------------------------------------------------------------
        # 1) Images
        # ---------------------------------------------------------------------
        img_l_path, img_r_path = self._build_image_paths(row)
        image_l = self._load_image(img_l_path)
        image_r = self._load_image(img_r_path)

        # ---------------------------------------------------------------------
        # 2) Labels
        # ---------------------------------------------------------------------
        score_r = int(row["score"])
        score_c = int(row["score_classification"])

        # ---------------------------------------------------------------------
        # 3) Gaze flags and paths
        # ---------------------------------------------------------------------
        has_eye_flag = False
        if self.use_gaze and ("has_eyetracker" in row.index) and pd.notna(row["has_eyetracker"]):
            try:
                has_eye_flag = bool(int(row["has_eyetracker"]))
            except Exception:
                has_eye_flag = bool(row["has_eyetracker"])

        if not self.use_gaze:
            has_eye_flag = False

        gaze_file_l = row["npy_file_l"] if ("npy_file_l" in row.index and pd.notna(row["npy_file_l"])) else None
        gaze_file_r = row["npy_file_r"] if ("npy_file_r" in row.index and pd.notna(row["npy_file_r"])) else None

        if has_eye_flag:
            gaze_l, ok_l = self._load_gaze_npy(gaze_file_l)
            gaze_r, ok_r = self._load_gaze_npy(gaze_file_r)
        else:
            gaze_l, ok_l = self._gaze_dummy.clone(), False
            gaze_r, ok_r = self._gaze_dummy.clone(), False
            

        # Only enable gaze supervision when both sides contain valid gaze
        has_eye_tensor = torch.tensor(bool(ok_l and ok_r), dtype=torch.bool)

        # ---------------------------------------------------------------------
        # 4) Sample dict
        # ---------------------------------------------------------------------
        sample = {
            "image_l": image_l,
            "image_r": image_r,
            "score_r": score_r,
            "score_c": score_c,
            "image_l_name": img_l_path,
            "image_r_name": img_r_path,
            "has_eyetracker": has_eye_tensor,
            "gaze_l": gaze_l,
            "gaze_r": gaze_r,
            "gaze_ok_l": torch.tensor(ok_l, dtype=torch.bool),
            "gaze_ok_r": torch.tensor(ok_r, dtype=torch.bool),
            "gaze_path_l": str(self._resolve_gaze_path(gaze_file_l)) if gaze_file_l else "",
            "gaze_path_r": str(self._resolve_gaze_path(gaze_file_r)) if gaze_file_r else "",

        }

        # ---------------------------------------------------------------------
        # 5) Transforms (responsible for resizing/cropping/normalizing images and
        #    aligning gaze to backbone-specific grid when needed)
        # ---------------------------------------------------------------------
        if self.transform is not None:
            sample = self.transform(sample)

        end = timer()
        if self.logger:
            self.logger.info(f"DATALOADER, {end - start:.4f}")

        return sample


class ResizeCenterCropAlignGaze:
    """
    Deterministic preprocessing:
      - Resize images by short side (aspect ratio preserved)
      - Center crop images to a fixed square
      - If enable_gaze=True:
          - if has_eyetracker=True: resize + crop gaze to match image geometry, then downsample to gaze_grid_size
          - else: create a fixed-shape dummy gaze tensor for collation safety
      - If enable_gaze=False: do not create, modify, or delete any gaze keys
    """

    def __init__(
        self,
        resize_dim: int,
        target_crop: int,
        img_interp,
        gaze_grid_size,
        enable_gaze: bool = True,
    ):
        self.resize_dim = int(resize_dim)
        self.target_crop = int(target_crop)
        self.img_interp = img_interp
        self.gaze_grid_size = (int(gaze_grid_size[0]), int(gaze_grid_size[1]))
        self.enable_gaze = bool(enable_gaze)

    @staticmethod
    def _as_bool(x) -> bool:
        if torch.is_tensor(x):
            return bool(x.item())
        return bool(x)

    @staticmethod
    def _to_1ch_float(g) -> torch.Tensor:
        # Accept [H,W], [1,H,W], [C,H,W]; return contiguous [1,H,W] float32
        if not torch.is_tensor(g):
            g = torch.as_tensor(g)
        g = g.float()

        if g.ndim == 2:
            g = g.unsqueeze(0)  # [1,H,W]
        elif g.ndim == 3:
            if g.shape[0] != 1:
                g = g.mean(dim=0, keepdim=True)  # [1,H,W]
        else:
            raise ValueError(f"Unexpected gaze shape: {tuple(g.shape)}")

        return g.contiguous()

    @staticmethod
    def _resize_gaze(g_1chw: torch.Tensor, size_hw) -> torch.Tensor:
        # g_1chw: [1,H,W] -> [1,H2,W2] using bilinear interpolation
        if g_1chw.ndim != 3 or g_1chw.shape[0] != 1:
            raise ValueError(f"Expected gaze [1,H,W], got {tuple(g_1chw.shape)}")

        x = g_1chw.unsqueeze(0)  # [1,1,H,W]
        x = nnF.interpolate(x, size=tuple(size_hw), mode="bilinear", align_corners=False)
        return x.squeeze(0).contiguous()  # [1,H2,W2]

    def _process_side(self, sample: dict, side: str) -> None:
        img_key = f"image_{side}"
        gaze_key = f"gaze_{side}"

        # 1) Resize image by short-side policy
        img = TF.resize(sample[img_key], self.resize_dim, interpolation=self.img_interp)

        # Ensure center-crop is feasible
        w, h = img.size  # PIL: (W,H)
        if min(w, h) < self.target_crop:
            img = TF.resize(img, self.target_crop, interpolation=self.img_interp)
            w, h = img.size

        # 2) Center-crop image deterministically
        img = TF.center_crop(img, [self.target_crop, self.target_crop])
        sample[img_key] = img

        # 3) If gaze is disabled, leave gaze keys untouched
        if not self.enable_gaze:
            return

        has_real_gaze = self._as_bool(sample.get("has_eyetracker", False))
        if not has_real_gaze:
            sample[gaze_key] = torch.zeros((1, *self.gaze_grid_size), dtype=torch.float32)
            return

        # 4) Align gaze to resized image geometry, then crop and downsample
        g = self._to_1ch_float(sample[gaze_key])
        g = self._resize_gaze(g, size_hw=(h, w))  # match resized image H,W
        g = TF.center_crop(g, [self.target_crop, self.target_crop])
        g = self._resize_gaze(g, size_hw=self.gaze_grid_size)  # supervision grid
        sample[gaze_key] = g

    def __call__(self, sample: dict) -> dict:
        self._process_side(sample, "l")
        self._process_side(sample, "r")
        return sample


def _get_interp_mode(mode_str):
    if mode_str is None:
        raise ValueError("Interpolation mode is None; backbone config may be incomplete.")

    mode_str = str(mode_str).lower()
    mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
    }

    if mode_str not in mapping:
        raise ValueError(f"Unknown interpolation mode '{mode_str}'")

    return mapping[mode_str]

def build_eval_transforms(specs: dict, gaze_grid_size=(14, 14), enable_gaze: bool = True):
    """
    Build deterministic validation/test preprocessing.

    Steps:
      - Resize by short side (preserve aspect ratio), then center-crop to the model input size
      - If enabled and eyetracker data is present, apply the same geometric ops to gaze and downsample it
      - Convert left/right images to tensors and normalize with backbone stats
    """
    target_crop = int(specs["input_size"][-1])
    crop_pct = float(specs["crop_pct"])

    # timm-style eval resize: input_size / crop_pct, clamped to >= input_size
    resize_dim = max(target_crop, int(round(target_crop / crop_pct)))

    img_interp = _get_interp_mode(specs["interpolation"])
    mean = tuple(specs["mean"])
    std = tuple(specs["std"])

    def _to_tensor_and_normalize_pair(sample: dict) -> dict:
        # Apply the same deterministic post-processing to both image views
        for k in ("image_l", "image_r"):
            if k in sample:
                x = TF.to_tensor(sample[k])
                sample[k] = TF.normalize(x, mean=mean, std=std)
        return sample

    eval_tfms = transforms.Compose(
        [
            ResizeCenterCropAlignGaze(
                resize_dim=resize_dim,
                target_crop=target_crop,
                img_interp=img_interp,
                gaze_grid_size=gaze_grid_size,
                enable_gaze=bool(enable_gaze),
            ),
            transforms.Lambda(_to_tensor_and_normalize_pair),
        ]
    )

    meta = {
        "target_crop": target_crop,
        "resize_dim": resize_dim,
        "crop_pct": crop_pct,
        "interpolation": str(specs.get("interpolation", "bilinear")),
        "mean": mean,
        "std": std,
        "gaze_grid_size": tuple(gaze_grid_size),
        "enable_gaze": bool(enable_gaze),
        "eval_policy": (
            "Resize(short,img) -> CenterCrop(img); "
            "if enable_gaze: Resize(match,gaze)->CenterCrop(gaze)->ResizeDown(gaze); "
            "ToTensor/Norm(img)"
        ),
    }
    return eval_tfms, meta

# =============================================================================================== #
# Augmentation presets (per-op paired toggles)
# =============================================================================================== #
# Semantics:
#   paired_* = True  -> same random decision/params for left and right
#   paired_* = False -> independent random decision/params for left and right
#
# Notes for pairwise ranking:
#   - Zoom-out is disabled (scale_range min must be >= 1.0) to avoid padding/black bars.
#   - Swap is inherently paired and should not have a paired_* toggle.

AUG_PRESETS = {
    "light": dict(
        # Pairing toggles
        paired_scale=True,
        paired_hflip=True,
        paired_crop=True,
        paired_rotation=True,
        paired_color_jitter=False,
        paired_gray=False,
        paired_erase=True,

        # Params
        hflip_p=0.50,
        swap_p=0.50,

        crop_p=0.00,

        scale_p=0.00,
        scale_range=(1.00, 1.00),

        rotation_p=0.00,
        max_rotation_deg=0.0,

        color_jitter_p=0.00,
        jitter_brightness=0.0,
        jitter_contrast=0.0,
        jitter_saturation=0.0,
        jitter_hue=0.0,

        gray_p=0.00,

        erase_p=0.00,
        erase_scale=(0.02, 0.06),
        erase_ratio=(0.3, 3.3),
        erase_value=0.0,
    ),

    "heavy": dict(
        # Pairing toggles
        paired_scale=False,
        paired_hflip=False,
        paired_crop=False,
        paired_rotation=False,
        paired_color_jitter=False,
        paired_gray=False,
        paired_erase=False,

        # Params (label-stable, ViT-safe)
        hflip_p=0.5,
        swap_p=0.5,

        # Paired translation jitter
        crop_p=1,

        scale_p=0.1,
        #scale_p=0.5,
        scale_range=(1, 1.3),

        # Small, infrequent rotation
        rotation_p=0.25,
        max_rotation_deg=4,

        # Unpaired photometric jitter (moderate)
        color_jitter_p=0.2,
        jitter_brightness=0.40,
        jitter_contrast=0.40,
        jitter_saturation=0.45,
        jitter_hue=0.10,

        gray_p=0.05,

        # Keep erasing off by default (enable only after baseline is stable)
        erase_p=0.05,
        erase_scale=(0.03, 0.1),
        erase_ratio=(0.3, 3.3),
        erase_value=0.0,
    ),
}

# ==============================================================================
# Augmentation Class
# ==============================================================================

class Augmentation:
    """
    Pairwise augmentation pipeline for Siamese / ranking tasks.

    Probability semantics:
      - If paired_* is True: the transform is applied to BOTH images with probability p.
      - If paired_* is False: the transform is applied to EACH image independently with probability p.
        (p=1.0 implies each image always receives the transform; p=0.0 implies never.)

    Geometry:
      - Scale is zoom-in only (scale_min clamped to >= 1.0).
      - Crop always outputs out_size; when crop is not applied to a side, center-crop is used for that side.

    Photometric:
      - Unpaired photometric transforms sample independent parameters per image.

    Swap:
      - Always a pair operation, updates labels deterministically.
    """

    def __init__(
        self,
        augment: bool,
        ties: bool,
        resize_short: int,
        out_size: int,
        interpolation,
        mean: tuple,
        std: tuple,

        # Probabilities & ranges
        hflip_p: float = 0.5,
        swap_p: float = 0.5,

        crop_p: float = 0.0,

        scale_p: float = 0.0,
        scale_range=(1.0, 1.0),
        scale_fill: float = 0.0,

        rotation_p: float = 0.0,
        max_rotation_deg: float = 0.0,

        color_jitter_p: float = 0.0,
        jitter_brightness: float = 0.0,
        jitter_contrast: float = 0.0,
        jitter_saturation: float = 0.0,
        jitter_hue: float = 0.0,

        gray_p: float = 0.0,

        erase_p: float = 0.0,
        erase_scale=(0.02, 0.20),
        erase_ratio=(0.3, 3.3),
        erase_value: float = 0.0,

        # Pairing config
        paired_scale: bool = True,
        paired_hflip: bool = True,
        paired_crop: bool = True,
        paired_rotation: bool = True,
        paired_color_jitter: bool = False,
        paired_gray: bool = False,
        paired_erase: bool = True,
    ):
        self.augment = bool(augment)
        self.ties = bool(ties)

        self.resize_short = int(resize_short)
        self.out_size = int(out_size)
        self.interpolation = interpolation

        self.mean = mean
        self.std = std

        self.hflip_p = float(hflip_p)
        self.swap_p = float(swap_p)

        self.crop_p = float(crop_p)

        self.scale_p = float(scale_p)
        s0, s1 = float(scale_range[0]), float(scale_range[1])
        self.scale_min = max(1.0, min(s0, s1))  # forbid zoom-out
        self.scale_max = max(1.0, max(s0, s1))
        self.scale_fill = float(scale_fill)

        self.rotation_p = float(rotation_p)
        self.max_rotation_deg = float(max_rotation_deg)

        self.color_jitter_p = float(color_jitter_p)
        self.jitter_brightness = float(jitter_brightness)
        self.jitter_contrast = float(jitter_contrast)
        self.jitter_saturation = float(jitter_saturation)
        self.jitter_hue = float(jitter_hue)

        self.gray_p = float(gray_p)

        self.erase_p = float(erase_p)
        self.erase_scale_min, self.erase_scale_max = float(erase_scale[0]), float(erase_scale[1])
        self.erase_ratio_min, self.erase_ratio_max = float(erase_ratio[0]), float(erase_ratio[1])
        self.erase_value = float(erase_value)

        self.paired_scale = bool(paired_scale)
        self.paired_hflip = bool(paired_hflip)
        self.paired_crop = bool(paired_crop)
        self.paired_rotation = bool(paired_rotation)
        self.paired_color_jitter = bool(paired_color_jitter)
        self.paired_gray = bool(paired_gray)
        self.paired_erase = bool(paired_erase)

    # ------------------------------------------------------------------
    # RNG helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coin(p: float) -> bool:
        p = float(p)
        if p <= 0.0:
            return False
        if p >= 1.0:
            return True
        return random.random() < p

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def _score_r_to_score_c(self, score_r: int) -> int:
        score_r = int(score_r)
        if self.ties:
            return score_r + 1  # {-1,0,+1} -> {0,1,2}
        return 0 if score_r < 0 else 1

    # ------------------------------------------------------------------
    # PIL helpers
    # ------------------------------------------------------------------

    def _resize_short_side(self, img):
        return TF.resize(img, self.resize_short, interpolation=self.interpolation)

    @staticmethod
    def _center_crop_to_common(img_l, img_r):
        wl, hl = img_l.size
        wr, hr = img_r.size
        w = min(wl, wr)
        h = min(hl, hr)
        if (wl, hl) != (w, h):
            img_l = TF.center_crop(img_l, [h, w])
        if (wr, hr) != (w, h):
            img_r = TF.center_crop(img_r, [h, w])
        return img_l, img_r

    def _ensure_min_size_pair(self, img_l, img_r, th, tw):
        w, h = img_l.size
        if w >= tw and h >= th:
            return img_l, img_r
        new_short = max(self.resize_short, th, tw)
        img_l = TF.resize(img_l, new_short, interpolation=self.interpolation)
        img_r = TF.resize(img_r, new_short, interpolation=self.interpolation)
        return self._center_crop_to_common(img_l, img_r)

    @staticmethod
    def _sample_crop_coords(w, h, th, tw):
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j

    def _apply_scale(self, img, s: float):
        w, h = img.size
        cx = w // 2
        cy = h // 2
        return TF.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=float(s),
            shear=[0.0, 0.0],
            interpolation=self.interpolation,
            fill=self.scale_fill,
            center=[cx, cy],
        )

    def _apply_rotate(self, img, angle: float):
        return TF.rotate(
            img,
            angle=float(angle),
            interpolation=self.interpolation,
            expand=False,
            fill=self.scale_fill,
        )

    def _sample_color_jitter_ops(self):
        def sample_factor(a: float):
            a = float(a)
            if a <= 0.0:
                return None
            lo = max(0.0, 1.0 - a)
            hi = 1.0 + a
            return random.uniform(lo, hi)

        def sample_hue(a: float):
            a = float(a)
            if a <= 0.0:
                return None
            a = min(0.5, a)
            return random.uniform(-a, a)

        b = sample_factor(self.jitter_brightness)
        c = sample_factor(self.jitter_contrast)
        s = sample_factor(self.jitter_saturation)
        h = sample_hue(self.jitter_hue)

        ops = []
        if b is not None:
            ops.append(("b", b))
        if c is not None:
            ops.append(("c", c))
        if s is not None:
            ops.append(("s", s))
        if h is not None:
            ops.append(("h", h))
        random.shuffle(ops)
        return ops

    @staticmethod
    def _apply_color_jitter_ops(img, ops):
        for k, v in ops:
            if k == "b":
                img = TF.adjust_brightness(img, v)
            elif k == "c":
                img = TF.adjust_contrast(img, v)
            elif k == "s":
                img = TF.adjust_saturation(img, v)
            elif k == "h":
                img = TF.adjust_hue(img, v)
        return img

    # ------------------------------------------------------------------
    # Tensor helpers (erase)
    # ------------------------------------------------------------------

    def _sample_erasing_rect(self, H, W, max_tries=10):
        area = H * W
        for _ in range(max_tries):
            erase_area = area * random.uniform(self.erase_scale_min, self.erase_scale_max)
            aspect = random.uniform(self.erase_ratio_min, self.erase_ratio_max)
            eh = int(round(math.sqrt(erase_area * aspect)))
            ew = int(round(math.sqrt(erase_area / aspect)))
            if 0 < eh < H and 0 < ew < W:
                top = random.randint(0, H - eh)
                left = random.randint(0, W - ew)
                return top, left, eh, ew
        return None

    def _apply_erase_rect(self, x, rect):
        top, left, eh, ew = rect
        x[:, top : top + eh, left : left + ew] = self.erase_value
        return x

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def __call__(self, sample: dict) -> dict:
        # Gaze safety check
        has_eye = sample.get("has_eyetracker", False)
        if torch.is_tensor(has_eye):
            has_eye = bool(has_eye.item())
        else:
            has_eye = bool(has_eye)
        if has_eye:
            raise RuntimeError("Augmentation is not supported when gaze alignment is active.")

        img_l = sample["image_l"]
        img_r = sample["image_r"]

        score_r = int(sample["score_r"])
        score_c = int(sample.get("score_c", self._score_r_to_score_c(score_r)))

        do_aug = bool(self.augment)
        pil_inputs = (not torch.is_tensor(img_l)) and (not torch.is_tensor(img_r))

        # -------------------------
        # PIL branch
        # -------------------------
        if pil_inputs:
            img_l = self._resize_short_side(img_l)
            img_r = self._resize_short_side(img_r)
            img_l, img_r = self._center_crop_to_common(img_l, img_r)

            # Scale jitter (zoom-in only)
            if do_aug and (self.scale_p > 0.0):
                if self.paired_scale:
                    if self._coin(self.scale_p):
                        s = random.uniform(self.scale_min, self.scale_max)
                        img_l = self._apply_scale(img_l, s)
                        img_r = self._apply_scale(img_r, s)
                else:
                    if self._coin(self.scale_p):
                        img_l = self._apply_scale(img_l, random.uniform(self.scale_min, self.scale_max))
                    if self._coin(self.scale_p):
                        img_r = self._apply_scale(img_r, random.uniform(self.scale_min, self.scale_max))
                img_l, img_r = self._center_crop_to_common(img_l, img_r)

            # Horizontal flip
            if do_aug and (self.hflip_p > 0.0):
                if self.paired_hflip:
                    if self._coin(self.hflip_p):
                        img_l = TF.hflip(img_l)
                        img_r = TF.hflip(img_r)
                else:
                    if self._coin(self.hflip_p):
                        img_l = TF.hflip(img_l)
                    if self._coin(self.hflip_p):
                        img_r = TF.hflip(img_r)

            # Rotation
            if do_aug and (self.rotation_p > 0.0) and (self.max_rotation_deg > 0.0):
                if self.paired_rotation:
                    if self._coin(self.rotation_p):
                        ang = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
                        img_l = self._apply_rotate(img_l, ang)
                        img_r = self._apply_rotate(img_r, ang)
                else:
                    if self._coin(self.rotation_p):
                        img_l = self._apply_rotate(img_l, random.uniform(-self.max_rotation_deg, self.max_rotation_deg))
                    if self._coin(self.rotation_p):
                        img_r = self._apply_rotate(img_r, random.uniform(-self.max_rotation_deg, self.max_rotation_deg))
                img_l, img_r = self._center_crop_to_common(img_l, img_r)

            # Crop to output size
            th = tw = self.out_size
            img_l, img_r = self._ensure_min_size_pair(img_l, img_r, th, tw)
            w, h = img_l.size

            if do_aug and (self.crop_p > 0.0):
                if self.paired_crop:
                    if self._coin(self.crop_p):
                        i, j = self._sample_crop_coords(w, h, th, tw)
                        img_l = TF.crop(img_l, i, j, th, tw)
                        img_r = TF.crop(img_r, i, j, th, tw)
                    else:
                        img_l = TF.center_crop(img_l, [th, tw])
                        img_r = TF.center_crop(img_r, [th, tw])
                else:
                    if self._coin(self.crop_p):
                        i, j = self._sample_crop_coords(w, h, th, tw)
                        img_l = TF.crop(img_l, i, j, th, tw)
                    else:
                        img_l = TF.center_crop(img_l, [th, tw])

                    if self._coin(self.crop_p):
                        i, j = self._sample_crop_coords(w, h, th, tw)
                        img_r = TF.crop(img_r, i, j, th, tw)
                    else:
                        img_r = TF.center_crop(img_r, [th, tw])
            else:
                img_l = TF.center_crop(img_l, [th, tw])
                img_r = TF.center_crop(img_r, [th, tw])

            # Color jitter
            if do_aug and (self.color_jitter_p > 0.0):
                if self.paired_color_jitter:
                    if self._coin(self.color_jitter_p):
                        ops = self._sample_color_jitter_ops()
                        img_l = self._apply_color_jitter_ops(img_l, ops)
                        img_r = self._apply_color_jitter_ops(img_r, ops)
                else:
                    if self._coin(self.color_jitter_p):
                        img_l = self._apply_color_jitter_ops(img_l, self._sample_color_jitter_ops())
                    if self._coin(self.color_jitter_p):
                        img_r = self._apply_color_jitter_ops(img_r, self._sample_color_jitter_ops())

            # Grayscale
            if do_aug and (self.gray_p > 0.0):
                if self.paired_gray:
                    if self._coin(self.gray_p):
                        img_l = TF.rgb_to_grayscale(img_l, num_output_channels=3)
                        img_r = TF.rgb_to_grayscale(img_r, num_output_channels=3)
                else:
                    if self._coin(self.gray_p):
                        img_l = TF.rgb_to_grayscale(img_l, num_output_channels=3)
                    if self._coin(self.gray_p):
                        img_r = TF.rgb_to_grayscale(img_r, num_output_channels=3)

            # Swap (pair operation + label update)
            if do_aug and (self.swap_p > 0.0) and self._coin(self.swap_p):
                img_l, img_r = img_r, img_l
                score_r = -score_r
                score_c = self._score_r_to_score_c(score_r)

            x_l = TF.to_tensor(img_l)
            x_r = TF.to_tensor(img_r)

        # -------------------------
        # Tensor branch
        # -------------------------
        else:
            x_l = img_l
            x_r = img_r

        # Random erasing (tensor level)
        if do_aug and (self.erase_p > 0.0):
            if self.paired_erase:
                if self._coin(self.erase_p):
                    _, H, W = x_l.shape
                    rect = self._sample_erasing_rect(H, W)
                    if rect is not None:
                        x_l = self._apply_erase_rect(x_l, rect)
                        x_r = self._apply_erase_rect(x_r, rect)
            else:
                if self._coin(self.erase_p):
                    _, H, W = x_l.shape
                    rect = self._sample_erasing_rect(H, W)
                    if rect is not None:
                        x_l = self._apply_erase_rect(x_l, rect)

                if self._coin(self.erase_p):
                    _, H, W = x_r.shape
                    rect = self._sample_erasing_rect(H, W)
                    if rect is not None:
                        x_r = self._apply_erase_rect(x_r, rect)

        # Normalize
        x_l = TF.normalize(x_l, mean=self.mean, std=self.std)
        x_r = TF.normalize(x_r, mean=self.mean, std=self.std)

        # Finalize sample
        sample["image_l"] = x_l
        sample["image_r"] = x_r
        sample["score_r"] = torch.tensor(int(score_r), dtype=torch.long)
        sample["score_c"] = torch.tensor(int(score_c), dtype=torch.long)
        return sample


        