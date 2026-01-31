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

        If fname is absolute, keep it.
        If fname is relative and gaze_root is provided:
            gaze_root / fname
        """
        if not fname:
            return ""

        if os.path.isabs(fname):
            return fname

        if self.gaze_root is None:
            return fname

        return os.path.join(self.gaze_root, fname)


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
          - if has_eyetracker=True: resize + crop gaze to match image geometry
            - gaze_output="align": downsample to gaze_grid_size
            - gaze_output="guide": keep at target_crop x target_crop
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
        gaze_output: str = "align",  # "align" or "guide"
    ):
        self.resize_dim = int(resize_dim)
        self.target_crop = int(target_crop)
        self.img_interp = img_interp
        self.gaze_grid_size = (int(gaze_grid_size[0]), int(gaze_grid_size[1]))
        self.enable_gaze = bool(enable_gaze)

        gaze_output = str(gaze_output).strip().lower()
        if gaze_output not in ("align", "guide"):
            raise ValueError(f"gaze_output must be 'align' or 'guide', got: {gaze_output}")
        self.gaze_output = gaze_output

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

    def _gaze_out_hw(self) -> Tuple[int, int]:
        if self.gaze_output == "align":
            return self.gaze_grid_size
        return (self.target_crop, self.target_crop)

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

        out_hw = self._gaze_out_hw()

        has_real_gaze = self._as_bool(sample.get("has_eyetracker", False))
        if not has_real_gaze:
            sample[gaze_key] = torch.zeros((1, out_hw[0], out_hw[1]), dtype=torch.float32)
            return

        # 4) Align gaze to resized image geometry, then crop; optional final resize depends on mode
        g = self._to_1ch_float(sample[gaze_key])
        g = self._resize_gaze(g, size_hw=(h, w))  # match resized image H,W
        g = TF.center_crop(g, [self.target_crop, self.target_crop])

        if self.gaze_output == "align":
            g = self._resize_gaze(g, size_hw=out_hw)  # supervision grid

        sample[gaze_key] = g

    def __call__(self, sample: dict) -> dict:
        self._process_side(sample, "l")
        self._process_side(sample, "r")
        return sample




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
    "light": {
        "paired_scale": True,
        "paired_hflip": True,
        "paired_crop": True,
        "paired_rotation": True,
        "paired_color_jitter": False,
        "paired_gray": False,
        "paired_erase": True,

        "swap_p": 0.50,
        "hflip_p": 0.20,
        "rot_deg": 0.0,
        "rot_p": 0.0,

        "crop_scale": (1.00, 1.00),
        "crop_ratio": (1.00, 1.00),

        "color_jitter": None,
        "gray_p": 0.0,
        "blur_p": 0.0,
        "blur_kernel": 23,
        "blur_sigma": (0.1, 2.0),

        "erase_p": 0.0,
        "erase_scale": (0.02, 0.06),
        "erase_ratio": (0.3, 3.3),
    },

    "heavy": {
        "paired_scale": True,
        "paired_hflip": True,
        "paired_crop": True,
        "paired_rotation": True,
        "paired_color_jitter": True,
        "paired_gray": True,
        "paired_erase": True,

        "swap_p": 0.50,
        "hflip_p": 0.30,
        "rot_deg": 3.0,
        "rot_p": 0.10,

        "crop_scale": (0.80, 1.00),
        "crop_ratio": (0.75, 1.3333333333333333),

        "color_jitter": (0.20, 0.20, 0.20, 0.03),
        "gray_p": 0.01,
        "blur_p": 0.3,
        "blur_kernel": 23,
        "blur_sigma": (0.75, 2.0),

        "erase_p": 0.04,
        "erase_scale": (0.02, 0.08),
        "erase_ratio": (0.3, 3.3),
    },
}

# ==============================================================================
# Augmentation Class
# ==============================================================================

class Augmentation:
    """
    Paired image augmentation for pairwise ranking.
    
    Input:
      - sample["image_l"], sample["image_r"]: PIL.Image
      - sample["score_r"]: int (pairwise label; sign indicates preference, 0 allowed when ties=True)
      - Optional gaze fields when enable_gaze=True and sample["has_eyetracker"]=True:
          • sample["gaze_l"], sample["gaze_r"]: array/tensor gaze maps
    
    Output:
      - sample["image_l"], sample["image_r"]: torch.FloatTensor [3, out_size, out_size], normalized
      - sample["score_r"]: possibly sign-flipped if the pair is swapped
      - sample["score_c"]: derived classification label
      - When enable_gaze=True:
          • sample["gaze_l"], sample["gaze_r"]: torch.FloatTensor [1, grid_h, grid_w]
    
    Policy:
      - Geometry: resize(short side -> resize_short) -> ensure min side >= out_size
                -> optional paired/unpaired hflip and rotation
                -> one RandomResizedCrop-style crop per side, resized to (out_size, out_size)
      - Pairing flags control whether random decisions are shared across left/right:
          • paired_hflip / paired_rotation: share flip decision / rotation angle
          • paired_scale: share crop scale fraction and aspect ratio (size/shape)
          • paired_crop: share crop position in normalized coordinates (location)
          • paired_color_jitter / paired_gray / paired_erase: share photometric / erasing decisions
      - Swap: with probability swap_p, left/right are swapped and score_r sign is inverted
      - Photometric: optional color jitter, grayscale, gaussian blur (PIL domain)
      - Tensor: to_tensor -> optional random erasing (tensor domain) -> normalize
      - Gaze (if enabled and available): geometric ops and crop mirror the corresponding image,
              then downsample to gaze_grid_size; erased regions are zeroed on the gaze grid
    """


    def __init__(
        self,
        augment: bool = True,
        ties: bool = True,
        resize_short: int = 256,
        out_size: int = 224,
        interpolation=InterpolationMode.BILINEAR,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),

        enable_gaze: bool = False,
        gaze_grid_size=(14, 14),
        gaze_output: str = "align",  # "align" or "guide" (also accepts "align+guide")


        paired_scale: bool = True,
        paired_hflip: bool = True,
        paired_crop: bool = True,
        paired_rotation: bool = True,
        paired_color_jitter: bool = False,
        paired_gray: bool = False,
        paired_erase: bool = True,

        swap_p: float = 0.5,
        hflip_p: float = 0.5,
        rot_deg: float = 0.0,
        rot_p: float = 0.0,
        crop_scale=(0.80, 1.00),
        crop_ratio=(3 / 4, 4 / 3),

        color_jitter=None,
        gray_p: float = 0.0,
        blur_p: float = 0.0,
        blur_kernel: int = 23,
        blur_sigma=(0.1, 2.0),

        erase_p: float = 0.0,
        erase_scale=(0.02, 0.20),
        erase_ratio=(0.3, 3.3),

        **kwargs,
    ):
        # Legacy alias mapping
        if "flip_p" in kwargs and kwargs["flip_p"] is not None:
            hflip_p = kwargs["flip_p"]
        if "rotation" in kwargs and kwargs["rotation"] is not None:
            rot_deg = kwargs["rotation"]
        if "rotation_p" in kwargs and kwargs["rotation_p"] is not None:
            rot_p = kwargs["rotation_p"]
        if "scale" in kwargs and kwargs["scale"] is not None:
            crop_scale = kwargs["scale"]
        if "ratio" in kwargs and kwargs["ratio"] is not None:
            crop_ratio = kwargs["ratio"]
        if "jitter" in kwargs and kwargs["jitter"] is not None:
            color_jitter = kwargs["jitter"]
        if "erase" in kwargs and kwargs["erase"] is not None:
            erase_p = kwargs["erase"]

        self.augment = bool(augment)
        self.ties = bool(ties)

        self.resize_short = int(resize_short)
        self.out_size = int(out_size)
        self.interpolation = interpolation
        self.mean = tuple(mean)
        self.std = tuple(std)

        self.enable_gaze = bool(enable_gaze)
        self.gaze_grid_size = (int(gaze_grid_size[0]), int(gaze_grid_size[1]))

        gaze_output = str(gaze_output).lower().strip()
        if gaze_output == "align+guide":
            gaze_output = "guide"
        if gaze_output not in ("align", "guide"):
            gaze_output = "align"
        self.gaze_output = gaze_output


        self.paired_scale = bool(paired_scale)
        self.paired_hflip = bool(paired_hflip)
        self.paired_crop = bool(paired_crop)
        self.paired_rotation = bool(paired_rotation)
        self.paired_color_jitter = bool(paired_color_jitter)
        self.paired_gray = bool(paired_gray)
        self.paired_erase = bool(paired_erase)

        self.swap_p = float(swap_p)
        self.hflip_p = float(hflip_p)
        self.rot_deg = float(rot_deg)
        self.rot_p = float(rot_p)

        self.crop_scale = tuple(crop_scale) if crop_scale is not None else None
        self.crop_ratio = tuple(crop_ratio) if crop_ratio is not None else None

        self.color_jitter = color_jitter
        self.gray_p = float(gray_p)
        self.blur_p = float(blur_p)
        self.blur_kernel = int(blur_kernel)
        self.blur_sigma = tuple(blur_sigma)

        self.erase_p = float(erase_p)
        self.erase_scale = tuple(erase_scale)
        self.erase_ratio = tuple(erase_ratio)

    # -------------------------
    # Small utilities
    # -------------------------
    def _gaze_out_hw(self):
        if self.gaze_output == "align":
            return self.gaze_grid_size
        return (self.out_size, self.out_size)

    @staticmethod
    def _as_bool(x) -> bool:
        if torch.is_tensor(x):
            return bool(x.item())
        return bool(x)

    @staticmethod
    def _to_int(x) -> int:
        if torch.is_tensor(x):
            return int(x.item())
        return int(x)

    def _score_r_to_score_c(self, score_r: int) -> int:
        if self.ties:
            return int(score_r) + 1
        if score_r <= 0:
            return 0
        return 1

    @staticmethod
    def _to_1ch_float(g) -> torch.Tensor:
        if g is None:
            return torch.zeros((1, 1, 1), dtype=torch.float32)
        if not torch.is_tensor(g):
            g = torch.as_tensor(g)
        g = g.float()

        if g.ndim == 2:
            g = g.unsqueeze(0)
        elif g.ndim == 3:
            if g.size(0) != 1:
                g = g.mean(dim=0, keepdim=True)
        else:
            return torch.zeros((1, 1, 1), dtype=torch.float32)

        return g.contiguous()

    @staticmethod
    def _resize_gaze_to_hw(g_1chw: torch.Tensor, size_hw, mode="bilinear") -> torch.Tensor:
        gh, gw = int(size_hw[0]), int(size_hw[1])
        x = g_1chw.unsqueeze(0)  # [1,1,H,W]
        x = nnF.interpolate(x, size=(gh, gw), mode=mode, align_corners=False if mode != "nearest" else None)
        return x.squeeze(0).contiguous()  # [1,gh,gw]

    @staticmethod
    def _hflip_gaze(g_1chw: torch.Tensor) -> torch.Tensor:
        return torch.flip(g_1chw, dims=[2])

    def _rotate_gaze(self, g_1chw: torch.Tensor, angle: float) -> torch.Tensor:
        return TF.rotate(
            g_1chw,
            angle=float(angle),
            interpolation=InterpolationMode.BILINEAR,
            expand=False,
            fill=0.0,
            center=None,
        )

    def _downsample_gaze_to_grid(self, g_1chw: torch.Tensor) -> torch.Tensor:
        return self._resize_gaze_to_hw(g_1chw, self.gaze_grid_size, mode="bilinear")

    @staticmethod
    def _cj_range(v, is_hue: bool = False):
        if v is None:
            return None
        if isinstance(v, (tuple, list)):
            if len(v) != 2:
                raise ValueError("Color jitter ranges must be (min, max).")
            lo, hi = float(v[0]), float(v[1])
            if is_hue:
                lo = max(lo, -0.5)
                hi = min(hi, 0.5)
            return (lo, hi)

        v = float(v)
        if v <= 0.0:
            return None

        if is_hue:
            lo, hi = -v, v
            lo = max(lo, -0.5)
            hi = min(hi, 0.5)
            return (lo, hi)

        return (max(0.0, 1.0 - v), 1.0 + v)

    @staticmethod
    def _ensure_min_side_pil(img, min_side: int, interpolation):
        w, h = img.size
        if min(w, h) >= min_side:
            return img
        return TF.resize(img, min_side, interpolation=interpolation)

    @staticmethod
    def _sample_erase_rect(H: int, W: int, p: float, scale, ratio):
        if p <= 0.0 or (random.random() >= p):
            return None

        area = H * W
        for _ in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect = random.uniform(ratio[0], ratio[1])

            eh = int(round(math.sqrt(erase_area * aspect)))
            ew = int(round(math.sqrt(erase_area / aspect)))

            if 0 < eh < H and 0 < ew < W:
                top = random.randint(0, H - eh)
                left = random.randint(0, W - ew)
                return top, left, eh, ew

        return None

    @staticmethod
    def _sample_crop_hw(H: int, W: int, scale_range, ratio_range):
        area = H * W
        log_ratio = (math.log(ratio_range[0]), math.log(ratio_range[1]))

        for _ in range(10):
            target_area = random.uniform(scale_range[0], scale_range[1]) * area
            aspect = math.exp(random.uniform(log_ratio[0], log_ratio[1]))

            w = int(round(math.sqrt(target_area * aspect)))
            h = int(round(math.sqrt(target_area / aspect)))

            if 0 < h <= H and 0 < w <= W:
                return h, w

        side = min(H, W)
        return side, side

    @staticmethod
    def _center_ij(H: int, W: int, h: int, w: int):
        i = max(0, (H - h) // 2)
        j = max(0, (W - w) // 2)
        return i, j

    @staticmethod
    def _uv_ij(H: int, W: int, h: int, w: int, u: float, v: float):
        max_i = max(0, H - h)
        max_j = max(0, W - w)
        i = 0 if max_i == 0 else int(round(u * max_i))
        j = 0 if max_j == 0 else int(round(v * max_j))
        i = max(0, min(max_i, i))
        j = max(0, min(max_j, j))
        return i, j

    def _maybe_swap_pair(self, sample: dict, img_l, img_r, score_r: int):
        if not (self.augment and (random.random() < self.swap_p)):
            return img_l, img_r, score_r

        img_l, img_r = img_r, img_l
        score_r = -score_r

        swap_pairs = [
            ("image_l_name", "image_r_name"),
            ("gaze_l", "gaze_r"),
            ("gaze_ok_l", "gaze_ok_r"),
            ("gaze_path_l", "gaze_path_r"),
        ]
        for k1, k2 in swap_pairs:
            if (k1 in sample) and (k2 in sample):
                sample[k1], sample[k2] = sample[k2], sample[k1]

        return img_l, img_r, score_r
    
    @staticmethod
    def _sample_color_jitter_params(brightness, contrast, saturation, hue):
        params = []
        if brightness is not None:
            params.append(("brightness", random.uniform(brightness[0], brightness[1])))
        if contrast is not None:
            params.append(("contrast", random.uniform(contrast[0], contrast[1])))
        if saturation is not None:
            params.append(("saturation", random.uniform(saturation[0], saturation[1])))
        if hue is not None:
            params.append(("hue", random.uniform(hue[0], hue[1])))

        random.shuffle(params)
        return params

    @staticmethod
    def _apply_color_jitter(img, params):
        for name, factor in params:
            if name == "brightness":
                img = TF.adjust_brightness(img, factor)
            elif name == "contrast":
                img = TF.adjust_contrast(img, factor)
            elif name == "saturation":
                img = TF.adjust_saturation(img, factor)
            elif name == "hue":
                img = TF.adjust_hue(img, factor)
        return img

    # -------------------------
    # Crop policy (single crop per side)
    # -------------------------
    @staticmethod
    def _sample_scale_aspect(scale_range, ratio_range):
        scale_frac = random.uniform(scale_range[0], scale_range[1])
        log_r0 = math.log(ratio_range[0])
        log_r1 = math.log(ratio_range[1])
        aspect = math.exp(random.uniform(log_r0, log_r1))
        return scale_frac, aspect
    
    @staticmethod
    def _crop_hw_from_scale_aspect(H: int, W: int, scale_frac: float, aspect: float):
        area = H * W
        target_area = scale_frac * area
    
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
    
        if 0 < h <= H and 0 < w <= W:
            return h, w
        return None
        
    def _sample_crop_pair(self, img_l, img_r):
        Hl, Wl = img_l.size[1], img_l.size[0]
        Hr, Wr = img_r.size[1], img_r.size[0]
    
        scale_rng = self.crop_scale
        ratio_rng = self.crop_ratio
    
        if not (self.augment and (scale_rng is not None) and (ratio_rng is not None)):
            hl = wl = self.out_size
            hr = wr = self.out_size
            il, jl = self._center_ij(Hl, Wl, hl, wl)
            ir, jr = self._center_ij(Hr, Wr, hr, wr)
            return (il, jl, hl, wl), (ir, jr, hr, wr)
    
        u = random.random() if self.paired_crop else None
        v = random.random() if self.paired_crop else None
    
        if self.paired_scale:
            for _ in range(10):
                scale_frac, aspect = self._sample_scale_aspect(scale_rng, ratio_rng)
    
                hw_l = self._crop_hw_from_scale_aspect(Hl, Wl, scale_frac, aspect)
                hw_r = self._crop_hw_from_scale_aspect(Hr, Wr, scale_frac, aspect)
    
                if (hw_l is not None) and (hw_r is not None):
                    hl, wl = hw_l
                    hr, wr = hw_r
                    break
            else:
                sl = min(Hl, Wl)
                sr = min(Hr, Wr)
                hl = wl = sl
                hr = wr = sr
        else:
            hl, wl = self._sample_crop_hw(Hl, Wl, scale_rng, ratio_rng)
            hr, wr = self._sample_crop_hw(Hr, Wr, scale_rng, ratio_rng)
    
        if self.paired_crop:
            il, jl = self._uv_ij(Hl, Wl, hl, wl, u, v)
            ir, jr = self._uv_ij(Hr, Wr, hr, wr, u, v)
        else:
            ul, vl = random.random(), random.random()
            ur, vr = random.random(), random.random()
            il, jl = self._uv_ij(Hl, Wl, hl, wl, ul, vl)
            ir, jr = self._uv_ij(Hr, Wr, hr, wr, ur, vr)
    
        return (il, jl, hl, wl), (ir, jr, hr, wr)

    # -------------------------
    # Main entry point
    # -------------------------
    def __call__(self, sample: dict) -> dict:
        img_l = sample["image_l"]
        img_r = sample["image_r"]

        score_r = self._to_int(sample.get("score_r", 0))
        has_eye = self._as_bool(sample.get("has_eyetracker", False))

        img_l, img_r, score_r = self._maybe_swap_pair(sample, img_l, img_r, score_r)
        sample["score_r"] = score_r
        sample["score_c"] = self._score_r_to_score_c(score_r)

        img_l = TF.resize(img_l, self.resize_short, interpolation=self.interpolation)
        img_r = TF.resize(img_r, self.resize_short, interpolation=self.interpolation)
        img_l = self._ensure_min_side_pil(img_l, self.out_size, self.interpolation)
        img_r = self._ensure_min_side_pil(img_r, self.out_size, self.interpolation)

        if self.enable_gaze and has_eye:
            g_l = self._to_1ch_float(sample.get("gaze_l", None))
            g_r = self._to_1ch_float(sample.get("gaze_r", None))

            wl, hl = img_l.size
            wr, hr = img_r.size
            g_l = self._resize_gaze_to_hw(g_l, (hl, wl), mode="bilinear")
            g_r = self._resize_gaze_to_hw(g_r, (hr, wr), mode="bilinear")
        else:
            g_l = None
            g_r = None

        # HFlip
        if self.augment and self.hflip_p > 0.0:
            if self.paired_hflip:
                do_flip = (random.random() < self.hflip_p)
                if do_flip:
                    img_l = TF.hflip(img_l)
                    img_r = TF.hflip(img_r)
                    if self.enable_gaze and has_eye:
                        g_l = self._hflip_gaze(g_l)
                        g_r = self._hflip_gaze(g_r)
            else:
                do_flip_l = (random.random() < self.hflip_p)
                do_flip_r = (random.random() < self.hflip_p)
                if do_flip_l:
                    img_l = TF.hflip(img_l)
                    if self.enable_gaze and has_eye:
                        g_l = self._hflip_gaze(g_l)
                if do_flip_r:
                    img_r = TF.hflip(img_r)
                    if self.enable_gaze and has_eye:
                        g_r = self._hflip_gaze(g_r)

        # Rotation
        if self.augment and self.rot_deg > 0.0 and self.rot_p > 0.0:
            if self.paired_rotation:
                do_rot = (random.random() < self.rot_p)
                if do_rot:
                    ang = random.uniform(-self.rot_deg, self.rot_deg)
                    img_l = TF.rotate(img_l, ang, interpolation=self.interpolation, expand=False, fill=0)
                    img_r = TF.rotate(img_r, ang, interpolation=self.interpolation, expand=False, fill=0)
                    if self.enable_gaze and has_eye:
                        g_l = self._rotate_gaze(g_l, ang)
                        g_r = self._rotate_gaze(g_r, ang)
            else:
                do_rot_l = (random.random() < self.rot_p)
                do_rot_r = (random.random() < self.rot_p)
                if do_rot_l:
                    ang_l = random.uniform(-self.rot_deg, self.rot_deg)
                    img_l = TF.rotate(img_l, ang_l, interpolation=self.interpolation, expand=False, fill=0)
                    if self.enable_gaze and has_eye:
                        g_l = self._rotate_gaze(g_l, ang_l)
                if do_rot_r:
                    ang_r = random.uniform(-self.rot_deg, self.rot_deg)
                    img_r = TF.rotate(img_r, ang_r, interpolation=self.interpolation, expand=False, fill=0)
                    if self.enable_gaze and has_eye:
                        g_r = self._rotate_gaze(g_r, ang_r)

        # Single crop per side (RandomResizedCrop-style), then resize to out_size
        (il, jl, hl, wl), (ir, jr, hr, wr) = self._sample_crop_pair(img_l, img_r)

        img_l = TF.resized_crop(img_l, il, jl, hl, wl, [self.out_size, self.out_size], interpolation=self.interpolation)
        img_r = TF.resized_crop(img_r, ir, jr, hr, wr, [self.out_size, self.out_size], interpolation=self.interpolation)

        if self.enable_gaze and has_eye:
            g_l = TF.resized_crop(g_l, il, jl, hl, wl, [self.out_size, self.out_size], interpolation=InterpolationMode.BILINEAR)
            g_r = TF.resized_crop(g_r, ir, jr, hr, wr, [self.out_size, self.out_size], interpolation=InterpolationMode.BILINEAR)

        # Photometric
        if self.augment:
            if self.color_jitter is not None:
                b, c, s, h = self.color_jitter
                b = self._cj_range(b, is_hue=False)
                c = self._cj_range(c, is_hue=False)
                s = self._cj_range(s, is_hue=False)
                h = self._cj_range(h, is_hue=True)

                if (b is not None) or (c is not None) or (s is not None) or (h is not None):
                    if self.paired_color_jitter:
                        params = self._sample_color_jitter_params(b, c, s, h)
                        img_l = self._apply_color_jitter(img_l, params)
                        img_r = self._apply_color_jitter(img_r, params)
                    else:
                        params_l = self._sample_color_jitter_params(b, c, s, h)
                        params_r = self._sample_color_jitter_params(b, c, s, h)
                        img_l = self._apply_color_jitter(img_l, params_l)
                        img_r = self._apply_color_jitter(img_r, params_r)


            if self.gray_p > 0.0:
                if self.paired_gray:
                    do_gray = (random.random() < self.gray_p)
                    if do_gray:
                        img_l = TF.rgb_to_grayscale(img_l, num_output_channels=3)
                        img_r = TF.rgb_to_grayscale(img_r, num_output_channels=3)
                else:
                    if random.random() < self.gray_p:
                        img_l = TF.rgb_to_grayscale(img_l, num_output_channels=3)
                    if random.random() < self.gray_p:
                        img_r = TF.rgb_to_grayscale(img_r, num_output_channels=3)

            if self.blur_p > 0.0:
                if self.paired_color_jitter:
                    do_blur = (random.random() < self.blur_p)
                    if do_blur:
                        sigma = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
                        blur = transforms.GaussianBlur(kernel_size=self.blur_kernel, sigma=(sigma, sigma))
                        img_l = blur(img_l)
                        img_r = blur(img_r)
                else:
                    if random.random() < self.blur_p:
                        sigma_l = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
                        blur_l = transforms.GaussianBlur(kernel_size=self.blur_kernel, sigma=(sigma_l, sigma_l))
                        img_l = blur_l(img_l)
                    if random.random() < self.blur_p:
                        sigma_r = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
                        blur_r = transforms.GaussianBlur(kernel_size=self.blur_kernel, sigma=(sigma_r, sigma_r))
                        img_r = blur_r(img_r)

        # ToTensor
        x_l = TF.to_tensor(img_l)
        x_r = TF.to_tensor(img_r)

        # Erase
        erase_rect_l = None
        erase_rect_r = None
        if self.augment:
            if self.paired_erase:
                rect = self._sample_erase_rect(self.out_size, self.out_size, self.erase_p, self.erase_scale, self.erase_ratio)
                erase_rect_l = rect
                erase_rect_r = rect
            else:
                erase_rect_l = self._sample_erase_rect(self.out_size, self.out_size, self.erase_p, self.erase_scale, self.erase_ratio)
                erase_rect_r = self._sample_erase_rect(self.out_size, self.out_size, self.erase_p, self.erase_scale, self.erase_ratio)

        if erase_rect_l is not None:
            top, left, eh, ew = erase_rect_l
            x_l[:, top:top + eh, left:left + ew] = 0.0
        if erase_rect_r is not None:
            top, left, eh, ew = erase_rect_r
            x_r[:, top:top + eh, left:left + ew] = 0.0

        # Normalize
        x_l = TF.normalize(x_l, mean=self.mean, std=self.std)
        x_r = TF.normalize(x_r, mean=self.mean, std=self.std)

        sample["image_l"] = x_l
        sample["image_r"] = x_r

        # Gaze output
        if self.enable_gaze:
            out_h, out_w = self._gaze_out_hw()

            if (not has_eye) or (g_l is None) or (g_r is None):
                sample["gaze_l"] = torch.zeros((1, out_h, out_w), dtype=torch.float32)
                sample["gaze_r"] = torch.zeros((1, out_h, out_w), dtype=torch.float32)
            else:
                if self.gaze_output == "align":
                    # Final downsample to transformer/CNN attention grid
                    g_l = self._downsample_gaze_to_grid(g_l)
                    g_r = self._downsample_gaze_to_grid(g_r)

                    gh, gw = self.gaze_grid_size

                    if erase_rect_l is not None:
                        top, left, eh, ew = erase_rect_l
                        y0 = int(math.floor(top * gh / self.out_size))
                        x0 = int(math.floor(left * gw / self.out_size))
                        y1 = int(math.ceil((top + eh) * gh / self.out_size))
                        x1 = int(math.ceil((left + ew) * gw / self.out_size))
                        y0 = max(0, min(gh, y0))
                        y1 = max(0, min(gh, y1))
                        x0 = max(0, min(gw, x0))
                        x1 = max(0, min(gw, x1))
                        if y1 > y0 and x1 > x0:
                            g_l[:, y0:y1, x0:x1] = 0.0

                    if erase_rect_r is not None:
                        top, left, eh, ew = erase_rect_r
                        y0 = int(math.floor(top * gh / self.out_size))
                        x0 = int(math.floor(left * gw / self.out_size))
                        y1 = int(math.ceil((top + eh) * gh / self.out_size))
                        x1 = int(math.ceil((left + ew) * gw / self.out_size))
                        y0 = max(0, min(gh, y0))
                        y1 = max(0, min(gh, y1))
                        x0 = max(0, min(gw, x0))
                        x1 = max(0, min(gw, x1))
                        if y1 > y0 and x1 > x0:
                            g_r[:, y0:y1, x0:x1] = 0.0

                else:
                    # guide: keep gaze at out_size x out_size (same as model image input)
                    if erase_rect_l is not None:
                        top, left, eh, ew = erase_rect_l
                        g_l[:, top:top + eh, left:left + ew] = 0.0
                    if erase_rect_r is not None:
                        top, left, eh, ew = erase_rect_r
                        g_r[:, top:top + eh, left:left + ew] = 0.0

                sample["gaze_l"] = g_l
                sample["gaze_r"] = g_r

        return sample
