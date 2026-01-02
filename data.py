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

class ComparisonsDataset(Dataset):
    """Cycling Safety Perception dataset."""

    def __init__(self, dataframe, root_dir, transform=None, logger=None, gaze_root=None, use_gaze=True, use_seg=False):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with comparisons images and scores.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            logger (logging, optional): Logger object
            gaze_root (string, optional): base folder for npy_file_l / npy_file_r
        """
        self.comparisons_frame = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.logger = logger
        self.gaze_root = gaze_root
        self.use_gaze = use_gaze
        self.use_seg = use_seg
        
    def __len__(self):
        return len(self.comparisons_frame)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        return img

    def _load_gaze_npy(self, fname):
        """
        Try to load gaze heatmap from .npy and return (tensor [14,14], found_flag).
        If file is missing or unreadable, return zeros and found_flag=False.
        """

        if not fname:
            return torch.zeros((14, 14), dtype=torch.float32), False

        full_path = (
            os.path.join(self.gaze_root, fname)
            if (self.gaze_root is not None and not os.path.isabs(fname))
            else fname
        )

        if not os.path.exists(full_path):
            return torch.zeros((14, 14), dtype=torch.float32), False

        try:
            arr = np.load(full_path)  # expected shape [14,14]
            tensor = torch.from_numpy(arr).float()
            return tensor, True
        except Exception:
            # Corrupt or unreadable file: degrade gracefully
            return torch.zeros((14, 14), dtype=torch.float32), False


    def __getitem__(self, idx):
        start = timer()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        row = self.comparisons_frame.iloc[idx]

        # -------------------------------------------------
        # Get left and right image paths
        # -------------------------------------------------
        # Optional: city / dataset subfolder
        city = row['dataset'] if 'dataset' in row and pd.notna(row['dataset']) else None

        if city:
            img_l_name = os.path.join(self.root_dir, city, row['image_l'])
            img_r_name = os.path.join(self.root_dir, city, row['image_r'])
        else:
            # fallback: old behavior (for legacy runs where dataset col might not exist)
            img_l_name = os.path.join(self.root_dir, row['image_l'])
            img_r_name = os.path.join(self.root_dir, row['image_r'])

        # If requested, swap to *_seg.jpg filenames
        if self.use_seg:
            img_l_seg = re.sub(r'(?i)\.jpg$', '_seg.jpg', img_l_name)
            img_r_seg = re.sub(r'(?i)\.jpg$', '_seg.jpg', img_r_name)
        
            if not os.path.exists(img_l_seg):
                raise FileNotFoundError(f"[--use_seg] Segmented file missing: {img_l_seg}")
            if not os.path.exists(img_r_seg):
                raise FileNotFoundError(f"[--use_seg] Segmented file missing: {img_r_seg}")
        
            img_l_name = img_l_seg
            img_r_name = img_r_seg


        # 🔹 ACTUALLY LOAD THE IMAGES
        image_l = self._load_image(img_l_name)
        image_r = self._load_image(img_r_name)


        # -------------------------------------------------
        # Labels
        # -------------------------------------------------
        # Ranking label (-1 / 0 / +1)
        score = int(row['score'])
        # Classification label (0/1 or 0/1/2)
        score_classification = int(row['score_classification'])

        # -------------------------------------------------
        # Eyetracker / gaze
        # -------------------------------------------------
        # has_eyetracker might not exist for all rows in future, so be defensive
        # --- Eyetracker / gaze ---
        has_eye_flag = bool(row['has_eyetracker']) if 'has_eyetracker' in row else False

        # Respect the global toggle
        if not self.use_gaze:
            has_eye_flag = False

        gaze_file_l = row['npy_file_l'] if 'npy_file_l' in row else None
        gaze_file_r = row['npy_file_r'] if 'npy_file_r' in row else None

        if has_eye_flag:
            gaze_l, ok_l = self._load_gaze_npy(gaze_file_l)  # [14,14], bool
            gaze_r, ok_r = self._load_gaze_npy(gaze_file_r)  # [14,14], bool
        else:
            gaze_l, ok_l = torch.zeros((14,14), dtype=torch.float32), False
            gaze_r, ok_r = torch.zeros((14,14), dtype=torch.float32), False

        # Combined mask: only use attention loss if BOTH sides have gaze
        has_eye_tensor = torch.tensor(bool(ok_l and ok_r), dtype=torch.bool)


        # Optional metadata (kept for debugging / analysis, not needed for loss)
        survey_id = row['survey_id'] if 'survey_id' in row else None
        trial_id = row['trial_id'] if 'trial_id' in row else None
        
        sample = {
            'image_l': image_l, 
            'image_r': image_r, 

            'score_r': score,
            'score_c': score_classification,

            'image_l_name': img_l_name,
            'image_r_name': img_r_name,

            # NEW FIELDS:
            'has_eyetracker': has_eye_tensor,  # [bool]
            'gaze_l': gaze_l,                  # [14,14] float32 tensor
            'gaze_r': gaze_r,                  # [14,14] float32 tensor
            #'survey_id': survey_id,
            #'trial_id': trial_id,
        }
        
        if self.transform:
            sample = self.transform(sample)

        end = timer()
        if self.logger:
            self.logger.info(f'DATALOADER, {end-start:.4f}')

        return sample

class DictTransform:
    """
    Wraps a standard image transform (Resize, Crop, etc.) so it can work 
    on your dictionary-based dataset (image_l, image_r).
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        # Apply the transform to left and right images if they exist
        if "image_l" in data:
            data["image_l"] = self.transform(data["image_l"])
        
        if "image_r" in data:
            data["image_r"] = self.transform(data["image_r"])
            
        # Note: We do NOT transform 'gaze' maps with standard image ops 
        # (like Normalize) because gaze maps have different semantics.
        return data


# ========================================================================= #
# CONFIGURATION & PRESETS for Transformations
# ========================================================================= #

# Default ImageNet-style fallback (used only if timm config cannot be resolved)
DEFAULT_SPECS = {
    "input_size": (3, 224, 224),
    "crop_pct": 0.9,
    "interpolation": "bicubic",
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}

# -----------------------------------------------------------------------------------------------
# Backbone alias → timm model id mapping
# -----------------------------------------------------------------------------------------------
BACKBONE_ALIAS_TO_TIMM_ID = {
    # 1) DINOv3
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",

    # 2) BEiT v2
    "beitv2_base_patch16_224": "beitv2_base_patch16_224.in1k_ft_in22k",

    # 3) DeiT III
    "deit3_base_patch16_224": "deit3_base_patch16_224.fb_in22k_ft_in1k",

    # 4) SigLIP
    "siglip_base_patch16_224": "vit_base_patch16_siglip_224",

    # 5) CLIP
    "vit_base_patch16_clip_224": "vit_base_patch16_clip_224.openai",

    # EVA-02 (native 448 in your codebase)
    "eva02_base": "eva02_base_patch14_448.mim_in22k_ft_in1k",
}


def get_model_specs(backbone_name: str):
    """
    Resolve preprocessing specs (mean/std, crop_pct, interpolation, input_size) for a backbone.

    Critical:
      - Your sweeps/CLI use aliases (e.g., "vit_base_patch16_clip_224") that are not valid timm IDs.
      - If we fail to resolve config, we silently use DEFAULT_SPECS (ImageNet mean/std),
        which is wrong for CLIP/SigLIP and can wreck performance.

    This function maps alias → timm ID before calling timm.
    """
    import timm

    # 0) Map alias → timm id (if applicable)
    timm_id = BACKBONE_ALIAS_TO_TIMM_ID.get(backbone_name, backbone_name)

    try:
        # We do NOT need pretrained weights to resolve preprocessing config
        dummy_model = timm.create_model(timm_id, pretrained=False)
        cfg = timm.data.resolve_data_config(dummy_model.pretrained_cfg)

        specs = {
            "input_size": cfg.get("input_size", DEFAULT_SPECS["input_size"]),
            "crop_pct": cfg.get("crop_pct", DEFAULT_SPECS["crop_pct"]),
            "interpolation": cfg.get("interpolation", DEFAULT_SPECS["interpolation"]),
            "mean": cfg.get("mean", DEFAULT_SPECS["mean"]),
            "std": cfg.get("std", DEFAULT_SPECS["std"]),
        }

        # Your codebase trains transformers at 224 except EVA-02 at 448.
        if backbone_name == "eva02_base" or "eva02" in backbone_name:
            specs["input_size"] = (3, 448, 448)
            specs["interpolation"] = specs.get("interpolation") or "bicubic"
        else:
            specs["input_size"] = (3, 224, 224)

        return specs

    except Exception:
        print(
            f"Warning: Could not auto-config '{backbone_name}' (timm id='{timm_id}'). Using defaults."
        )
        specs = dict(DEFAULT_SPECS)

        # Preserve your EVA-02 behavior
        if backbone_name == "eva02_base" or "eva02" in backbone_name:
            specs["input_size"] = (3, 448, 448)
            specs["interpolation"] = "bicubic"
        else:
            specs["input_size"] = (3, 224, 224)

        return specs


def _get_interp_mode(mode_str):
    mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
    }
    return mapping.get(str(mode_str).lower(), InterpolationMode.BILINEAR)

# =============================================================================================== #
# Augmentation presets
# =============================================================================================== #

# Light augmentation:
#   - paired horizontal flip
#   - label-aware left/right swap
#   - paired random crop keeping ~85% of area
#
# Heavy augmentation:
#   - includes light ops
#   - adds paired color jitter, grayscale, random erasing
#   - adds small-angle rotation

AUG_PRESETS = {
    "light": dict(
        hflip_p=0.35,
        swap_p=0.50,
        crop_p=0.30,
        crop_keep_area=0.85,
        rotation_p=0.0,
        max_rotation_deg=0.0,
        color_jitter_p=0.0,
        jitter_brightness=0.0,
        jitter_contrast=0.0,
        jitter_saturation=0.0,
        jitter_hue=0.0,
        gray_p=0.0,
        erase_p=0.0,
        erase_scale=(0.02, 0.06),
        erase_ratio=(0.3, 3.3),
        erase_value=0.0,
    ),
    "heavy": dict(
        hflip_p=0.35,
        swap_p=0.50,
        crop_p=0.60,
        crop_keep_area=0.75,
        rotation_p=0.25,
        max_rotation_deg=5.0,
        color_jitter_p=0.45,
        jitter_brightness=0.25,
        jitter_contrast=0.25,
        jitter_saturation=0.20,
        jitter_hue=0.08,
        gray_p=0.10,
        erase_p=0.15,
        erase_scale=(0.03, 0.12),
        erase_ratio=(0.3, 3.3),
        erase_value=0.0,
    ),
}


def build_eval_transforms(specs: dict):
    """Build deterministic evaluation preprocessing.

    Returns:
        eval_tfms: torchvision Compose of DictTransform wrappers
        meta: dict with resolved sizes and normalization statistics
    """
    target_crop = int(specs["input_size"][-1])
    crop_pct = float(specs["crop_pct"])

    resize_dim = int(round(target_crop / crop_pct))
    resize_dim = max(resize_dim, target_crop)

    interp_mode = _get_interp_mode(specs["interpolation"])
    mean = tuple(specs["mean"])
    std = tuple(specs["std"])

    eval_tfms = transforms.Compose(
        [
            DictTransform(transforms.Resize(resize_dim, interpolation=interp_mode)),
            DictTransform(transforms.CenterCrop(target_crop)),
            DictTransform(transforms.ToTensor()),
            DictTransform(transforms.Normalize(mean=mean, std=std)),
        ]
    )

    meta = {
        "target_crop": target_crop,
        "resize_dim": resize_dim,
        "crop_pct": crop_pct,
        "interpolation": str(specs.get("interpolation", "bilinear")),
        "mean": mean,
        "std": std,
        "eval_policy": "Resize(short) -> CenterCrop -> ToTensor -> Normalize",
    }

    return eval_tfms, meta


class Augmentation:
    """
    Paired augmentation pipeline for pairwise comparisons.

    Expected input sample (dict):
        sample = {
            "image_l": PIL.Image or torch.Tensor [C,H,W],
            "image_r": PIL.Image or torch.Tensor [C,H,W],
            "score_r": int or tensor scalar in {-1, 0, +1},
            "score_c": int or tensor scalar (classification label encoding),
        }

    Output:
        Same dict with images converted to normalized tensors and labels updated
        if a swap occurs.
    """

    def __init__(
        self,
        augment: bool = True,
        ties: bool = True,

        # Paired invariances
        hflip_p: float = 0.35,
        swap_p: float = 0.50,

        # Paired crop
        crop_p: float = 0.30,
        crop_keep_area: float = 0.85,

        # Paired small rotation
        rotation_p: float = 0.0,
        max_rotation_deg: float = 0.0,

        # Paired photometric
        color_jitter_p: float = 0.0,
        jitter_brightness: float = 0.0,
        jitter_contrast: float = 0.0,
        jitter_saturation: float = 0.0,
        jitter_hue: float = 0.0,
        gray_p: float = 0.0,

        # Paired random erasing (tensor space)
        erase_p: float = 0.0,
        erase_scale: tuple = (0.02, 0.06),
        erase_ratio: tuple = (0.3, 3.3),
        erase_value: float = 0.0,

        # Output geometry and normalization
        resize_short: int = 256,
        out_size: int = 224,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
    ):
        self.augment = bool(augment)
        self.ties = bool(ties)

        self.hflip_p = float(hflip_p)
        self.swap_p = float(swap_p)

        self.crop_p = float(crop_p)
        self.crop_keep_area = float(crop_keep_area)

        self.rotation_p = float(rotation_p)
        self.max_rotation_deg = float(max_rotation_deg)

        self.color_jitter_p = float(color_jitter_p)
        self.jitter_brightness = float(jitter_brightness)
        self.jitter_contrast = float(jitter_contrast)
        self.jitter_saturation = float(jitter_saturation)
        self.jitter_hue = float(jitter_hue)
        self.gray_p = float(gray_p)

        self.erase_p = float(erase_p)
        self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio
        self.erase_value = float(erase_value)

        self.resize_short = int(resize_short)
        self.out_size = int(out_size)

        self.interpolation = interpolation
        self.mean = mean
        self.std = std

    def _score_r_to_score_c(self, score_r: int) -> int:
        """Convert ranking score_r in {-1,0,+1} to a classification index."""
        if self.ties:
            if score_r == -1:
                return 0
            if score_r == 0:
                return 1
            return 2
        return 0 if score_r == -1 else 1

    @staticmethod
    def _resize_short_side(pil_img: Image.Image, short_side: int) -> Image.Image:
        """Resize image so that the shorter side equals `short_side`."""
        w, h = pil_img.size
        if w <= h:
            new_w = short_side
            new_h = int(round(h * (short_side / w)))
        else:
            new_h = short_side
            new_w = int(round(w * (short_side / h)))
        return pil_img.resize((new_w, new_h), Image.BILINEAR)

    def _sample_keep_area_crop(self, w: int, h: int):
        """Sample a paired random crop keeping a fixed fraction of area."""
        keep = max(0.10, min(1.0, float(self.crop_keep_area)))
        scale = math.sqrt(keep)
        crop_w = max(1, min(w, int(round(w * scale))))
        crop_h = max(1, min(h, int(round(h * scale))))

        if crop_w == w and crop_h == h:
            return 0, 0, w, h

        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        return left, top, crop_w, crop_h

    def _apply_paired_color_jitter(self, im_l: Image.Image, im_r: Image.Image):
        """Apply identical brightness/contrast/saturation/hue perturbations to both images."""
        b = 1.0 + random.uniform(-self.jitter_brightness, self.jitter_brightness)
        c = 1.0 + random.uniform(-self.jitter_contrast, self.jitter_contrast)
        s = 1.0 + random.uniform(-self.jitter_saturation, self.jitter_saturation)
        h_shift = random.uniform(-self.jitter_hue, self.jitter_hue)

        def jitter_one(im: Image.Image) -> Image.Image:
            im = ImageEnhance.Brightness(im).enhance(b)
            im = ImageEnhance.Contrast(im).enhance(c)
            im = ImageEnhance.Color(im).enhance(s)
            hsv = np.array(im.convert("HSV"), dtype=np.uint8)
            hsv[..., 0] = (hsv[..., 0].astype(int) + int(h_shift * 255)) % 256
            return Image.fromarray(hsv, mode="HSV").convert("RGB")

        return jitter_one(im_l), jitter_one(im_r)

    def _sample_erasing_rect(self, H: int, W: int):
        """Sample an erasing rectangle (top, left, height, width) in tensor space."""
        area = H * W
        for _ in range(20):
            target_area = random.uniform(*self.erase_scale) * area
            log_ratio = (math.log(self.erase_ratio[0]), math.log(self.erase_ratio[1]))
            aspect = math.exp(random.uniform(*log_ratio))

            eh = int(round(math.sqrt(target_area / aspect)))
            ew = int(round(math.sqrt(target_area * aspect)))

            if 0 < eh < H and 0 < ew < W:
                top = random.randint(0, H - eh)
                left = random.randint(0, W - ew)
                return top, left, eh, ew
        return None

    def _paired_erase(self, x_l: torch.Tensor, x_r: torch.Tensor):
        """Apply the same erasing rectangle to both tensors x_l and x_r."""
        _, H, W = x_l.shape
        rect = self._sample_erasing_rect(H, W)
        if rect is None:
            return x_l, x_r
        top, left, eh, ew = rect
        x_l[:, top:top + eh, left:left + ew] = self.erase_value
        x_r[:, top:top + eh, left:left + ew] = self.erase_value
        return x_l, x_r

    def __call__(self, sample: dict) -> dict:
        image_l = sample["image_l"]
        image_r = sample["image_r"]

        score_r = int(sample["score_r"].item()) if torch.is_tensor(sample["score_r"]) else int(sample["score_r"])
        score_c = int(sample["score_c"].item()) if torch.is_tensor(sample["score_c"]) else int(sample["score_c"])

        do_aug = bool(self.augment)

        pil_inputs = (not torch.is_tensor(image_l)) and (not torch.is_tensor(image_r))
        if pil_inputs:
            image_l = self._resize_short_side(image_l, self.resize_short)
            image_r = self._resize_short_side(image_r, self.resize_short)

            if do_aug:
                if random.random() < self.hflip_p:
                    image_l = TF.hflip(image_l)
                    image_r = TF.hflip(image_r)

                if random.random() < self.crop_p:
                    w, h = image_l.size
                    left, top, cw, ch = self._sample_keep_area_crop(w, h)
                    image_l = TF.resized_crop(
                        image_l, top, left, ch, cw, (self.out_size, self.out_size),
                        interpolation=self.interpolation
                    )
                    image_r = TF.resized_crop(
                        image_r, top, left, ch, cw, (self.out_size, self.out_size),
                        interpolation=self.interpolation
                    )
                else:
                    image_l = TF.resize(image_l, (self.out_size, self.out_size), interpolation=self.interpolation)
                    image_r = TF.resize(image_r, (self.out_size, self.out_size), interpolation=self.interpolation)

                if self.rotation_p > 0.0 and random.random() < self.rotation_p:
                    angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
                    image_l = TF.rotate(image_l, angle=angle, interpolation=self.interpolation, expand=False)
                    image_r = TF.rotate(image_r, angle=angle, interpolation=self.interpolation, expand=False)

                if random.random() < self.color_jitter_p:
                    image_l, image_r = self._apply_paired_color_jitter(image_l, image_r)

                if random.random() < self.gray_p:
                    image_l = TF.to_grayscale(image_l, num_output_channels=3)
                    image_r = TF.to_grayscale(image_r, num_output_channels=3)

                if random.random() < self.swap_p:
                    image_l, image_r = image_r, image_l
                    score_r = -score_r
                    score_c = self._score_r_to_score_c(score_r)

            else:
                image_l = TF.resize(image_l, (self.out_size, self.out_size), interpolation=self.interpolation)
                image_r = TF.resize(image_r, (self.out_size, self.out_size), interpolation=self.interpolation)

            x_l = TF.to_tensor(image_l)
            x_r = TF.to_tensor(image_r)
        else:
            x_l = image_l if torch.is_tensor(image_l) else TF.to_tensor(image_l)
            x_r = image_r if torch.is_tensor(image_r) else TF.to_tensor(image_r)

        if do_aug and (random.random() < self.erase_p):
            x_l, x_r = self._paired_erase(x_l, x_r)

        # Ensure training and eval share the same normalization convention.
        x_l = TF.normalize(x_l, mean=self.mean, std=self.std)
        x_r = TF.normalize(x_r, mean=self.mean, std=self.std)

        sample["image_l"] = x_l
        sample["image_r"] = x_r
        sample["score_r"] = torch.tensor(score_r, dtype=torch.long)
        sample["score_c"] = torch.tensor(score_c, dtype=torch.long)
        return sample


def build_train_transforms(args, eval_meta: dict):
    """Build the training transform based on args.augment.

    If augmentation is disabled, returns (None, meta) so the caller can set
    train_tfms = eval_tfms explicitly.

    Returns:
        train_tfms_or_none: callable transform or None
        meta: dictionary describing the training policy
    """
    augment_level = str(getattr(args, "augment", "none")).lower().strip()
    if augment_level not in ("none", "light", "heavy"):
        augment_level = "none"

    if augment_level == "none":
        meta = {"train_policy": "deterministic (same as eval)", "augment": "none"}
        return None, meta

    preset = AUG_PRESETS[augment_level]
    aug = Augmentation(
        augment=True,
        ties=bool(getattr(args, "ties", True)),
        resize_short=int(eval_meta["resize_dim"]),
        out_size=int(eval_meta["target_crop"]),
        interpolation=_get_interp_mode(eval_meta["interpolation"]),
        mean=tuple(eval_meta["mean"]),
        std=tuple(eval_meta["std"]),
        **preset,
    )

    meta = {"train_policy": f"pairwise augmentation ({augment_level})", "augment": augment_level, "params": preset}
    return aug, meta
