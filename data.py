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


class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image_l, image_r = sample['image_l'], sample['image_r']

        return {
            'image_l': self.transform(image_l),
            'image_r': self.transform(image_r),

            'score_r': sample['score_r'],
            'score_c': sample['score_c'],

            'image_l_name': sample['image_l_name'],
            'image_r_name': sample['image_r_name'],

            # keep the new fields in the batch after transforms
            'has_eyetracker': sample['has_eyetracker'],
            'gaze_l': sample['gaze_l'],
            'gaze_r': sample['gaze_r'],
            #'survey_id': sample['survey_id'],
            #'trial_id': sample['trial_id'],
        }


# =============================================================================
# PairwiseAugmentationPipeline
# =============================================================================
# This transform operates on a *pair* of images (left, right) representing a
# single comparison, and applies *paired* augmentations so that both images
# experience the same geometric/photometric changes.
#
# Core rules:
#   - All "paired" operations apply the same parameters to BOTH images.
#   - If images are swapped, the label is updated accordingly.
#   - When gaze supervision is active, we can optionally disable geometry/color
#     augmentations to avoid misalignment with gaze maps.
# =============================================================================


class PairwiseAugmentationPipeline:
    """
    Paired augmentation pipeline for pairwise comparisons.

    Expected input sample (dict):
        sample = {
            "image_l": PIL.Image or torch.Tensor [C,H,W],
            "image_r": PIL.Image or torch.Tensor [C,H,W],
            "score_r": int or tensor scalar in {-1, 0, +1},
            "score_c": int or tensor scalar (classification label encoding),
            "has_eyetracker": bool or tensor scalar (optional),
            "gaze_l": torch.Tensor (optional; typically [H,W] or [1,H,W]),
            "gaze_r": torch.Tensor (optional)
        }

    Output:
        Same dict with:
            - "image_l", "image_r" converted to float tensors [C,H,W] in [0,1]
            - labels updated if swap occurs
            - gaze tensors passed through (and flipped/swapped if required)

    Notes on design:
        - All geometry is applied consistently across the pair.
        - Sky-removal is implemented as a bottom-anchored band crop.
        - Random erasing is applied in tensor space to keep it simple and fast.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(
        self,
        augment: bool = True,
        ties: bool = True,

        # Gaze behavior:
        # If gaze is active, you may want to disable augmentations that would
        # invalidate spatial correspondence between image and gaze map.
        disable_aug_when_gaze: bool = True,
        allow_swap_when_gaze: bool = False,

        # Paired invariance ops
        hflip_p: float = 0.5,
        swap_p: float = 0.5,

        # Paired photometric ops
        color_jitter_p: float = 0.15,
        jitter_brightness: float = 0.30,
        jitter_contrast: float = 0.30,
        jitter_saturation: float = 0.30,
        jitter_hue: float = 0.05,
        gray_p: float = 0.05,

        # Paired geometry: bottom-band crop (sky removal)
        bottom_crop_p: float = 1.0,
        bottom_keep_h: tuple = (0.65, 0.85),     # keep bottom 65%..85% of height
        bottom_x_jitter_frac: float = 0.02,      # small horizontal jitter

        # Paired random erasing (tensor space)
        erase_p: float = 0.40,
        erase_scale: tuple = (0.05, 0.08),
        erase_ratio: tuple = (0.3, 3.3),
        erase_value: float = 0.0,

        # Final output geometry
        resize_short: int = 256,   # resize short side first
        out_size: int = 224,       # final square size
    ):
        self.augment = augment
        self.ties = ties

        self.disable_aug_when_gaze = disable_aug_when_gaze
        self.allow_swap_when_gaze = allow_swap_when_gaze

        self.hflip_p = hflip_p
        self.swap_p = swap_p

        self.color_jitter_p = color_jitter_p
        self.jitter_brightness = jitter_brightness
        self.jitter_contrast = jitter_contrast
        self.jitter_saturation = jitter_saturation
        self.jitter_hue = jitter_hue
        self.gray_p = gray_p

        self.bottom_crop_p = bottom_crop_p
        self.bottom_keep_h = bottom_keep_h
        self.bottom_x_jitter_frac = bottom_x_jitter_frac

        self.erase_p = erase_p
        self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio
        self.erase_value = erase_value

        self.resize_short = resize_short
        self.out_size = out_size

    # -------------------------------------------------------------------------
    # Label encoding helper
    # -------------------------------------------------------------------------
    def _score_r_to_score_c(self, score_r: int) -> int:
        """
        Convert ranking label score_r in {-1,0,+1} to classification index.

        If ties are enabled (3-class):
            -1 -> 0  (left wins)
             0 -> 1  (tie)
            +1 -> 2  (right wins)

        If ties are disabled (2-class):
            -1 -> 0
            +1 -> 1
        """
        if self.ties:
            if score_r == -1:
                return 0
            if score_r == 0:
                return 1
            return 2
        # ties disabled
        return 0 if score_r == -1 else 1

    # -------------------------------------------------------------------------
    # PIL geometry helpers
    # -------------------------------------------------------------------------
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

    def _sample_bottom_band_crop(self, w: int, h: int):
        """
        Sample a bottom-anchored crop that removes a portion of the top.

        Returns:
            x0, y0, cw, ch

        Behavior:
            - Choose a kept height fraction in `bottom_keep_h`.
            - Anchor crop at bottom: y0 = h - ch.
            - Keep (almost) full width, with small left/right jitter.
        """
        keep_frac = random.uniform(*self.bottom_keep_h)
        ch = max(1, min(h, int(round(h * keep_frac))))
        y0 = h - ch  # anchor at bottom

        jitter_px = int(round(w * self.bottom_x_jitter_frac))
        left_j = random.randint(0, max(0, jitter_px)) if jitter_px > 0 else 0
        right_j = random.randint(0, max(0, jitter_px)) if jitter_px > 0 else 0

        x0 = left_j
        cw = max(1, w - left_j - right_j)

        return x0, y0, cw, ch

    # -------------------------------------------------------------------------
    # PIL photometric helpers (paired)
    # -------------------------------------------------------------------------
    def _apply_paired_color_jitter(self, im_l: Image.Image, im_r: Image.Image):
        """
        Apply the same brightness/contrast/saturation/hue perturbation to both images.
        Implemented using PIL ImageEnhance + HSV hue shift.
        """
        b = 1.0 + random.uniform(-self.jitter_brightness, self.jitter_brightness)
        c = 1.0 + random.uniform(-self.jitter_contrast, self.jitter_contrast)
        s = 1.0 + random.uniform(-self.jitter_saturation, self.jitter_saturation)
        h_shift = random.uniform(-self.jitter_hue, self.jitter_hue)

        def jitter_one(im: Image.Image) -> Image.Image:
            im = ImageEnhance.Brightness(im).enhance(b)
            im = ImageEnhance.Contrast(im).enhance(c)
            im = ImageEnhance.Color(im).enhance(s)

            # Hue shift via HSV conversion
            hsv = np.array(im.convert("HSV"), dtype=np.uint8)
            hsv[..., 0] = (hsv[..., 0].astype(int) + int(h_shift * 255)) % 256
            return Image.fromarray(hsv, mode="HSV").convert("RGB")

        return jitter_one(im_l), jitter_one(im_r)

    # -------------------------------------------------------------------------
    # Tensor random erasing helpers (paired)
    # -------------------------------------------------------------------------
    def _sample_erasing_rect(self, H: int, W: int):
        """
        Sample an erasing rectangle (top, left, height, width) in tensor space.
        Returns None if no valid rectangle found.
        """
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
        """
        Apply the same erasing rectangle to both tensors x_l and x_r.
        Expects tensors of shape [C,H,W] with float dtype.
        """
        _, H, W = x_l.shape
        rect = self._sample_erasing_rect(H, W)
        if rect is None:
            return x_l, x_r

        top, left, eh, ew = rect
        x_l[:, top:top + eh, left:left + ew] = self.erase_value
        x_r[:, top:top + eh, left:left + ew] = self.erase_value
        return x_l, x_r

    # -------------------------------------------------------------------------
    # Main call
    # -------------------------------------------------------------------------
    def __call__(self, sample: dict) -> dict:
        # --- Extract required fields ---
        image_l = sample["image_l"]
        image_r = sample["image_r"]

        # score_r and score_c can arrive as python ints or tensors
        score_r = int(sample["score_r"].item()) if torch.is_tensor(sample["score_r"]) else int(sample["score_r"])
        score_c = int(sample["score_c"].item()) if torch.is_tensor(sample["score_c"]) else int(sample["score_c"])

        # Optional gaze fields
        has_eyetracker = sample.get("has_eyetracker", False)
        gaze_active = bool(has_eyetracker.item()) if torch.is_tensor(has_eyetracker) else bool(has_eyetracker)
        gaze_l = sample.get("gaze_l", None)
        gaze_r = sample.get("gaze_r", None)

        # --- Determine whether to augment this sample ---
        do_aug = self.augment
        if gaze_active and self.disable_aug_when_gaze:
            do_aug = False

        # ---------------------------------------------------------------------
        # PIL branch (common case): geometry + photometric in PIL space
        # ---------------------------------------------------------------------
        pil_inputs = (not torch.is_tensor(image_l)) and (not torch.is_tensor(image_r))
        if pil_inputs:
            # 1) Resize short side to a standard size so crop is meaningful
            image_l = self._resize_short_side(image_l, self.resize_short)
            image_r = self._resize_short_side(image_r, self.resize_short)

            if do_aug:
                # 2) Paired horizontal flip
                if random.random() < self.hflip_p:
                    image_l = TF.hflip(image_l)
                    image_r = TF.hflip(image_r)
                    # If gaze is present and we ever decide to allow flip with gaze,
                    # the gaze maps must also be flipped horizontally.
                    if gaze_active and gaze_l is not None and gaze_r is not None:
                        gaze_l = torch.flip(gaze_l, dims=[-1])
                        gaze_r = torch.flip(gaze_r, dims=[-1])

                # 3) Paired bottom-band crop (sky removal)
                if random.random() < self.bottom_crop_p:
                    w, h = image_l.size
                    x0, y0, cw, ch = self._sample_bottom_band_crop(w, h)
                    image_l = TF.resized_crop(image_l, y0, x0, ch, cw, (self.out_size, self.out_size))
                    image_r = TF.resized_crop(image_r, y0, x0, ch, cw, (self.out_size, self.out_size))
                else:
                    # If crop is not applied, still enforce final size
                    image_l = TF.resize(image_l, (self.out_size, self.out_size))
                    image_r = TF.resize(image_r, (self.out_size, self.out_size))

                # 4) Paired color jitter
                if random.random() < self.color_jitter_p:
                    image_l, image_r = self._apply_paired_color_jitter(image_l, image_r)

                # 5) Paired grayscale
                if random.random() < self.gray_p:
                    image_l = TF.to_grayscale(image_l, num_output_channels=3)
                    image_r = TF.to_grayscale(image_r, num_output_channels=3)

                # 6) Optional swap (paired): swaps images and flips ranking label
                if random.random() < self.swap_p:
                    image_l, image_r = image_r, image_l
                    if gaze_active and gaze_l is not None and gaze_r is not None:
                        gaze_l, gaze_r = gaze_r, gaze_l

                    score_r = -score_r
                    score_c = self._score_r_to_score_c(score_r)

            else:
                # No augmentation: just normalize geometry
                image_l = TF.resize(image_l, (self.out_size, self.out_size))
                image_r = TF.resize(image_r, (self.out_size, self.out_size))

                # If gaze is active, optionally allow swap only (no geometry/color)
                if gaze_active and self.allow_swap_when_gaze and (random.random() < self.swap_p):
                    image_l, image_r = image_r, image_l
                    if gaze_l is not None and gaze_r is not None:
                        gaze_l, gaze_r = gaze_r, gaze_l

                    score_r = -score_r
                    score_c = self._score_r_to_score_c(score_r)

            # Convert to tensor in [0,1]
            x_l = TF.to_tensor(image_l)
            x_r = TF.to_tensor(image_r)

        # ---------------------------------------------------------------------
        # Tensor branch: if upstream already returned tensors
        # ---------------------------------------------------------------------
        else:
            x_l = image_l if torch.is_tensor(image_l) else TF.to_tensor(image_l)
            x_r = image_r if torch.is_tensor(image_r) else TF.to_tensor(image_r)

        # ---------------------------------------------------------------------
        # Tensor-only augmentation (paired): random erasing
        # Only apply when augmenting and (typically) when gaze is not active.
        # ---------------------------------------------------------------------
        if do_aug and (not gaze_active) and (random.random() < self.erase_p):
            x_l, x_r = self._paired_erase(x_l, x_r)

        # --- Write back into sample dict ---
        sample["image_l"] = x_l
        sample["image_r"] = x_r
        sample["score_r"] = torch.tensor(score_r, dtype=torch.long)
        sample["score_c"] = torch.tensor(score_c, dtype=torch.long)

        if gaze_active and gaze_l is not None and gaze_r is not None:
            sample["gaze_l"] = gaze_l
            sample["gaze_r"] = gaze_r

        return sample
    