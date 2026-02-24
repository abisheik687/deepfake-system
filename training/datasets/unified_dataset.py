"""
DeepShield AI — Unified Deepfake Dataset

Merges multiple deepfake datasets into a single PyTorch Dataset.
Data must be pre-processed via:  python training/preprocessing/extract_faces.py

Expected directory structure:
  data_dir/
  ├── ff++/
  │   ├── real/   (frame images or face crops)
  │   └── fake/
  ├── celeb_df/
  │   ├── real/
  │   └── fake/
  ├── wild/
  │   ├── real/
  │   └── fake/
  └── dfdc/
      ├── real/
      └── fake/
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Augmentation pipelines ───────────────────────────────────────────────────

def make_train_transform(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
        ], p=0.4),
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, p=1.0),
            A.ElasticTransform(p=1.0),
        ], p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
        A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def make_val_transform(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─── Dataset ─────────────────────────────────────────────────────────────────

class UnifiedDeepfakeDataset(Dataset):
    """
    PyTorch dataset that merges multiple deepfake datasets.

    Labels: 0 = REAL, 1 = FAKE

    Parameters
    ----------
    data_dir   : root directory containing dataset subdirs
    datasets   : list of dataset names to include (e.g. ["ff++", "celeb_df"])
    image_size : resize all images to this square size
    augment    : apply training augumentations (True for train, False for val)
    max_per_ds : cap per dataset to balance classes (None = use all)
    """

    DATASET_DIRS = {
        "ff++":     "ff++",
        "celeb_df": "celeb_df",
        "dfdc":     "dfdc",
        "wild":     "wild",
        "timit":    "timit",
    }
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        data_dir:   str,
        datasets:   List[str],
        image_size: int  = 224,
        augment:    bool = True,
        max_per_ds: Optional[int] = None,
    ):
        self.data_dir   = Path(data_dir)
        self.image_size = image_size
        self.augment    = augment
        self.train_tf   = make_train_transform(image_size)
        self.val_tf     = make_val_transform(image_size)

        self.samples: List[Tuple[Path, int]] = []

        for ds_name in datasets:
            ds_dir_name = self.DATASET_DIRS.get(ds_name, ds_name)
            ds_dir      = self.data_dir / ds_dir_name

            if not ds_dir.exists():
                print(f"[Dataset] WARNING: {ds_dir} not found — skipping '{ds_name}'")
                continue

            real_files = self._scan(ds_dir / "real", max_per_ds)
            fake_files = self._scan(ds_dir / "fake", max_per_ds)

            # Balance classes within each dataset
            min_count  = min(len(real_files), len(fake_files))
            real_files = real_files[:min_count]
            fake_files = fake_files[:min_count]

            self.samples.extend([(f, 0) for f in real_files])
            self.samples.extend([(f, 1) for f in fake_files])

            print(f"[Dataset] {ds_name}: {min_count} real + {min_count} fake = {2*min_count} samples")

        if not self.samples:
            print("[Dataset] WARNING: No samples found. "
                  "Run preprocessing first: python training/preprocessing/extract_faces.py")
        else:
            print(f"[Dataset] TOTAL: {len(self.samples)} samples "
                  f"({sum(1 for _,l in self.samples if l==0)} real / "
                  f"{sum(1 for _,l in self.samples if l==1)} fake)")

    def _scan(self, directory: Path, limit: Optional[int]) -> List[Path]:
        if not directory.exists():
            return []
        files = [f for f in directory.rglob("*") if f.suffix.lower() in self.VALID_EXTS]
        if limit:
            files = files[:limit]
        return files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            # Fallback to a blank image if file corrupt
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tf  = self.train_tf if self.augment else self.val_tf
        out = tf(image=img)["image"]
        return out, label
