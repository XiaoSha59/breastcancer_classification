"""
segmentation/dataset_seg.py
Dataset and DataLoader pipeline for Breast Ultrasound Tumor Segmentation (Image + Mask).
Reference for Data Augmentation: Zhang et al. (2022) - "Fully automatic tumor segmentation of breast ultrasound images with deep learning".
"""

import os
import glob
import random
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms

CLASSES = ["benign", "malignant", "normal"]


class BUSISegDataset(Dataset):
    """
    Dataset for Breast Ultrasound Segmentation.
    Loads paired (Image, Binary Mask) and applies synchronized spatial augmentations.
    """
    def __init__(self, image_paths: List[str], is_train: bool = True):
        self.image_paths = image_paths
        self.is_train = is_train
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_combined_mask(self, img_path: str, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """Find and combine all associated mask files (*_mask.png, *_mask_1.png) for an image."""
        base_dir = os.path.dirname(img_path)
        base_stem = os.path.splitext(os.path.basename(img_path))[0]

        mask_files = glob.glob(os.path.join(base_dir, f"{base_stem}_mask*.png"))

        if not mask_files:
            # For 'normal' class or cases with no lesions, return blank black mask
            return Image.fromarray(np.zeros(target_size, dtype=np.uint8))

        combined = np.zeros(target_size, dtype=np.uint8)
        for mf in mask_files:
            m_img = Image.open(mf).convert("L").resize(target_size, Image.NEAREST)
            m_arr = np.array(m_img)
            combined = np.maximum(combined, m_arr)

        # Binarize mask to 0 or 255
        combined = (combined > 127).astype(np.uint8) * 255
        return Image.fromarray(combined)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]

        # Load RGB Image and grayscale Mask
        image = Image.open(img_path).convert("RGB")
        mask = self._load_combined_mask(img_path)

        # 1. Resize both image and mask
        image = TF.resize(image, [224, 224], interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [224, 224], interpolation=transforms.InterpolationMode.NEAREST)

        # 2. Synchronized Data Augmentation for training
        if self.is_train:
            # Synchronized Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Synchronized Random Rotation (-15 to +15 degrees)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

        # 3. Transform to Tensor and Normalize
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=self.imagenet_mean, std=self.imagenet_std)

        # Mask Tensor [1, 224, 224] with values in {0.0, 1.0}
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)  # Shape: [1, 224, 224]

        return image_tensor, mask_tensor


def get_seg_dataloaders(
    data_dir: str = "data",
    batch_size: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build 3-way paired (Image, Mask) DataLoaders for Segmentation.
    """
    random.seed(random_seed)
    all_image_paths = []
    class_groups: Dict[str, List[str]] = {}

    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            continue

        raw_files = sorted(glob.glob(os.path.join(cls_dir, "*.png")))
        valid_files = [f for f in raw_files if "_mask" not in os.path.basename(f).lower()]
        class_groups[cls] = valid_files

    train_paths, val_paths, test_paths = [], [], []

    for cls, paths in sorted(class_groups.items()):
        shuffled = paths.copy()
        random.shuffle(shuffled)
        n = len(shuffled)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_paths.extend(shuffled[:n_train])
        val_paths.extend(shuffled[n_train:n_train + n_val])
        test_paths.extend(shuffled[n_train + n_val:])

    train_dataset = BUSISegDataset(train_paths, is_train=True)
    val_dataset = BUSISegDataset(val_paths, is_train=False)
    test_dataset = BUSISegDataset(test_paths, is_train=False)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    print(f"[INFO] Segmentation Dataset Loaded: Total {len(train_paths) + len(val_paths) + len(test_paths)} curated pairs")
    print(f"       -> Train pairs : {len(train_dataset)} ({train_ratio*100:.0f}%)")
    print(f"       -> Val pairs   : {len(val_dataset)} ({val_ratio*100:.0f}%)")
    print(f"       -> Test pairs  : {len(test_dataset)} ({test_ratio*100:.0f}%)")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("[INFO] Testing Segmentation Dataset and DataLoaders...")
    train_loader, val_loader, test_loader = get_seg_dataloaders(data_dir="data", batch_size=8)

    images, masks = next(iter(train_loader))
    print(f"[TEST] Batch Images Tensor Shape : {images.shape} (Expected: [8, 3, 224, 224])")
    print(f"[TEST] Batch Masks Tensor Shape  : {masks.shape}  (Expected: [8, 1, 224, 224])")
    print(f"[TEST] Mask Value Range          : [{masks.min().item():.1f}, {masks.max().item():.1f}]")
    print("[INFO] Segmentation Dataset test passed successfully.")
