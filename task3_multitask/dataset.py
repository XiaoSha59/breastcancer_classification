"""
segmentation/dataset_multitask.py
Multi-Task Dataset & DataLoader: Returns (Image, Class_Label, Segmentation_Mask).
Reference: Aumente-Maestro et al. (2025) & Zhang et al. (2022).
"""

import os
import glob
import random
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms

CLASSES = ["benign", "malignant", "normal"]


class BUSIMultiTaskDataset(Dataset):
    """
    Multi-Task Dataset returning:
    - image_tensor: [3, 224, 224]
    - label: int (0: benign, 1: malignant, 2: normal)
    - mask_tensor: [1, 224, 224]
    """
    def __init__(self, image_paths: List[str], labels: List[int], is_train: bool = True):
        self.image_paths = image_paths
        self.labels = labels
        self.is_train = is_train
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_combined_mask(self, img_path: str, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
        base_dir = os.path.dirname(img_path)
        base_stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_files = glob.glob(os.path.join(base_dir, f"{base_stem}_mask*.png"))

        if not mask_files:
            return Image.fromarray(np.zeros(target_size, dtype=np.uint8))

        combined = np.zeros(target_size, dtype=np.uint8)
        for mf in mask_files:
            m_img = Image.open(mf).convert("L").resize(target_size, Image.NEAREST)
            m_arr = np.array(m_img)
            combined = np.maximum(combined, m_arr)

        combined = (combined > 127).astype(np.uint8) * 255
        return Image.fromarray(combined)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        mask = self._load_combined_mask(img_path)

        # 1. Resize
        image = TF.resize(image, [224, 224], interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [224, 224], interpolation=transforms.InterpolationMode.NEAREST)

        # 2. Synchronized Data Augmentation
        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

        # 3. To Tensor
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=self.imagenet_mean, std=self.imagenet_std)

        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)

        return image_tensor, label, mask_tensor


def get_multitask_dataloaders(
    data_dir: str = "data",
    batch_size: int = 16,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build 3-way Multi-Task DataLoaders."""
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
    random.seed(random_seed)

    all_file_paths = []
    all_labels = []

    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            continue

        raw_files = sorted(glob.glob(os.path.join(cls_dir, "*.png")))
        valid_files = [f for f in raw_files if "_mask" not in os.path.basename(f).lower()]

        for f in valid_files:
            all_file_paths.append(f)
            all_labels.append(class_to_idx[cls])

    # 3-Way Stratified split
    class_groups: Dict[int, List[str]] = {}
    for p, l in zip(all_file_paths, all_labels):
        class_groups.setdefault(l, []).append(p)

    train_paths, val_paths, test_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for label, paths in sorted(class_groups.items()):
        shuffled = paths.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_paths.extend(shuffled[:n_train])
        train_labels.extend([label] * len(shuffled[:n_train]))

        val_paths.extend(shuffled[n_train:n_train + n_val])
        val_labels.extend([label] * len(shuffled[n_train:n_train + n_val]))

        test_paths.extend(shuffled[n_train + n_val:])
        test_labels.extend([label] * len(shuffled[n_train + n_val:]))

    train_dataset = BUSIMultiTaskDataset(train_paths, train_labels, is_train=True)
    val_dataset = BUSIMultiTaskDataset(val_paths, val_labels, is_train=False)
    test_dataset = BUSIMultiTaskDataset(test_paths, test_labels, is_train=False)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

    print(f"[INFO] Multi-Task Dataset Loaded: Total {len(all_file_paths)} curated samples")
    print(f"       -> Train samples : {len(train_dataset)} ({train_ratio*100:.0f}%)")
    print(f"       -> Val samples   : {len(val_dataset)} ({val_ratio*100:.0f}%)")
    print(f"       -> Test samples  : {len(test_dataset)} ({test_ratio*100:.0f}%)")

    return train_loader, val_loader, test_loader, CLASSES
