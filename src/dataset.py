"""
src/dataset.py
Dataset management and 3-way Stratified DataLoader pipeline (Train / Val / Test).
Reference for Medical Data Augmentations: Zhang et al. (2022) - "Fully automatic tumor segmentation of breast ultrasound images with deep learning".
"""

import os
import glob
import random
from pathlib import Path
from typing import Tuple, List, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CLASSES = ["benign", "malignant", "normal"]


class BUSIDataset(Dataset):
    """
    Custom PyTorch Dataset for Breast Ultrasound Images.
    Loads RGB images and applies respective data transformations.
    """
    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.file_paths[idx]
        label = self.labels[idx]

        # Convert to 3-channel RGB for ResNet compatibility
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Construct training (with medical augmentations) and evaluation transforms.
    Standardized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    return train_transform, eval_transform


def stratified_split_3way(
    file_paths: List[str],
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
    """
    Split dataset into 3 sets (Train / Val / Test) while preserving class distribution.
    Implemented in pure Python for maximum portability and zero extra dependencies.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"

    random.seed(random_seed)
    class_groups: Dict[int, List[str]] = {}

    for path, label in zip(file_paths, labels):
        class_groups.setdefault(label, []).append(path)

    train_paths, val_paths, test_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for label, paths in sorted(class_groups.items()):
        shuffled = paths.copy()
        random.shuffle(shuffled)
        n = len(shuffled)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_subset = shuffled[:n_train]
        val_subset = shuffled[n_train:n_train + n_val]
        test_subset = shuffled[n_train + n_val:]

        train_paths.extend(train_subset)
        train_labels.extend([label] * len(train_subset))

        val_paths.extend(val_subset)
        val_labels.extend([label] * len(val_subset))

        test_paths.extend(test_subset)
        test_labels.extend([label] * len(test_subset))

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build 3-way DataLoaders (Train, Val, Test) from data_dir.
    
    Args:
        data_dir (str): Root directory containing 'benign', 'malignant', and 'normal' subdirectories.
        batch_size (int): Mini-batch size.
        train_ratio (float): Proportion of samples for training (default: 0.70).
        val_ratio (float): Proportion of samples for validation (default: 0.15).
        test_ratio (float): Proportion of samples for testing (default: 0.15).
        random_seed (int): Seed for reproducibility.
        num_workers (int): DataLoader subprocess count.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, List[str]]: (train_loader, val_loader, test_loader, class_names)
    """
    class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

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

    if len(all_file_paths) == 0:
        raise ValueError(f"[ERROR] No valid ultrasound images found in '{data_dir}'. Please verify the path.")

    # 3-Way Stratified Split (Train 70% / Val 15% / Test 15%)
    (
        train_paths, val_paths, test_paths,
        train_labels, val_labels, test_labels
    ) = stratified_split_3way(
        all_file_paths,
        all_labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )

    train_transform, eval_transform = get_transforms()

    train_dataset = BUSIDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = BUSIDataset(val_paths, val_labels, transform=eval_transform)
    test_dataset = BUSIDataset(test_paths, test_labels, transform=eval_transform)

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

    print(f"[INFO] Dataset Loaded: Total {len(all_file_paths)} curated images")
    print(f"       -> Training set   : {len(train_dataset)} images ({train_ratio*100:.0f}%)")
    print(f"       -> Validation set : {len(val_dataset)} images ({val_ratio*100:.0f}%)")
    print(f"       -> Test set       : {len(test_dataset)} images ({test_ratio*100:.0f}%)")
    print(f"       -> Class mapping  : {class_to_idx}")

    return train_loader, val_loader, test_loader, CLASSES


if __name__ == "__main__":
    print("[INFO] Testing 3-Way Stratified BUSI Dataset and DataLoader...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(data_dir="data", batch_size=16)

    # Test loading one batch from train_loader
    images, labels = next(iter(train_loader))
    print(f"[TEST] Train batch images shape: {images.shape} (Expected: [16, 3, 224, 224])")
    print(f"[TEST] Train batch labels shape: {labels.shape} (Expected: [16])")
    print(f"[TEST] Train batch sample labels: {labels.tolist()}")
    print("[INFO] 3-Way Dataset pipeline test passed successfully.")
