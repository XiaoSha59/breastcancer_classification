"""
segmentation/train_seg.py
Training and evaluation pipeline for Breast Ultrasound Tumor Segmentation.
Reference: Zhang et al. (2022) - "Fully automatic tumor segmentation of breast ultrasound images with deep learning".
"""

import os
import sys
import time
import argparse
from typing import Tuple, Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure root directory is accessible in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segmentation.dataset_seg import get_seg_dataloaders
from segmentation.model_seg import get_segmentation_model
from segmentation.losses import BCEDiceLoss, calculate_metrics


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    """Train segmentation model for one epoch."""
    model.train()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0

    pbar = tqdm(loader, desc="  [TRAIN SEG]", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        dice, iou = calculate_metrics(outputs, masks)
        total_dice += dice
        total_iou += iou
        batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice:.1f}%"})

    epoch_loss = running_loss / batches
    epoch_dice = total_dice / batches
    epoch_iou = total_iou / batches
    return epoch_loss, epoch_dice, epoch_iou


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "  [EVAL SEG]"
) -> Tuple[float, float, float]:
    """Evaluate segmentation model performance (Val or Test set)."""
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()
            dice, iou = calculate_metrics(outputs, masks)
            total_dice += dice
            total_iou += iou
            batches += 1

    total_loss = running_loss / batches
    total_dice = total_dice / batches
    total_iou = total_iou / batches
    return total_loss, total_dice, total_iou


def plot_seg_curves(history: Dict[str, List[float]], save_path: str = "weights/training_curves_seg.png") -> None:
    """Plot and save Loss and Dice Score curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train BCE+Dice Loss")
    plt.plot(epochs, history["val_loss"], "r-o", label="Val BCE+Dice Loss")
    plt.title("BCE + Dice Loss vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Dice Score Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_dice"], "b-o", label="Train Dice Score")
    plt.plot(epochs, history["val_dice"], "r-o", label="Val Dice Score")
    plt.title("Dice Score (%) vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Segmentation training curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet18-UNet on BUSI Segmentation Dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset classes")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory to save weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    args = parser.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("[INFO] RESNET18-UNET TUMOR SEGMENTATION CONFIGURATION")
    print("=" * 70)
    print(f" - Execution Device : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" - Architecture     : ResNet18-UNet (Skip-connection Decoder)")
    print(f" - Total Epochs     : {args.epochs}")
    print(f" - Batch Size       : {args.batch_size}")
    print(f" - Optimizer        : Adam (lr = {args.lr})")
    print(f" - Loss Function    : BCEDiceLoss (0.5 * BCE + 0.5 * Dice)")
    print("=" * 70 + "\n")

    # 1. Load Data (Train 70% / Val 15% / Test 15%)
    train_loader, val_loader, test_loader = get_seg_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # 2. Initialize Model
    model = get_segmentation_model(pretrained=True)
    model = model.to(device)

    # 3. Loss & Optimizer
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = 0.0
    best_model_path = os.path.join(args.weights_dir, "model_seg_best.pth")

    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    start_time = time.time()
    print("\n[INFO] Starting segmentation training loop...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device, desc="  [VAL SEG]")

        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["train_iou"].append(train_iou)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        epoch_duration = time.time() - epoch_start

        # Save checkpoint on highest validation Dice score
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_iou": val_iou
            }, best_model_path)
            marker = "--> [BEST MODEL SAVED]"
        else:
            marker = ""

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.2f}%, IoU: {train_iou:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.2f}%, IoU: {val_iou:.2f}% {marker}")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[SUMMARY] Segmentation training finished in {total_time / 60:.2f} minutes.")
    print(f"[SUMMARY] Best Validation Dice Score : {best_val_dice:.2f}%")
    print(f"[SUMMARY] Best weights saved at      : {best_model_path}")
    print("=" * 70)

    # 4. Save Curves
    plot_seg_curves(history, save_path=os.path.join(args.weights_dir, "training_curves_seg.png"))

    # 5. Final Evaluation on Independent Test Set
    print("\n" + "=" * 70)
    print("[INFO] Performing Final Evaluation on Unseen TEST SET...")
    print("=" * 70)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_dice, test_iou = evaluate(model, test_loader, criterion, device, desc="  [TEST SEG]")
    print(f"[TEST RESULT] Test BCE+Dice Loss : {test_loss:.4f}")
    print(f"[TEST RESULT] Test Dice Score    : {test_dice:.2f}%")
    print(f"[TEST RESULT] Test IoU Score     : {test_iou:.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
