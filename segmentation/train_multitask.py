"""
segmentation/train_multitask.py
Joint Multi-Task Training Pipeline (Simultaneous 3-class Classification + Tumor Segmentation).
Reference: Aumente-Maestro et al. (2025) - "A multi-task framework for breast cancer segmentation and classification in ultrasound imaging".
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segmentation.dataset_multitask import get_multitask_dataloaders
from segmentation.model_multitask import get_multitask_model
from segmentation.losses import BCEDiceLoss, calculate_metrics


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    cls_criterion: nn.Module,
    seg_criterion: nn.Module,
    optimizer: optim.Optimizer,
    seg_weight: float,
    device: torch.device
) -> Tuple[float, float, float]:
    """Train multi-task network for one epoch."""
    model.train()
    running_loss = 0.0
    correct_cls = 0
    total_samples = 0
    total_dice = 0.0
    batches = 0

    pbar = tqdm(loader, desc="  [TRAIN MULTI-TASK]", leave=False)
    for images, labels, masks in pbar:
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        cls_logits, seg_logits = model(images)

        loss_cls = cls_criterion(cls_logits, labels)
        loss_seg = seg_criterion(seg_logits, masks)
        total_loss = loss_cls + seg_weight * loss_seg

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        _, preds = torch.max(cls_logits, 1)
        correct_cls += torch.sum(preds == labels).item()
        total_samples += labels.size(0)

        dice, _ = calculate_metrics(seg_logits, masks)
        total_dice += dice
        batches += 1

        pbar.set_postfix({"loss": f"{total_loss.item():.4f}", "acc": f"{(correct_cls/total_samples)*100:.1f}%", "dice": f"{(total_dice/batches):.1f}%"})

    epoch_loss = running_loss / batches
    epoch_acc = (correct_cls / total_samples) * 100.0
    epoch_dice = total_dice / batches
    return epoch_loss, epoch_acc, epoch_dice


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    cls_criterion: nn.Module,
    seg_criterion: nn.Module,
    seg_weight: float,
    device: torch.device,
    desc: str = "  [EVAL MULTI-TASK]"
) -> Tuple[float, float, float, float]:
    """Evaluate multi-task performance."""
    model.eval()
    running_loss = 0.0
    correct_cls = 0
    total_samples = 0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for images, labels, masks in pbar:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            cls_logits, seg_logits = model(images)

            loss_cls = cls_criterion(cls_logits, labels)
            loss_seg = seg_criterion(seg_logits, masks)
            total_loss = loss_cls + seg_weight * loss_seg

            running_loss += total_loss.item()
            _, preds = torch.max(cls_logits, 1)
            correct_cls += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

            dice, iou = calculate_metrics(seg_logits, masks)
            total_dice += dice
            total_iou += iou
            batches += 1

    total_loss = running_loss / batches
    total_acc = (correct_cls / total_samples) * 100.0
    total_dice = total_dice / batches
    total_iou = total_iou / batches
    return total_loss, total_acc, total_dice, total_iou


def plot_multitask_curves(history: Dict[str, List[float]], save_path: str = "weights/training_curves_multitask.png") -> None:
    """Plot and save Loss, Classification Accuracy, and Segmentation Dice curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Total Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    axes[0].set_title("Total Multi-Task Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Classification Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Cls Acc")
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Cls Acc")
    axes[1].set_title("Classification Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Segmentation Dice Score
    axes[2].plot(epochs, history["train_dice"], "b-o", label="Train Seg Dice")
    axes[2].plot(epochs, history["val_dice"], "r-o", label="Val Seg Dice")
    axes[2].set_title("Segmentation Dice Score (%)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dice Score (%)")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Multi-task training curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Joint Multi-Task Training on BUSI Dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Weights directory")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--seg_weight", type=float, default=1.0, help="Weight for segmentation loss (default: 1.0)")
    args = parser.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("[INFO] MULTI-TASK JOINT TRAINING CONFIGURATION")
    print("=" * 70)
    print(f" - Execution Device   : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" - Architecture       : MultiTaskResNet18UNet")
    print(f" - Tasks              : 3-Class Classification + Binary Tumor Segmentation")
    print(f" - Total Epochs       : {args.epochs}")
    print(f" - Batch Size         : {args.batch_size}")
    print(f" - Optimizer          : Adam (lr = {args.lr})")
    print(f" - Combined Loss      : CrossEntropy + {args.seg_weight} * BCEDiceLoss")
    print("=" * 70 + "\n")

    # 1. Load Data
    train_loader, val_loader, test_loader, class_names = get_multitask_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # 2. Model
    model = get_multitask_model(pretrained=True)
    model = model.to(device)

    # 3. Criteria & Optimizer
    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_combined_score = 0.0
    best_model_path = os.path.join(args.weights_dir, "model_multitask_best.pth")

    history = {
        "train_loss": [], "train_acc": [], "train_dice": [],
        "val_loss": [], "val_acc": [], "val_dice": []
    }

    start_time = time.time()
    print("\n[INFO] Starting Multi-Task joint training loop...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_dice = train_one_epoch(
            model, train_loader, cls_criterion, seg_criterion, optimizer, args.seg_weight, device
        )
        val_loss, val_acc, val_dice, val_iou = evaluate(
            model, val_loader, cls_criterion, seg_criterion, args.seg_weight, device, desc="  [VAL MULTI-TASK]"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_dice"].append(val_dice)

        epoch_duration = time.time() - epoch_start

        # Combined metric for checkpoint selection: 0.5 * Accuracy + 0.5 * Dice
        combined_score = 0.5 * val_acc + 0.5 * val_dice
        is_best = combined_score > best_combined_score
        if is_best:
            best_combined_score = combined_score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_dice": val_dice,
                "class_names": class_names
            }, best_model_path)
            marker = "--> [BEST MODEL SAVED]"
        else:
            marker = ""

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) | "
              f"Loss: {val_loss:.4f} | "
              f"Cls Acc: {val_acc:.2f}% | "
              f"Seg Dice: {val_dice:.2f}% | "
              f"Combined: {combined_score:.2f}% {marker}")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[SUMMARY] Multi-Task training finished in {total_time / 60:.2f} minutes.")
    print(f"[SUMMARY] Best Multi-Task weights saved at: {best_model_path}")
    print("=" * 70)

    # 4. Save Curves
    plot_multitask_curves(history, save_path=os.path.join(args.weights_dir, "training_curves_multitask.png"))

    # 5. Final Evaluation on Test Set
    print("\n" + "=" * 70)
    print("[INFO] Final Evaluation on Independent TEST SET...")
    print("=" * 70)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_dice, test_iou = evaluate(
        model, test_loader, cls_criterion, seg_criterion, args.seg_weight, device, desc="  [TEST MULTI-TASK]"
    )
    print(f"[TEST RESULT] Test Multi-Task Loss   : {test_loss:.4f}")
    print(f"[TEST RESULT] Test Classification Acc: {test_acc:.2f}%")
    print(f"[TEST RESULT] Test Segmentation Dice : {test_dice:.2f}%")
    print(f"[TEST RESULT] Test Segmentation IoU  : {test_iou:.2f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
