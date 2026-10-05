"""
train.py
Main training and evaluation pipeline for Breast Ultrasound Image (BUSI) Classification.
References: Zhang et al. (2022) & Aumente-Maestro et al. (2025).

Workflow:
1. Load 3-way Stratified DataLoaders (Train 70% / Val 15% / Test 15%) from src.dataset.
2. Initialize ResNet18 Transfer Learning model from src.model.
3. Train for 15 epochs using CrossEntropyLoss and Adam optimizer (lr=1e-4).
4. Track Validation Accuracy per epoch and save the best checkpoint to 'weights/model_best.pth'.
5. Plot Loss and Accuracy curves to 'weights/training_curves.png'.
6. Perform final unbiased evaluation on the unseen Test set.
"""

import os
import time
import argparse
from typing import Tuple, Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders
from src.model import get_resnet_model


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  [TRAIN]", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "  [EVAL]"
) -> Tuple[float, float, List[int], List[int]]:
    """Evaluate model performance on a DataLoader (Validation or Test set)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    total_loss = running_loss / total
    total_acc = (correct / total) * 100.0
    return total_loss, total_acc, all_preds, all_targets


def plot_training_curves(history: Dict[str, List[float]], save_path: str = "weights/training_curves.png") -> None:
    """Plot and save Loss and Accuracy curves across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], "r-o", label="Val Accuracy")
    plt.title("Accuracy (%) vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Training curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 on Curated BUSI Dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing dataset classes (default: 'data')")
    parser.add_argument("--weights_dir", type=str, default="weights", help="Directory to save checkpoint weights (default: 'weights')")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer (default: 1e-4)")
    args = parser.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("[INFO] TRAINING CONFIGURATION")
    print("=" * 70)
    print(f" - Execution Device : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" - Model Backbone    : ResNet18 (ImageNet Pre-trained)")
    print(f" - Total Epochs      : {args.epochs}")
    print(f" - Batch Size        : {args.batch_size}")
    print(f" - Optimizer         : Adam (Learning Rate = {args.lr})")
    print(f" - Loss Function     : CrossEntropyLoss")
    print("=" * 70 + "\n")

    # 1. Load Data (70% Train / 15% Val / 15% Test)
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # 2. Initialize Model
    model = get_resnet_model(num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_model_path = os.path.join(args.weights_dir, "model_best.pth")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    start_time = time.time()
    print("\n[INFO] Starting training loop...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, desc="  [VAL]")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_duration = time.time() - epoch_start

        # Checkpoint saving on best validation accuracy
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names
            }, best_model_path)
            marker = "--> [BEST MODEL SAVED]"
        else:
            marker = ""

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% {marker}")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[SUMMARY] Training finished in {total_time / 60:.2f} minutes.")
    print(f"[SUMMARY] Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"[SUMMARY] Best weights saved at   : {best_model_path}")
    print("=" * 70)

    # 4. Save Training Curves
    plot_training_curves(history, save_path=os.path.join(args.weights_dir, "training_curves.png"))

    # 5. Final Evaluation on Independent Test Set
    print("\n" + "=" * 70)
    print("[INFO] Performing Final Evaluation on Unseen TEST SET...")
    print("=" * 70)
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device, desc="  [TEST]")
    print(f"[TEST RESULT] Test Loss     : {test_loss:.4f}")
    print(f"[TEST RESULT] Test Accuracy : {test_acc:.2f}%\n")

    # Per-Class Accuracy Report
    for idx, cls in enumerate(class_names):
        cls_total = sum(1 for t in test_targets if t == idx)
        cls_correct = sum(1 for p, t in zip(test_preds, test_targets) if p == idx and t == idx)
        cls_acc = (cls_correct / cls_total * 100.0) if cls_total > 0 else 0.0
        print(f" - Class '{cls:<10}': {cls_correct:2d}/{cls_total:2d} correct ({cls_acc:6.2f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
