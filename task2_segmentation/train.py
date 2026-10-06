"""
task2_segmentation/train.py
Training and evaluation pipeline for Single-Task Tumor Segmentation.
Architecture: ResNet18-UNet
Target: Pixel-level tumor boundary delineation
Loss: Combined BCE + Dice Loss from src.losses
"""

import os
import sys
import time
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from task2_segmentation.dataset import get_seg_dataloaders
from task2_segmentation.model import get_segmentation_model
from src.losses import BCEDiceLoss


def compute_batch_dice_iou(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6):
    probs = torch.sigmoid(preds)
    preds_bin = (probs > 0.5).float()

    preds_flat = preds_bin.view(-1)
    targets_flat = targets.view(-1)

    intersection = (preds_flat * targets_flat).sum().item()
    total = preds_flat.sum().item() + targets_flat.sum().item()
    union = total - intersection

    dice = (2.0 * intersection + smooth) / (total + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader, desc="Train Seg", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dices, ious = [], []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Val/Test Seg", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            for b in range(images.size(0)):
                d, j = compute_batch_dice_iou(outputs[b:b+1], masks[b:b+1])
                dices.append(d)
                ious.append(j)

    mean_dice = sum(dices) / len(dices) if dices else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return running_loss / len(loader.dataset), mean_dice * 100.0, mean_iou * 100.0


def main():
    parser = argparse.ArgumentParser(description="Task 2: Single-Task Segmentation Training")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weights_dir", type=str, default="weights")
    args = parser.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    train_loader, val_loader, test_loader = get_seg_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    model = get_segmentation_model(pretrained=True).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_dice = 0.0
    best_weights_path = os.path.join(args.weights_dir, "seg_best.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}, Dice: {val_dice:.2f}%, IoU: {val_iou:.2f}% | Time: {elapsed:.1f}s")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_weights_path)

    print(f"\n[INFO] Best Validation Dice: {best_val_dice:.2f}%")
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_loss, test_dice, test_iou = evaluate(model, test_loader, criterion, device)
    print(f"[RESULT] Final Test Set Segmentation (N=71) -> Dice: {test_dice:.2f}%, IoU: {test_iou:.2f}%")


if __name__ == "__main__":
    main()
