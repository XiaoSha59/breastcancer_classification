"""
task3_multitask/train.py
Training and evaluation pipeline for Joint Multi-Task Learning (Segmentation + Classification).
Architecture: MultiTask ResNet18-UNet
Loss: Unified MultiTaskLoss from src.losses
"""

import os
import sys
import time
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from task3_multitask.dataset import get_multitask_dataloaders
from task3_multitask.model import get_multitask_model
from src.losses import MultiTaskLoss


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_total, running_cls, running_seg = 0.0, 0.0, 0.0
    for images, masks, labels in tqdm(loader, desc="Train MultiTask", leave=False):
        images, masks, labels = images.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        cls_logits, seg_logits = model(images)
        total_loss, loss_cls, loss_seg = criterion(cls_logits, labels, seg_logits, masks)
        total_loss.backward()
        optimizer.step()

        running_total += total_loss.item() * images.size(0)
        running_cls += loss_cls.item() * images.size(0)
        running_seg += loss_seg.item() * images.size(0)

    N = len(loader.dataset)
    return running_total / N, running_cls / N, running_seg / N


def evaluate(model, loader, criterion, device):
    model.eval()
    running_total, running_cls, running_seg = 0.0, 0.0, 0.0
    correct_cls, total_cls = 0, 0
    dices, ious = [], []

    with torch.no_grad():
        for images, masks, labels in tqdm(loader, desc="Val/Test MultiTask", leave=False):
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            cls_logits, seg_logits = model(images)
            total_loss, loss_cls, loss_seg = criterion(cls_logits, labels, seg_logits, masks)

            running_total += total_loss.item() * images.size(0)
            running_cls += loss_cls.item() * images.size(0)
            running_seg += loss_seg.item() * images.size(0)

            _, preds_cls = torch.max(cls_logits, 1)
            correct_cls += (preds_cls == labels).sum().item()
            total_cls += labels.size(0)

            for b in range(images.size(0)):
                probs = torch.sigmoid(seg_logits[b:b+1])
                bin_mask = (probs > 0.5).float()
                p_flat = bin_mask.view(-1)
                t_flat = masks[b:b+1].view(-1)
                inter = (p_flat * t_flat).sum().item()
                tot = p_flat.sum().item() + t_flat.sum().item()
                un = tot - inter
                d = (2.0 * inter + 1e-6) / (tot + 1e-6)
                j = (inter + 1e-6) / (un + 1e-6)
                dices.append(d)
                ious.append(j)

    N = len(loader.dataset)
    acc = 100.0 * correct_cls / total_cls if total_cls else 0.0
    mean_dice = (sum(dices) / len(dices)) * 100.0 if dices else 0.0
    mean_iou = (sum(ious) / len(ious)) * 100.0 if ious else 0.0
    return running_total / N, running_cls / N, running_seg / N, acc, mean_dice, mean_iou


def main():
    parser = argparse.ArgumentParser(description="Task 3: Joint Multi-Task Framework Training")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--weights_dir", type=str, default="weights")
    args = parser.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    train_loader, val_loader, test_loader = get_multitask_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    model = get_multitask_model(pretrained=True).to(device)
    criterion = MultiTaskLoss(seg_weight=args.seg_weight, cls_weight=args.cls_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_score = 0.0
    best_weights_path = os.path.join(args.weights_dir, "multitask_best.pth")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_tot, tr_cls, tr_seg = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_tot, val_cls, val_seg, val_acc, val_dice, val_iou = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Val Loss: {val_tot:.4f} | Acc: {val_acc:.2f}%, Dice: {val_dice:.2f}%, IoU: {val_iou:.2f}% | Time: {elapsed:.1f}s")

        val_score = val_acc + val_dice
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), best_weights_path)

    print(f"\n[INFO] Best Composite Val Score: {best_val_score:.2f}")
    if os.path.exists(best_weights_path):
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
    _, _, _, test_acc, test_dice, test_iou = evaluate(model, test_loader, criterion, device)
    print(f"[RESULT] Final Test Set Multi-Task (N=71) -> Cls Acc: {test_acc:.2f}%, Seg Dice: {test_dice:.2f}%, Seg IoU: {test_iou:.2f}%")


if __name__ == "__main__":
    main()
