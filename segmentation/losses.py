"""
segmentation/losses.py
Loss functions and performance evaluation metrics for Breast Ultrasound Tumor Segmentation.
Reference: Zhang et al. (2022) - "Fully automatic tumor segmentation of breast ultrasound images with deep learning".
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary tumor segmentation.
    Loss = 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    """
    def __init__(self, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice Loss for robust medical image segmentation.
    Loss = BCE(logits, targets) + DiceLoss(logits, targets)
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


def calculate_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5
) -> Tuple[float, float]:
    """
    Calculate Dice Score (F1-score) and IoU (Jaccard Index) for binary segmentation.
    
    Args:
        logits (torch.Tensor): Raw model output tensor [B, 1, H, W].
        targets (torch.Tensor): Ground-truth binary mask tensor [B, 1, H, W].
        threshold (float): Binarization probability threshold (default: 0.5).
        smooth (float): Small epsilon to prevent division by zero.

    Returns:
        Tuple[float, float]: (dice_score, iou_score) in percentage [0, 100].
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        intersection = (preds_flat * targets_flat).sum().item()
        total_pred = preds_flat.sum().item()
        total_target = targets_flat.sum().item()

        # Dice Score: 2 * |A ∩ B| / (|A| + |B|)
        dice = (2.0 * intersection + smooth) / (total_pred + total_target + smooth)

        # IoU (Jaccard): |A ∩ B| / |A ∪ B| = |A ∩ B| / (|A| + |B| - |A ∩ B|)
        union = total_pred + total_target - intersection
        iou = (intersection + smooth) / (union + smooth)

    return dice * 100.0, iou * 100.0


if __name__ == "__main__":
    print("[INFO] Testing BCEDiceLoss and Segmentation Metrics...")
    criterion = BCEDiceLoss()

    dummy_logits = torch.randn(2, 1, 224, 224)
    dummy_targets = torch.randint(0, 2, (2, 1, 224, 224)).float()

    loss = criterion(dummy_logits, dummy_targets)
    dice, iou = calculate_metrics(dummy_logits, dummy_targets)

    print(f"[TEST] Combined BCE+Dice Loss : {loss.item():.4f}")
    print(f"[TEST] Calculated Dice Score  : {dice:.2f}%")
    print(f"[TEST] Calculated IoU Score   : {iou:.2f}%")
    print("[INFO] Losses and metrics test passed successfully.")
