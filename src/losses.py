"""
src/losses.py
Centralized loss functions for classification, segmentation, and multi-task learning.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""
    def __init__(self, smooth: float = 1.0):
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
    """Combined Binary Cross-Entropy and Dice Loss for tumor segmentation."""
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class MultiTaskLoss(nn.Module):
    """
    Joint Loss function for simultaneous classification and segmentation:
    Loss_total = seg_weight * Loss_seg + cls_weight * Loss_cls
    """
    def __init__(
        self,
        seg_weight: float = 1.0,
        cls_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super(MultiTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor
    ):
        loss_cls = self.cls_loss_fn(cls_logits, cls_targets)
        loss_seg = self.seg_loss_fn(seg_logits, seg_targets)
        total_loss = self.seg_weight * loss_seg + self.cls_weight * loss_cls
        return total_loss, loss_cls, loss_seg
