"""
task3_multitask/model.py
Multi-Task Learning Architecture: Joint Tumor Segmentation and Pathological Classification.
Architecture: Shared ResNet18 Encoder + Expansive U-Net Decoder + GAP Classification Head
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from task2_segmentation.model import DecoderBlock


class MultiTaskResNet18UNet(nn.Module):
    """
    Joint Multi-Task Network for Breast Ultrasound Images:
    Performs simultaneous 3-class Classification and Pixel-level Tumor Segmentation.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 3, pretrained: bool = True):
        super(MultiTaskResNet18UNet, self).__init__()

        # 1. Shared Encoder (ResNet18 Backbone)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc2 = resnet.layer2
        self.enc3 = resnet.layer3
        self.enc4 = resnet.layer4  # Bottleneck

        # 2. Branch 1: Segmentation Decoder (U-Net)
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        # 3. Branch 2: Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_size = x.shape[2:]

        # Shared Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Branch 1: Segmentation
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        seg_out = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=True)
        seg_logits = self.seg_head(seg_out)

        # Branch 2: Classification
        pooled = self.gap(e4)
        cls_logits = self.cls_head(pooled)

        return cls_logits, seg_logits


def get_multitask_model(pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
    model = MultiTaskResNet18UNet(num_classes=3, pretrained=pretrained)
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    return model
