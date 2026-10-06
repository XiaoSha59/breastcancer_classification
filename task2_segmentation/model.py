"""
segmentation/model_seg.py
ResNet18-UNet Architecture for Breast Ultrasound Tumor Segmentation.
Reference: Zhang et al. (2022) - "Fully automatic tumor segmentation of breast ultrasound images with deep learning".
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block: Upsample + Concat Skip Connection + Double Convolution.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample x to match spatial dimensions of skip connection
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet18UNet(nn.Module):
    """
    ResNet18-UNet: Pre-trained ResNet18 Encoder with U-Net Skip-Connection Decoder.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, pretrained: bool = True):
        super(ResNet18UNet, self).__init__()

        # 1. Encoder (ResNet18 Backbone)
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Encoder stages
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # [B, 64, 112, 112]
        self.enc1 = nn.Sequential(resnet.maxpool, resnet.layer1)          # [B, 64, 56, 56]
        self.enc2 = resnet.layer2                                         # [B, 128, 28, 28]
        self.enc3 = resnet.layer3                                         # [B, 256, 14, 14]
        self.enc4 = resnet.layer4                                         # [B, 512, 7, 7] (Bottleneck)

        # 2. Decoder with Skip Connections
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)  # 14x14
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)  # 28x28
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)    # 56x56
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)     # 112x112

        # 3. Final Output Segmentation Head
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]  # [224, 224]

        # Encoder Forward Pass
        e0 = self.enc0(x)   # 112x112
        e1 = self.enc1(e0)  # 56x56
        e2 = self.enc2(e1)  # 28x28
        e3 = self.enc3(e2)  # 14x14
        e4 = self.enc4(e3)  # 7x7

        # Decoder Forward Pass with Skip Connections
        d4 = self.dec4(e4, e3)  # 14x14
        d3 = self.dec3(d4, e2)  # 28x28
        d2 = self.dec2(d3, e1)  # 56x56
        d1 = self.dec1(d2, e0)  # 112x112

        # Final 224x224 Upsample and Output Projection
        out = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=True)
        out = self.final_conv(out)
        return out


def get_segmentation_model(pretrained: bool = True, weights_path: Optional[str] = None) -> nn.Module:
    """Build and initialize ResNet18-UNet model."""
    model = ResNet18UNet(in_channels=3, num_classes=1, pretrained=pretrained)

    if weights_path:
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"[INFO] Loaded segmentation weights from: {weights_path}")

    return model


if __name__ == "__main__":
    print("[INFO] Testing ResNet18-UNet Segmentation Architecture...")
    model = get_segmentation_model(pretrained=True)

    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_output = model(dummy_input)

    print(f"[TEST] Input shape  : {dummy_input.shape} (Expected: [2, 3, 224, 224])")
    print(f"[TEST] Output shape : {dummy_output.shape} (Expected: [2, 1, 224, 224])")
    print("[INFO] ResNet18-UNet forward pass test passed successfully.")
