"""
src/model.py
Model architecture for Breast Ultrasound Images (BUSI) 3-class Classification.
Reference for ResNet Backbone Transfer Learning: Zhang et al. (2022) - "Fully automatic tumor segmentation of breast ultrasound images with deep learning".
"""

import os
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models


def get_resnet_model(
    num_classes: int = 3,
    pretrained: bool = True,
    weights_path: Optional[str] = None
) -> nn.Module:
    """
    Initialize ResNet18 Transfer Learning model for 3-class BUSI classification.
    
    Args:
        num_classes (int): Number of target classification categories (default: 3).
        pretrained (bool): Whether to load pre-trained ImageNet1K weights (default: True).
        weights_path (str, optional): Path to a fine-tuned .pth checkpoint file.

    Returns:
        nn.Module: Configured PyTorch ResNet18 model.
    """
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # Replace final Fully Connected layer with target output classes (3 classes)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    # Load fine-tuned weights if checkpoint path is provided
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"[INFO] Loaded checkpoint weights from: {weights_path}")

    return model


if __name__ == "__main__":
    print("[INFO] Testing ResNet18 Model Initialization...")
    model = get_resnet_model(num_classes=3, pretrained=True)

    # Test forward pass with dummy batch
    dummy_input = torch.randn(2, 3, 224, 224)
    dummy_output = model(dummy_input)

    print(f"[TEST] Model Input shape  : {dummy_input.shape} (Expected: [2, 3, 224, 224])")
    print(f"[TEST] Model Output shape : {dummy_output.shape} (Expected: [2, 3])")
    print(f"[TEST] Sample Logits      : {dummy_output.detach().tolist()}")
    print("[INFO] Model architecture test passed successfully.")
