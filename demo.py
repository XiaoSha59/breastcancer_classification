"""
demo.py
Single-image inference script for Breast Ultrasound (BUSI) Tumor Classification.
Loads fine-tuned weights from 'weights/model_best.pth' and outputs predicted diagnosis with confidence score.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

from src.dataset import get_transforms, CLASSES
from src.model import get_resnet_model


def predict_image(
    image_path: str,
    weights_path: str = "weights/model_best.pth",
    device: torch.device = None
) -> None:
    """
    Perform inference on a single ultrasound image.
    
    Args:
        image_path (str): Path to target ultrasound image.
        weights_path (str): Path to trained model checkpoint weights.
        device (torch.device, optional): Execution device (CPU or CUDA).
    """
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found at: '{image_path}'")
        return

    if not os.path.exists(weights_path):
        print(f"[ERROR] Checkpoint weights not found at: '{weights_path}'. Please run 'python train.py' first!")
        return

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Model and Load Checkpoint Weights
    model = get_resnet_model(num_classes=len(CLASSES), pretrained=False)
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        class_names = checkpoint.get("class_names", CLASSES)
    else:
        model.load_state_dict(checkpoint)
        class_names = CLASSES

    model = model.to(device)
    model.eval()

    # 2. Load Image and Apply Evaluation Transform
    _, eval_transform = get_transforms()
    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Failed to open image: {e}")
        return

    # Add batch dimension: [1, 3, 224, 224]
    input_tensor = eval_transform(raw_image).unsqueeze(0).to(device)

    # 3. Model Inference & Softmax Probabilities
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, dim=0)

    predicted_label = class_names[predicted_idx.item()]
    confidence_pct = confidence.item() * 100.0

    # 4. Format and Print Diagnostics Summary
    print("\n" + "=" * 70)
    print("[DIAGNOSTIC RESULT] Breast Ultrasound Image Classification (BUSI)")
    print("=" * 70)
    print(f" - Image Path        : {image_path}")
    print(f" - Predicted Class   : {predicted_label.upper()}")
    print(f" - Confidence Score  : {confidence_pct:.2f}%")
    print("-" * 70)
    print(" [PROBABILITY DISTRIBUTION]")
    for idx, cls in enumerate(class_names):
        prob = probabilities[idx].item() * 100.0
        bar = "#" * int(prob / 4)
        print(f"   * {cls.capitalize():<11}: {prob:6.2f}%  | {bar}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Single Image Inference Demo for BUSI Classification")
    parser.add_argument("--image", type=str, default="", help="Path to the ultrasound image (.png / .jpg)")
    parser.add_argument("--weights", type=str, default="weights/model_best.pth", help="Path to checkpoint weights (default: weights/model_best.pth)")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        # Prompt for image path if not passed as CLI argument
        image_path = input("Enter path to breast ultrasound image (.png/.jpg): ").strip().strip('"').strip("'")

    predict_image(image_path=image_path, weights_path=args.weights)


if __name__ == "__main__":
    main()
