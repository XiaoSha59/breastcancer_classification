"""
segmentation/demo_seg.py
Visual Inference Demo for Breast Ultrasound Tumor Segmentation.
Loads trained ResNet18-UNet weights, segments the tumor, overlays tumor boundary in red, and saves visualization.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segmentation.model_seg import get_segmentation_model


def segment_image(
    image_path: str,
    weights_path: str = "weights/model_seg_best.pth",
    output_path: str = "segmentation_result.png",
    threshold: float = 0.5,
    device: torch.device = None
) -> None:
    """Perform tumor segmentation on a single ultrasound image and save visual overlay."""
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: '{image_path}'")
        return

    if not os.path.exists(weights_path):
        print(f"[ERROR] Segmentation weights not found: '{weights_path}'. Please run 'python segmentation/train_seg.py' first!")
        return

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = get_segmentation_model(pretrained=False, weights_path=weights_path)
    model = model.to(device)
    model.eval()

    # 2. Load and Preprocess Image
    raw_img = Image.open(image_path).convert("RGB")
    resized_img = TF.resize(raw_img, [224, 224])
    input_tensor = TF.to_tensor(resized_img)
    input_tensor = TF.normalize(input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 3. Model Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    binary_mask = (probs > threshold).astype(np.uint8)
    tumor_pixels = np.sum(binary_mask)
    tumor_percentage = (tumor_pixels / (224 * 224)) * 100.0

    # 4. Create Overlay (Red Highlight on Tumor Region)
    img_np = np.array(resized_img)
    overlay = img_np.copy()
    # Add red tint where tumor is detected
    overlay[binary_mask == 1, 0] = np.clip(overlay[binary_mask == 1, 0] * 0.5 + 128, 0, 255)
    overlay[binary_mask == 1, 1] = np.clip(overlay[binary_mask == 1, 1] * 0.5, 0, 255)
    overlay[binary_mask == 1, 2] = np.clip(overlay[binary_mask == 1, 2] * 0.5, 0, 255)

    # 5. Plot and Save Visual Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original Ultrasound Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(probs, cmap="jet")
    axes[1].set_title(f"Tumor Probability Heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Tumor Segmentation Overlay ({tumor_percentage:.1f}% area)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 6. Print Summary
    print("\n" + "=" * 70)
    print("[SEGMENTATION RESULT] Breast Ultrasound Tumor Delineation")
    print("=" * 70)
    print(f" - Input Image        : {image_path}")
    print(f" - Tumor Detected     : {'YES' if tumor_pixels > 0 else 'NO (Normal/Healthy Tissue)'}")
    print(f" - Tumor Area Ratio   : {tumor_percentage:.2f}% of image area")
    print(f" - Result Visualized  : {os.path.abspath(output_path)}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Single Image Tumor Segmentation Demo")
    parser.add_argument("--image", type=str, default="", help="Path to ultrasound image (.png / .jpg)")
    parser.add_argument("--weights", type=str, default="weights/model_seg_best.pth", help="Path to segmentation weights")
    parser.add_argument("--output", type=str, default="segmentation_result.png", help="Path to save result visualization")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        image_path = input("Enter path to breast ultrasound image (.png/.jpg): ").strip().strip('"').strip("'")

    segment_image(image_path=image_path, weights_path=args.weights, output_path=args.output)


if __name__ == "__main__":
    main()
