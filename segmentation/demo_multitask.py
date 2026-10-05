"""
segmentation/demo_multitask.py
Visual Inference Demo for Joint Multi-Task Model (Classification + Tumor Segmentation).
Takes a single ultrasound image, predicts the tumor class with confidence score, and overlays the segmented tumor contour.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segmentation.model_multitask import get_multitask_model
from segmentation.dataset_multitask import CLASSES


def predict_multitask(
    image_path: str,
    weights_path: str = "weights/model_multitask_best.pth",
    output_path: str = "multitask_demo_result.png",
    threshold: float = 0.5,
    device: torch.device = None
) -> None:
    """Run joint multi-task inference and export diagnostic visualization."""
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: '{image_path}'")
        return

    if not os.path.exists(weights_path):
        print(f"[ERROR] Multi-Task weights not found: '{weights_path}'. Please run 'python segmentation/train_multitask.py' first!")
        return

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = get_multitask_model(pretrained=False, weights_path=weights_path)
    model = model.to(device)
    model.eval()

    # 2. Preprocess Image
    raw_img = Image.open(image_path).convert("RGB")
    resized_img = TF.resize(raw_img, [224, 224])
    input_tensor = TF.to_tensor(resized_img)
    input_tensor = TF.normalize(input_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # 3. Model Inference (Single Forward Pass)
    with torch.no_grad():
        cls_logits, seg_logits = model(input_tensor)

        # Classification Probabilities
        cls_probs = F.softmax(cls_logits, dim=1)[0]
        confidence, pred_idx = torch.max(cls_probs, dim=0)
        predicted_class = CLASSES[pred_idx.item()]
        confidence_pct = confidence.item() * 100.0

        # Segmentation Mask
        seg_probs = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()

    binary_mask = (seg_probs > threshold).astype(np.uint8)
    tumor_pixels = np.sum(binary_mask)
    tumor_area_pct = (tumor_pixels / (224 * 224)) * 100.0

    # 4. Create Visual Overlay (Red Highlight on Tumor)
    img_np = np.array(resized_img)
    overlay = img_np.copy()
    overlay[binary_mask == 1, 0] = np.clip(overlay[binary_mask == 1, 0] * 0.5 + 128, 0, 255)
    overlay[binary_mask == 1, 1] = np.clip(overlay[binary_mask == 1, 1] * 0.5, 0, 255)
    overlay[binary_mask == 1, 2] = np.clip(overlay[binary_mask == 1, 2] * 0.5, 0, 255)

    # 5. Plot and Save Multi-Task Diagnostic Panel
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Original Image
    axes[0].imshow(img_np)
    axes[0].set_title(f"Input Ultrasound Image\nDiagnosis: {predicted_class.upper()} ({confidence_pct:.1f}%)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Tumor Probability Heatmap
    axes[1].imshow(seg_probs, cmap="jet")
    axes[1].set_title("Tumor Probability Heatmap", fontsize=12)
    axes[1].axis("off")

    # Panel 3: Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"Tumor Delineation Overlay\n(Detected Area: {tumor_area_pct:.1f}%)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 6. Terminal Report
    print("\n" + "=" * 70)
    print("[JOINT MULTI-TASK DIAGNOSTIC REPORT]")
    print("=" * 70)
    print(f" - Image Path              : {image_path}")
    print(f" - Predicted Category      : {predicted_class.upper()}")
    print(f" - Confidence Score        : {confidence_pct:.2f}%")
    print(f" - Tumor Detected          : {'YES' if tumor_pixels > 0 else 'NO (Normal/Healthy)'}")
    print(f" - Segmented Tumor Area    : {tumor_area_pct:.2f}% of total image")
    print("-" * 70)
    print(" [CLASS PROBABILITIES]")
    for idx, cls in enumerate(CLASSES):
        p = cls_probs[idx].item() * 100.0
        bar = "#" * int(p / 4)
        print(f"   * {cls.capitalize():<11}: {p:6.2f}% | {bar}")
    print(f"\n[INFO] Diagnostic visual panel saved to: {os.path.abspath(output_path)}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Task Joint Inference Demo")
    parser.add_argument("--image", type=str, default="", help="Path to ultrasound image")
    parser.add_argument("--weights", type=str, default="weights/model_multitask_best.pth", help="Path to multi-task weights")
    parser.add_argument("--output", type=str, default="multitask_demo_result.png", help="Path to output image")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        image_path = input("Enter path to breast ultrasound image (.png/.jpg): ").strip().strip('"').strip("'")

    predict_multitask(image_path=image_path, weights_path=args.weights, output_path=args.output)


if __name__ == "__main__":
    main()
