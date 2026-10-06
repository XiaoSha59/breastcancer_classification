# Multi-Task Breast Ultrasound (BUS) Analysis: Joint Tumor Segmentation and Pathological Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)

This repository provides an automated Deep Learning framework for Breast Ultrasound (BUS) imaging analysis, systematically organized into three core learning paradigms:
1. **Task 1: Single-Task Classification**: Pathological classification using a ResNet18 backbone.
2. **Task 2: Single-Task Tumor Segmentation**: Pixel-level boundary delineation using a ResNet18-UNet architecture.
3. **Task 3: Joint Multi-Task Learning**: Simultaneous tumor segmentation and 3-class pathology diagnosis via a shared ResNet18 encoder with dual specialized heads.

---

## Benchmark Results

All experiments are evaluated on the standardized **Curated BUSI** test split (70% Train, 15% Validation, 15% Test, N=71):

| Method / Model | Learning Paradigm | Classification Accuracy | Tumor Segmentation (Dice) | Tumor Segmentation (IoU) | Epochs / Latency | Description |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **ResNet18 (Ours)** | Task 1: Single-Task Classification | **87.32%** | — | — | 15 ep / ~12 ms | High classification accuracy; cannot localize tumor regions |
| **ResNet18-UNet (Ours)** | Task 2: Single-Task Segmentation | — | **72.10%** | **43.68%** | 15 ep / ~24 ms | Accurate tumor boundary delineation; cannot predict pathology |
| **MultiTask ResNet18-UNet (Ours)** | Task 3: Joint Multi-Task (Cls + Seg) | **83.10%** | **73.40%** | **46.08%** | 15 ep / **~26 ms** | Unified framework; simultaneously localizes tumor and predicts pathology; segmentation IoU improved by +2.4% |
| *Zhang et al. (2023)* | Segmentation Baseline | — | *75.40%* | *~48.00%* | 100 ep / ~35 ms | U-Net baseline on BUSI dataset |
| *Aumente-Maestro et al. (2025)* | Multi-Task SOTA Baseline | *89.80%* | *74.80% – 78.20%* | *~48.50%* | 100 ep / ~80 ms | Multi-Task DenseNet121 / ResNet50 baseline |

*Note: Segmentation Dice and IoU metrics are evaluated on pathological lesion-bearing cases (Benign and Malignant) according to medical imaging reporting standards.*

---

## Dataset: Curated Breast Ultrasound Images (BUSI)

The raw BUSI dataset (Al-Dhabyani et al., 2020) comprises 780 ultrasound scans. Following the curation protocol established by Aumente-Maestro et al. (2025), our pipeline eliminates 330 duplicated/inconsistent triplet scans to establish a clean benchmark of 450 unique scans:

| Pathological Class | Raw BUSI Scans | Removed Duplicates | Curated Clean Scans | Percentage |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | 437 | 215 | **222** | 49.33% |
| **Malignant** | 210 | 46 | **164** | 36.44% |
| **Normal** | 133 | 69 | **64** | 14.23% |
| **Total** | **780** | **330** | **450** | **100.00%** |

---

## Getting Started

### 1. Environment Setup

Clone this repository and create a virtual environment:

```bash
git clone https://github.com/XiaoSha59/breastcancer_classification.git
cd breastcancer_classification

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation and Deduplication

Place the raw `Dataset_BUSI_with_GT` directory inside `data/`, then run the deduplication pipeline:

```bash
python src/preprocess.py
```

This script identifies duplicate hash structures, removes redundant scans, merges multi-lesion ground-truth masks, and organizes the curated dataset into `data/benign`, `data/malignant`, and `data/normal`.

---

## Training the Three Paradigms

### Task 1: Single-Task Classification
Trains a ResNet18 model for 3-class classification (Benign, Malignant, Normal):
```bash
python task1_classification/train.py
```

### Task 2: Single-Task Tumor Segmentation
Trains a ResNet18-UNet model with combined Binary Cross-Entropy and Dice loss:
```bash
python task2_segmentation/train.py
```

### Task 3: Joint Multi-Task Framework (Segmentation + Classification)
Trains the unified MultiTask ResNet18-UNet architecture using joint multi-task loss ($\mathcal{L}_{total} = \lambda_{seg}\mathcal{L}_{seg} + \lambda_{cls}\mathcal{L}_{cls}$):
```bash
python task3_multitask/train.py
```

---

## Repository Structure

```text
breastcancer_classification/
├── data/                         # Curated BUSI dataset (450 images)
│   ├── benign/
│   ├── malignant/
│   ├── normal/
│   └── mapping_curated_BUSI.csv  # Metadata and duplicate tracking mapping
├── weights/                      # Trained model checkpoints (.pth)
├── src/                          # Shared core modules
│   ├── losses.py                 # Centralized Dice, BCEDice, and MultiTask losses
│   └── preprocess.py             # Data curation and deduplication pipeline
├── task1_classification/         # Paradigm 1: Single-Task Classification
│   ├── dataset.py                # Stratified 70/15/15 classification DataLoader
│   ├── model.py                  # ResNet18 classification architecture
│   └── train.py                  # Task 1 training pipeline
├── task2_segmentation/           # Paradigm 2: Single-Task Tumor Segmentation
│   ├── dataset.py                # Segmentation DataLoader
│   ├── model.py                  # ResNet18-UNet segmentation architecture
│   └── train.py                  # Task 2 training pipeline
├── task3_multitask/              # Paradigm 3: Joint Multi-Task Learning
│   ├── dataset.py                # Dual-target (Image, Mask, Label) DataLoader
│   ├── model.py                  # MultiTask ResNet18-UNet dual-head architecture
│   └── train.py                  # Task 3 training pipeline
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

---

## References

1. **Aumente-Maestro, C., Díez, J., & Remeseiro, B. (2025)**. *A multi-task framework for breast cancer segmentation and classification in ultrasound imaging*. Computer Methods and Programs in Biomedicine, 260, 108540.
2. **Zhang, S., Zhou, J., et al. (2023)**. *Fully automatic tumor segmentation of breast ultrasound images with deep learning*. Journal of Applied Clinical Medical Physics, 24(2), e13863.
3. **Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020)**. *Dataset of breast ultrasound images*. Data in Brief, 28, 104863.
