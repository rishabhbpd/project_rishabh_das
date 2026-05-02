# Detecting Humans in Forests

**Rishabh Das | 20231198 | DS3273**
Indian Institute of Science Education and Research, Pune

---

## Overview

This project develops a binary image classifier that detects whether a human is present in a forest image. The model is intended to assist in preventing illegal activities such as poaching and logging by flagging human presence in forested areas from camera or drone footage.

The classifier is built using a custom Convolutional Neural Network (CNN) trained on the ForestPersons dataset. Given an input image, the model outputs one of two classes: **human** or **no_human**.

---

## Motivation

Since the industrial revolution, human activity has caused drastic changes to ecosystems worldwide. Beyond pollution, poaching and illegal logging remain serious threats to biodiversity. Automated detection of humans in forest environments could serve as an early warning system for conservation efforts, enabling faster response by forest rangers and wildlife authorities.

This project was originally inspired by a visit to Satkosia Nature Reserve, Odisha.

---

## Dataset

**ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection**
Deokyun Kim et al., ICLR 2026
[HuggingFace](https://huggingface.co/datasets/etri/ForestPersons)

The dataset was originally designed for missing person detection in search-and-rescue scenarios. Images were captured using RGB sensors at ground and low-altitude perspectives (1.5m–2.0m), simulating UAV footage, using cameras including a GoPro HERO9 Black and Sony A57. All images are 1920×1080 resolution and annotated in COCO format with bounding boxes around each person.

| Split      | Images | Annotations |
|------------|--------|-------------|
| Train      | 67,686 | 145,816     |
| Validation | 18,243 | 37,395      |
| Test       | 10,553 | 20,867      |

For this project, a subset of the dataset was used:
- **Train**: 1,000 images per class (2,000 total)
- **Validation**: 200 images per class (400 total)
- **Test**: 100 images per class (200 total)

---

## Methodology

### The Two-Class Problem

The ForestPersons dataset contains only images with humans — every image has at least one annotated person. Since binary classification requires a negative class (no human), negative samples were generated synthetically using the bounding box annotations:

- **Positive samples (human)**: The full image is used as-is. Since every image contains a person, the full image always represents the positive class.
- **Negative samples (no_human)**: A random crop of size 224×224 is extracted from the image such that it does not overlap with any ground truth bounding box. This guarantees the cropped patch contains only forest background with no human visible.

This approach keeps the dataset balanced (equal positive and negative samples) and ensures negative samples are realistic forest backgrounds from the same distribution as positive samples.

### Data Pipeline

Images are stored locally in `data/` organized by split and class:

```
data/
├── train/
│   ├── human/
│   └── no_human/
├── val/
│   ├── human/
│   └── no_human/
└── test/
    ├── human/
    └── no_human/
```

PyTorch's `ImageFolder` is used to load images directly from this structure, automatically assigning labels based on subfolder names. Each image is resized to 224×224 and normalized using ImageNet mean and standard deviation.

---

## Model Architecture

A custom CNN (`ForestPersonCNN`) with three convolutional blocks followed by a fully connected classifier head.

```
Input: (batch, 3, 224, 224)

Block 1: Conv2d(3 → 32, 3×3) → BatchNorm2d → ReLU → MaxPool2d(2×2)
         Output: (batch, 32, 112, 112)

Block 2: Conv2d(32 → 64, 3×3) → BatchNorm2d → ReLU → MaxPool2d(2×2)
         Output: (batch, 64, 56, 56)

Block 3: Conv2d(64 → 128, 3×3) → BatchNorm2d → ReLU → MaxPool2d(2×2)
         Output: (batch, 128, 28, 28)

Classifier:
    Flatten → Linear(100352 → 256) → ReLU → Dropout(0.5) → Linear(256 → 2)

Output: (batch, 2)  — raw logits for [human, no_human]
```

All convolutional layers use `padding=1` to preserve spatial dimensions before pooling. Batch Normalization after each convolution stabilizes training. Dropout in the classifier head reduces overfitting.

**Total trainable parameters: 25,784,578**

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer      | Adam  |
| Learning rate  | 1e-3  |
| Batch size     | 16    |
| Epochs         | 10    |
| Dropout        | 0.5   |
| Loss function  | CrossEntropyLoss |
| Input size     | 224×224 |

---

## Results

Best validation accuracy: **99.50%**

---

## Setup

```bash
conda install pytorch torchvision -c pytorch
conda install scikit-learn -c conda-forge
pip install datasets huggingface_hub tqdm Pillow
```

---

## How to Run

**Training:**
```bash
python main.py
```

**Inference on a list of images:**
```python
from predict import predict

results = predict([
    "data/test/human/0001.jpg",
    "data/test/no_human/0001.jpg"
])
print(results)  # ["human", "no_human"]
```

**Via interface:**
```python
from interface import TheModel, the_trainer, the_predictor, TheDataset, the_dataloader
```

**Re-downloading data** (requires HuggingFace access):
```bash
hf auth login
python download_data.py
```

---

## Known Issues

> These are non-critical issues encountered during development and do not affect the final submission.

- **PyTorch not finding CUDA** — PyTorch was installed without CUDA support on this machine. Training runs on CPU. To enable GPU training, reinstall PyTorch with the correct CUDA version from [pytorch.org](https://pytorch.org).
- **`sklearn` not found** — Install via `conda install scikit-learn -c conda-forge`, not `conda install sklearn`.
- **HuggingFace authentication** — The ForestPersons dataset is gated and requires agreeing to terms of use. Run `hf auth login` before running `download_data.py`.
- **Git LFS pointers** — The raw dataset files in the cloned HuggingFace repo are LFS pointers and not actual images. Use `download_data.py` to download images properly.

---

## Citation

```bibtex
@inproceedings{kim2026forestpersons,
  title     = {ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection},
  author    = {Deokyun Kim and Jeongjun Lee and Jungwon Choi and Jonggeon Park and
               Giyoung Lee and Yookyung Kim and Myungseok Ki and Juho Lee and Jihun Cha},
  booktitle = {The Fourteenth International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://huggingface.co/datasets/etri/ForestPersons},
}
```
