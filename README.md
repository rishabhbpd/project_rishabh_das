# Detecting Humans in Forests

**Rishabh Das | 20231198 | DS3273**
Indian Institute of Science Education and Research, Pune

---

## Overview

A binary CNN classifier that detects whether a human is present in a forest image. Built using a custom 3-block CNN with Batch Normalization trained on the [ForestPersons dataset](https://huggingface.co/datasets/etri/ForestPersons) (ICLR 2026).

Positive samples are full images containing humans. Negative samples are background crops from the same images that do not overlap with any ground truth bounding box.

**Best validation accuracy: 99.50%**

---

## Requirements

```bash
conda install pytorch torchvision -c pytorch
conda install scikit-learn -c conda-forge
pip install datasets huggingface_hub tqdm Pillow
```

---

## Data Setup

Images are stored locally in `data/`. To re-download the dataset, run:

```bash
python download_data.py
```

This requires a HuggingFace account with access to the ForestPersons dataset. Log in first:

```bash
hf auth login
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
results = predict(["data/test/human/0001.jpg", "data/test/no_human/0001.jpg"])
print(results)  # ["human", "no_human"]
```

**Via interface:**
```python
from interface import the_predictor, the_trainer, TheModel
```

---

## Known Issues

> These are non-critical issues encountered during development and do not affect the final submission.

- **PyTorch not finding CUDA** — PyTorch was installed without CUDA support. Training runs on CPU. Installing the CUDA version of PyTorch requires specifying `-c pytorch` and the correct CUDA version.
- **`sklearn` not found** — Install via `conda install scikit-learn -c conda-forge`, not `conda install sklearn`.
- **HuggingFace authentication** — The ForestPersons dataset is gated. Run `hf auth login` before downloading.
- **Git LFS pointers** — The raw dataset files in the cloned HuggingFace repo are LFS pointers. Use `download_data.py` instead of relying on git lfs pull.

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
