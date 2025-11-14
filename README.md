# IPCV-project-sem1

# BCM — Lightweight Base-Class Mining for Few-Shot Semantic Segmentation

A compact, reproducible implementation of **Base-Class Mining (BCM)** adapted for low-compute environments (MobileNetV2 backbone + logistic regression refiners). This repository contains the runnable code used in the project, example outputs and CSV summaries, and the project report.

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [How it works (high-level)](#how-it-works-high-level)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

BCM (Base-Class Mining) finds frequently-predicted base classes for novel-class pixels (co-occurrence), maps novel→base (BNM), then trains light-weight classifiers that refine base-class pixels into novel classes. This implementation emphasizes simplicity and reproducibility.

This repo includes:
- A runnable PyTorch script implementing the pipeline (backbone features + optional DeepLabV3 base predictions).
- Scripts that compute co-occurrence, build BNM mappings, train logistic-regression refiners, run inference, and save metrics.
- Example outputs: per-image & per-class IoU CSVs, co-occurrence/BNM CSVs, confusion matrix, predicted single-channel masks.

---

## Features

- Works with a dataset ZIP where each class is a folder of images (and corresponding masks).
- Single-channel H×W predicted masks saved as `uint8` images.
- Saves CSVs under `OUT_DIR/csv/` and predicted masks under `OUT_DIR/visuals/`.
- CPU-friendly default; will use GPU if available and configured.

---

## Requirements

Recommended Python environment (example):

- Python 3.8+
- torch
- torchvision
- numpy
- pillow
- scikit-learn
- matplotlib
- tqdm

Install with pip (example):
```bash
pip install torch torchvision numpy pillow scikit-learn matplotlib tqdm
```

---

## Quick start

1. Place your dataset ZIP in the project folder or upload to Colab/Drive. Example dataset: [click here].
2. Edit the top of the script to set `ZIP_PATH`, `EXTRACT_DIR`, `OUT_DIR`, and other hyperparameters (e.g. `K_SHOT`, `NUM_NOVEL`, `IMG_SIZE`).
3. Run:
```bash
python bcm_fss1000.py
```
4. Results and visuals will be saved under the `OUT_DIR` directory.

> Example dataset link (used in this project): [click here].

---

## Configuration

At the top of `bcm_fss1000.py` (or your main script) you can change these values:

- `ZIP_PATH` — path to dataset zip
- `EXTRACT_DIR` — folder to extract the dataset
- `OUT_DIR` — output folder (CSVs & predicted masks)
- `NUM_NOVEL`, `K_SHOT`, `IMG_SIZE` — experiment settings
- `TOP_S`, `TAU`, `MAX_PIXELS_PER_CLASS_PER_IMAGE` — BCM-specific settings
- `PREFER_CUDA` — set to `True` to prefer CUDA when available

---

## Outputs

Saved by default in `OUT_DIR`:

- `csv/cooccurrence.csv` — co-occurrence counts between base and novel ids
- `csv/bnm_mapping.csv` — novel→base mapping used for training
- `csv/per_image_results.csv` — per-image base/novel mIoU
- `csv/per_class_iou.csv` — per-class IoU
- `csv/confusion_matrix.csv` — confusion matrix table
- `visuals/*_pred.png` — predicted single-channel masks for query images

Example outputs archive: [click here].

---

## How it works (high-level)

1. Randomly choose `NUM_NOVEL` classes and sample `K_SHOT` support images per novel class.
2. Run a base segmentation model (DeepLabV3 optional) to get base-class predictions for each support image.
3. Count co-occurrence: for support pixels labeled as a novel class, record which base class the base model predicted at that pixel.
4. For each novel class select the top-S base classes (BNM). Invert mapping to get `base -> [novels]` (set B).
5. For each base β in B, sample pixel features from support images and train a light logistic regression model to classify between β and the mapped novel IDs.
6. At inference, for pixels predicted as β by the base model, run β's trained classifier on pixel features to refine into novel classes.
7. Upsample to full resolution, save predictions, and compute IoU metrics & confusion matrix.

---

## Citation

If you use this implementation or the BCM idea, please cite the original BCM paper and this project.

**Original BCM paper** — see the project report and paper included in this repository.

Example BibTeX (paste into your `.bib` file):

```bibtex
@inproceedings{sakai2024bcm,
  title = {A Surprisingly Simple Approach to Generalized Few-Shot Semantic Segmentation},
  author = {Tomoya Sakai and Haoxiang Qiu and Takayuki Katsuki and Daiki Kimura and Takayuki Osogami and Tadanobu Inoue},
  booktitle = {NeurIPS (38th Conference on Neural Information Processing Systems)},
  year = {2024},
  note = {Dataset: [click here]; Outputs: [click here]}
}

@techreport{tanwar2025bcmreport,
  title = {BCM with Lightweight Feature-Based Few-Shot Segmentation},
  author = {Siddharth Tanwar},
  institution = {Project report},
  year = {2025},
  note = {Implementation and report included in this repository.}
}
```

> Replace `[click here]` placeholders in the BibTeX with the relevant URLs if preferred.

---

## License

This project is provided under the MIT License. Modify as needed.

---

## Acknowledgements

- The BCM authors for the original idea.
- Inspiration and README structure based on the IBM/BCM repository.

---








