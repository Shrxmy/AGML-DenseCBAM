# Legacy and Prototype Implementations

> These files are retained for traceability only. They include superseded single-split experiments and known implementation issues. Do not use them for final thesis results; use the root-level active scripts documented in `../README.md`.

`main-cbam-prototype.py` and `main-mtl-prototype.py` were moved here from `scripts/` to prevent accidental use.

## Original prototype description

# An Attention-Based Transfer Learning Model for Diagnosing Subluxation in Temporomandibular Joint Panoramic Radiographs

This repository contains the code for an **attention-based transfer learning model** to diagnose **subluxation in temporomandibular joint (TMJ) panoramic radiographs**.

## Overview

The model leverages pre-trained convolutional neural networks with an **attention mechanism** to focus on relevant regions in panoramic radiographs, improving diagnostic accuracy. Transfer learning allows the model to generalize well with limited medical imaging data.

## Features

- **Transfer Learning:** Pre-trained CNNs (e.g., ResNet, EfficientNet) for feature extraction.  
- **Attention Mechanism:** Focuses on critical regions for subluxation detection.  
- **Binary Classification:** Detects normal vs subluxated TMJ images.  
- **Training & Inference Pipeline:** Includes scripts for training, evaluation, and prediction.
 
