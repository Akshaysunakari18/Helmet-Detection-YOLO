# Helmet Detection Using Custom YOLO

## Project Overview

This project implements a custom YOLO-based object detection system for detecting helmet usage by two-wheeler riders.

The model identifies two classes:

- With Helmet
- Without Helmet

The trained model is integrated into a Streamlit web application where users can upload an image and receive helmet detection results with bounding boxes and confidence scores.

---

## Problem Statement

Helmet usage is an important safety requirement for two-wheeler riders. Manual monitoring of helmet usage can be difficult when large numbers of riders need to be observed.

This project uses computer vision and object detection to automatically identify whether riders are wearing helmets.

---

## Objective

The main objectives are:

- Build a custom YOLO object detection model.
- Detect riders with and without helmets.
- Train the model using an annotated dataset.
- Evaluate the model on an unseen test dataset.
- Build a Streamlit application for image-based detection.
- Display bounding boxes, class labels, and confidence scores.

---

## Dataset

The dataset contains 1,376 image-label pairs.

The dataset was reorganized into:

- Training: 963 images
- Validation: 275 images
- Testing: 138 images

The dataset was split approximately according to:

- 70% Training
- 20% Validation
- 10% Testing

Each image has YOLO-format annotations.

### Classes

1. With Helmet
2. Without Helmet

---

## Model

The project uses **YOLO11n** from Ultralytics.

YOLO11n was selected because it is a lightweight YOLO model suitable for object detection and practical deployment.

The model was initialized using pretrained YOLO weights and then trained on the custom helmet dataset.

---

## Training

Training configuration:

- Model: YOLO11n
- Image size: 640 × 640
- Epochs: 50
- Batch size: 8
- Optimizer: AdamW
- Dataset: Custom helmet detection dataset

The best trained model is stored as:

```text
models/best.pt