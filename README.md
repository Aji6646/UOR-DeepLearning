# 🚗 Road Object Detection and Segmentation using YOLOv5 & DeepLabV3+

This project implements a hybrid deep learning pipeline combining **YOLOv5** for object detection and **DeepLabV3+** for semantic road segmentation using the **BDD100K dataset**. It is designed to support real-time perception in autonomous driving scenarios.

---

## 📌 Overview

Autonomous vehicles require robust perception systems to detect obstacles (like cars and pedestrians) and identify drivable areas. This project integrates two state-of-the-art models:

- **YOLOv5** – Fast and accurate object detection.
- **DeepLabV3+** – High-quality semantic segmentation of road regions.

---

## 🧠 Key Features

- Detects multiple object classes (person, car, truck, etc.).
- Segments road area with high-resolution masks.
- Randomly selects and processes images from the BDD100K dataset.
- Displays **side-by-side comparison** of:
  - Raw Image
  - YOLO Detection
  - DeepLabV3+ Segmentation
  - Combined Output

---

## 🖼️ Sample Output

| Input Image | YOLOv5 Detection | Road Segmentation | Combined Output |
|-------------|------------------|-------------------|-----------------|
| ![Sample](docs/input.jpg) | ![Detection](docs/yolo.jpg) | ![Seg](docs/seg.jpg) | ![Output](docs/final.jpg) |

---

## 🗂️ Folder Structure
 ├── Code-v7.py # Main Python script ├── results/ # Saved output images ├── model/ # Pre-trained DeepLabV3+ model ├── requirements.txt # Python dependencies ├── streamlit_app.py # (Optional) Streamlit version └── README.md # This file
