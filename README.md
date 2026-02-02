# StudentSaftyAi

![StudentSaftyAi](https://img.shields.io/badge/StudentSaftyAi-v1.0-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-yellow) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Detector-orange)

AI-based real-time student uniform monitoring system using Python, OpenCV and YOLOv8. Detects students in a live camera feed, validates uniform color/compliance using HSV color analysis, and flags policy violations in real time.

---

## Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Tech Stack](#tech-stack)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training or Updating the Model](#training-or-updating-the-model)
- [HSV Color Calibration Tips](#hsv-color-calibration-tips)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License & Contact](#license--contact)

---

## Features
- Real-time object detection of students using YOLOv8
- Uniform color compliance check via HSV color-space analysis
- Configurable thresholds and camera sources
- Lightweight real-time processing with OpenCV
- Alerts and visual overlays for non-compliant detections

---

## Demo
(Replace with a GIF or screenshot of the running app)
- Live camera feed with detection boxes
- Per-person uniform status (Compliant / Non-compliant)
- Optional logging of violations

---

## Tech Stack
- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- (Optional) PyTorch (if using GPU acceleration)

---

## Requirements
- A webcam or IP camera stream
- Python 3.8 or later
- (Optional) CUDA-enabled GPU + compatible PyTorch for faster inference

Install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
