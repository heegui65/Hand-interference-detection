# Hand Detection System for Liquid handling robot

Real-time hand detection framework built on YOLO11-SLDH, optimized for edge computing and Raspberry Pi deployment.

## Overview

A computer vision system for detecting hand presence in video streams with real-time inference capability. Supports model optimization efficient deployment on resource-constrained devices.

## Features

- **Real-Time Detection**: Webcam and video file processing with adaptive resolution scaling
- **Model Formats**: PyTorch, ONNX, and TensorFlow model support
- **Optimization**: Model pruning and sparsity-aware training
- **Flexible Deployment**: GPU acceleration or CPU-only inference
- **Configurable Thresholds**: Adjustable confidence and NMS parameters

## Requirements

- Python 3.8+
- Raspberry Pi 4B or NVIDIA GPU
- 5GB storage for models and dataset

## Installation

```bash
cd d:\Hand
python -m venv venv
venv\Scripts\activate  # On Linux: source venv/bin/activate

pip install -r requirements.txt
```

## Project Structure

```
Hand/
├── main.py              # Inference pipeline
├── train.py             # Model training
├── sparse_train.py      # Sparsity training
├── prune.py             # Model pruning
├── export.py            # ONNX export
├── config.py            # Configuration
├── requirements.txt     # Dependencies
│
├── cfg/                 # Models and configs
├── core/                # Inference modules
│   ├── infer.py
│   ├── postprocess.py
│   └── logic.py
├── data/                # Dataset (YOLO format)
└── runs/                # Training outputs
```

## Configuration

Edit `config.py` to adjust:

```python
ORIGINAL_VIDEO_WIDTH = 2592     # Video resolution
ORIGINAL_VIDEO_HEIGHT = 1944
INPUT_SCALE = 0.5               # Scale factor for inference speed
CONF_THRESH = 0.6               # Confidence threshold
IOU_THRESH = 0.5                # NMS threshold
TARGET_SIZE = 640               # Model input size
VIDEO_SOURCE = "test.mp4"       # None for webcam
```

## Usage

```bash
# Real-time detection
python main.py

# Training
python train.py

# Sparsity training
python sparse_train.py

# Model pruning
python prune.py

# Export to ONNX
python export.py
```

## Troubleshooting

**Multiprocessing error on Windows**
```bash
# Set workers=0 in config.py
workers=0
```

**GPU not detected**
```bash
pip install onnxruntime-gpu
```

## References

- [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics)
- [Torch-Pruning](https://github.com/VainF/Torch-Pruning)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)

---

**Updated**: 2026-04-19 | **Python**: 3.8+ | **PyTorch**: 2.8.0+cu128
