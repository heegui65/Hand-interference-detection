# Hand Detection System for Liquid handling robot

Real-time hand detection framework built on YOLO11-SLDH, optimized for edge computing and Raspberry Pi deployment.

## Overview

A computer vision system for detecting hand presence in video streams with real-time inference capability. Supports model optimization efficient deployment on resource-constrained devices.


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


## References

- [YOLO - Ultralytics](https://github.com/ultralytics/ultralytics)
- [ONNX Runtime](https://github.com/microsoft/onnxruntime)

---

**Updated**: 2026-04-19 | **Python**: 3.8+ | **PyTorch**: 2.8.0+cu128
