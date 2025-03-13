
# Fast-YOLO for Pneumonia Detection

This repository contains the code and models for the **Fast-YOLO** deep learning network, optimized for the **detection of pneumonia from chest X-ray images**. This model is based on YOLOv11, incorporating specialized modules such as **C3k2**, **DCNv2**, and **DynamicConv** to enhance detection accuracy and computational efficiency.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Validation](#validation)
6. [Detection](#detection)
7. [Results](#results)
8. [License](#license)

## Overview

Early and accurate pneumonia detection is crucial to improving treatment outcomes. Traditional methods relying on physician experience are prone to subjectivity. The **Fast-YOLO** network significantly improves detection accuracy and speed for **pneumonia X-ray diagnosis** using advanced image enhancement techniques and a novel network architecture based on YOLOv11.

Key features of Fast-YOLO:
- Optimized for real-time detection
- Improved accuracy and recall for pneumonia lesion detection
- Reduced computational complexity for faster processing
- Robust to noisy, low-contrast, and complex medical images

## Installation

To set up the **Fast-YOLO** repository locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/SimonZhaoBin/Pneumonia.git
   cd Pneumonia
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the model in editable mode:
   ```bash
   pip install -e .
   ```

## Dataset

The dataset used for training is derived from **MIMIC-CXR**, a comprehensive chest X-ray dataset. It includes images of different lung conditions such as bacterial pneumonia, viral pneumonia, tuberculosis, and healthy lungs. 

The dataset has been annotated with **five categories**:
- Bacterial Pneumonia
- Viral Pneumonia
- Tuberculosis
- Healthy
- Others

For training, a dataset of **4,194 pneumonia images** was used.

## Training

To begin training the model, execute the following command:
```bash
python train.py --data coco8.yaml --imgsz 640 --batch 16 --conf 0.25 --iou 0.6 --device 0
```

**Training Parameters**:
- `data`: Path to the dataset configuration file (e.g., `coco8.yaml`)
- `imgsz`: Image size for input to the model (default: 640)
- `batch`: Batch size for training (default: 16)
- `conf`: Confidence threshold for object detection (default: 0.25)
- `iou`: IOU threshold for non-maximal suppression (default: 0.6)
- `device`: GPU device (default: 0)

### Hyperparameters:
- Epochs: 500
- Batch size: 16
- Learning rate: 0.01
- Momentum: 0.937
- Optimizer: Auto
- Image size: 640

## Validation

After training, validate the model on a test set:
```bash
python val.py --data coco8.yaml --imgsz 640 --batch 16 --conf 0.25 --iou 0.6 --device 0
```

## Detection

To detect pneumonia lesions on a given image, use the following command:
```bash
python detect.py --weights weights/best.pt --imgsz 640 --conf 0.25 --source path/to/test_image
```

Alternatively, for real-time detection using a webcam:
```bash
python Xdetect.py
```

## Results

The **Fast-YOLO** model demonstrates superior performance in various metrics:

- **Precision**: 95.2%
- **Recall**: 94.9%
- **mAP@0.5**: 97.8%
- **FPS**: 120

Compared to other models like **YOLOv5**, **YOLOv7**, **RTMDet-L**, and **D-FINE-L**, Fast-YOLO achieves faster inference times and higher accuracy.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
