# YOLOv8X Custom Object Detection Training on GPU


Introduction

This repository provides a comprehensive guide and codebase for training a custom object detection model using YOLOv8X on a GPU. YOLOv8X is an extension of YOLO (You Only Look Once), a real-time object detection system.

### Requirements :

* Python (3.6 or higher)
* CUDA-enabled GPU
* CUDA version of pytorch install

### Installation :
(For Windows)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
visit https://pytorch.org/get-started/locally/ for different OS installation
```
pip install ultralytics
```
### Code :
```
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
```