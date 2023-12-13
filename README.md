# YOLOv8X Custom Object Detection Training on GPU


Introduction

This repository provides a comprehensive guide and codebase for training a custom object detection model using YOLOv8X on a GPU. YOLOv8X is an extension of YOLO (You Only Look Once), a real-time object detection system.
We will be training a yolov8x model which is the most accurate model of all yolov8 models, we will be training the model on "alpaca" images with 448 images for 100 epochs on rtx 3060 gpu(12gb ram)


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

Dataset can be found in "data" folder which contains "train" and "val" folder with images and labels for each


### Code for Training :
```
from ultralytics import YOLO
# Load a model
model = YOLO('yolov8x.yaml')  # build a new model from YAML
# Train the model
if __name__ == '__main__':      
    results = model.train(data='config.yaml', resume=True, epochs=100) #path for config file should be given
```

### Code for Predicting model :

We will predict model output on a video and save the video in "predict" folder inside the "run" folder

```
import os

from ultralytics import YOLO
import cv2


VIDEOS_DIR = os.path.join('.', r"C:\Users\vaida\Videos")

video_path = os.path.join(VIDEOS_DIR, 'pexels-bogdan-krupin-12028405 (1080p).mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
os.chdir(r"C:\Users\vaida\OneDrive\Desktop")
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()
```

