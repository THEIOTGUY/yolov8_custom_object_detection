from ultralytics import YOLO
import os 
os.chdir(r"C:\Users\vaida\OneDrive\Desktop")
# Load a model
model = YOLO('yolov8x.yaml')  # build a new model from YAML
# Train the model
if __name__ == '__main__':      
    results = model.train(data='config.yaml', resume=True, epochs=100)
