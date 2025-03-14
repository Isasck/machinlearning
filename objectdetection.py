import torch
import cv2
import numpy as np

# Load YOLOv5 model (e.g., YOLOv5s)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load an image
img = cv2.imread('./data/zestywigger.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform inference
results = model(img)

# Print results
results.print()  # Print results to console
results.show()   # Display the image with detections