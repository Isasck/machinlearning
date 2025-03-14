# import torch
# #
# # import ssl
# # ssl._create_default_https_context = ssl._create_unverified_context
# # # Load a pre-trained YOLOv5 model
# # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is the small version
# #
# # # Set the model to evaluation mode
# # model.eval()
# #
# # from PIL import Image
# #
# # # Load an image
# # image_path = './data/wigger.jpeg'
# # image = Image.open(image_path)
# #
# # # Perform inference
# # results = model(image)
# #
# # # Display results
# # results.show()  # Opens the image with bounding boxes
# # results.save()  # Saves the image with bounding boxes
# #
# # # Access detection results
# # detections = results.xyxy[0]  # Bounding boxes in [x1, y1, x2, y2, confidence, class] format
# #
# # for detection in detections:
# #     x1, y1, x2, y2, conf, cls = detection
# #     print(f"Class: {model.names[int(cls)]}, Confidence: {conf:.2f}, Bounding Box: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")

from transformers import ViTModel, ViTFeatureExtractor
import torch

# Load the feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')



# Prepare an image (replace 'path_to_image' with your image path)
from PIL import Image
image = Image.open('./data/wigger.jpeg')

# Preprocess the image
inputs = feature_extractor(images=image, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Access the last hidden states or pooled output
last_hidden_states = outputs.last_hidden_state