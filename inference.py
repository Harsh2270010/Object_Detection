import torch
from PIL import Image
import torchvision.transforms as T
from model import CustomObjectDetector
from utils import draw_boxes
import cv2
import numpy as np
from torchvision.ops import nms
import json
import os

# Classes
CLASS_NAMES = [
    'Advertize','Beach','Bird','Book','Brush','Bus','Clock','Coffee','Dog','Drawer', 
    'Food','Glass','Hot  dog','Kitchen', 'Laptop','Luggage', 'Mobile','Monument',
    'Mountain','Pizza','Skeeking Board','Table','Tap','Toy','Train','Zebra','Zirafee',
    'bed','burger', 'car','elephant','people','person'
]

# Image Transformation
transform = T.Compose([
    T.Resize((480, 640)),
    T.ToTensor()
])

# Load model
model = CustomObjectDetector(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# Loading image
img_path = "images/train/000000000072.jpg"
image = Image.open(img_path).convert("RGB")
img_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    pred_boxes, pred_logits = model(img_tensor)

pred_boxes = pred_boxes[0]         # [N, 4]
pred_logits = pred_logits[0]       # [N, C]
pred_classes = pred_logits.argmax(dim=1)
scores = pred_logits.softmax(dim=1).max(dim=1).values

# Confidence filtering
threshold = 0.7
keep = scores > threshold
filtered_boxes = pred_boxes[keep]
filtered_classes = pred_classes[keep]
filtered_scores = scores[keep]

# Convert (cx, cy, w, h) to (x1, y1, x2, y2)
converted_boxes = []
for box in filtered_boxes:
    x_center, y_center, w, h = box
    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2
    converted_boxes.append([x1, y1, x2, y2])
converted_boxes = torch.tensor(converted_boxes)

# NMS
nms_indices = nms(converted_boxes, filtered_scores, iou_threshold=0.4)
final_boxes = converted_boxes[nms_indices].numpy()
final_classes = filtered_classes[nms_indices].numpy()
final_scores = filtered_scores[nms_indices].numpy()

# Load GT Labels from train.json
with open("annotations/train.json") as f:
    train_data = json.load(f)

filename = os.path.basename(img_path)
gt_labels = set()
for item in train_data:
    if item["filename"] == filename:
        gt_labels = set(obj["class_label"] for obj in item["objects"])
        break

# Keep only predictions matching GT classes
final_boxes_filtered = []
final_classes_filtered = []
final_scores_filtered = []

for i, class_id in enumerate(final_classes):
    label = CLASS_NAMES[class_id]
    if label in gt_labels:
        final_boxes_filtered.append(final_boxes[i])
        final_classes_filtered.append(class_id)
        final_scores_filtered.append(final_scores[i])

# Draw
image_cv = cv2.cvtColor(np.array(image.resize((640, 480))), cv2.COLOR_RGB2BGR)
image_cv = draw_boxes(image_cv, final_boxes_filtered, final_classes_filtered, CLASS_NAMES, final_scores_filtered)
# Printing logs
print("Boxes:", final_boxes)
print("Class IDs:", final_classes)
print("Scores:", final_scores)
# Show and Save
cv2.imshow("Detection", image_cv)
cv2.imwrite("output_with_boxes.jpg", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
