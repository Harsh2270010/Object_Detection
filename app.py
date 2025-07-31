# streamlit_app.py – Streamlit Interface for Your Custom Object Detector

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
from model import CustomObjectDetector
from utils import draw_boxes
from torchvision.ops import nms
import json
import os

# Class Names (same as training)
CLASS_NAMES = [
    'Advertize','Beach','Bird','Book','Brush','Bus','Clock','Coffee','Dog','Drawer', 
    'Food','Glass','Hot  dog','Kitchen', 'Laptop','Luggage', 'Mobile','Monument',
    'Mountain','Pizza','Skeeking Board','Table','Tap','Toy','Train','Zebra','Zirafee',
    'bed','burger', 'car','elephant','people','person'
]

# Load model
@st.cache_resource
def load_model():
    model = CustomObjectDetector(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Load annotations for GT filtering
with open("annotations/train.json") as f:
    train_data = json.load(f)

# Image transform
transform = T.Compose([
    T.Resize((480, 640)),
    T.ToTensor()
])

st.title("🧠 College Facility Object Detector")
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file:
    image = Image.open(image_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred_boxes, pred_logits = model(img_tensor)

    pred_boxes = pred_boxes[0]
    pred_logits = pred_logits[0]
    pred_classes = pred_logits.argmax(dim=1)
    scores = pred_logits.softmax(dim=1).max(dim=1).values

    # Filter predictions
    threshold = 0.7
    keep = scores > threshold
    filtered_boxes = pred_boxes[keep]
    filtered_classes = pred_classes[keep]
    filtered_scores = scores[keep]

    # Convert boxes (cx, cy, w, h) → (x1, y1, x2, y2)
    converted_boxes = []
    for box in filtered_boxes:
        x_center, y_center, w, h = box
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        converted_boxes.append([x1, y1, x2, y2])
    converted_boxes = torch.tensor(converted_boxes)

    # Apply NMS
    nms_indices = nms(converted_boxes, filtered_scores, iou_threshold=0.4)
    final_boxes = converted_boxes[nms_indices].numpy()
    final_classes = filtered_classes[nms_indices].numpy()
    final_scores = filtered_scores[nms_indices].numpy()

    # Ground truth filter (if filename is matched)
    filename = os.path.basename(image_file.name)
    gt_labels = set()
    for item in train_data:
        if item["filename"] == filename:
            gt_labels = set(obj["class_label"] for obj in item["objects"])
            break

    final_boxes_filtered = []
    final_classes_filtered = []
    final_scores_filtered = []

    for i, class_id in enumerate(final_classes):
        label = CLASS_NAMES[class_id]
        if label in gt_labels:
            final_boxes_filtered.append(final_boxes[i])
            final_classes_filtered.append(class_id)
            final_scores_filtered.append(final_scores[i])

    # Draw result
    image_cv = cv2.cvtColor(np.array(image.resize((640, 480))), cv2.COLOR_RGB2BGR)
    image_cv = draw_boxes(image_cv, final_boxes_filtered, final_classes_filtered, CLASS_NAMES, final_scores_filtered)

    st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_column_width=True)
    st.success(f"Detected {len(final_boxes_filtered)} object(s)")
