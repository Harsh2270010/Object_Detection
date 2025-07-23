import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

# Converts [x_min, y_min, x_max, y_max] → [x_center, y_center, width, height] (normalized)
def convert_bbox(x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return [x_center, y_center, width, height]

class CollegeFacilitiesDataset(Dataset):
    def __init__(self, annotations_file, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        # Extract unique class labels dynamically
        all_classes = set()
        for item in self.annotations:
            for obj in item['objects']:
                all_classes.add(obj['class_label'])
        
        self.classes = sorted(list(all_classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        print(f"Detected classes: {self.classes}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        record = self.annotations[idx]
        img_path = os.path.join(self.image_dir, record['filename'])
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        bboxes = []
        class_ids = []

        for obj in record['objects']:
            bbox = obj['bounding_box']
            bbox_norm = convert_bbox(
                bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max'], img_width, img_height
            )
            bboxes.append(bbox_norm)
            class_ids.append(self.class_to_idx[obj['class_label']])

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        class_ids = torch.tensor(class_ids, dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, bboxes, class_ids
