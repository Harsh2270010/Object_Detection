import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CollegeFacilitiesDataset(Dataset):
    def __init__(self, annotation_file, image_dir, class_names, transform=None):
        self.image_dir = image_dir
        self.class_names = class_names
        self.transform = transform

        # Load JSON annotations
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(self.image_dir, sample['filename'])
        image = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []

        for obj in sample['objects']:
            bbox = obj['bounding_box']
            x = bbox['x_center']
            y = bbox['y_center']
            w = bbox['width']
            h = bbox['height']
            boxes.append([x, y, w, h])
            labels.append(self.class_names.index(obj['class_label']))

        if self.transform:
            image = self.transform(image)

        # Return variable-length boxes and labels as lists or tensors
        return image, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
