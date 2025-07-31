import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CustomObjectDetector
from dataset import CollegeFacilitiesDataset
from utils import draw_boxes
import os

CLASS_NAMES = [
 'Advertize','Beach','Bird','Book','Brush','Bus','Clock','Coffee','Dog','Drawer', 'Food','Glass','Hot  dog','Kitchen',
 'Laptop','Luggage', 'Mobile','Monument','Mountain','Pizza','Skeeking Board','Table','Tap','Toy','Train',
'Zebra','Zirafee','bed','burger', 'car','elephant','people','person'
]

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


train_dataset = CollegeFacilitiesDataset(
    annotation_file='annotations/train.json',
    image_dir='images/train',
    class_names=CLASS_NAMES,
    transform=transform
)


def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    images = torch.stack(images)
    return images, boxes, labels

dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomObjectDetector(num_classes=len(CLASS_NAMES)).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
bbox_loss_fn = nn.SmoothL1Loss()
class_loss_fn = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),     
    transforms.ToTensor()
])

for epoch in range(1000):
    model.train()
    total_loss = 0.0  

    for imgs, bboxes, labels in dataloader:
        imgs = imgs.to(device)
        bboxes = [b.to(device) for b in bboxes]
        labels = [l.to(device) for l in labels]

        optimizer.zero_grad()
        pred_boxes_batch, pred_class_logits_batch = model(imgs)

        total_bbox_loss = 0.0
        total_class_loss = 0.0

        for i in range(len(imgs)):
            pred_boxes = pred_boxes_batch[i]
            pred_logits = pred_class_logits_batch[i]

            gt_boxes = bboxes[i]
            gt_labels = labels[i]

            N = min(pred_boxes.shape[0], gt_boxes.shape[0])
            loss_bbox = bbox_loss_fn(pred_boxes[:N], gt_boxes[:N])
            loss_class = class_loss_fn(pred_logits[:N], gt_labels[:N])

            total_bbox_loss += loss_bbox
            total_class_loss += loss_class

        loss = total_bbox_loss + total_class_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


torch.save(model.state_dict(), "model.pth")
