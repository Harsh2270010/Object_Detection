import torch
import torch.nn as nn
import torchvision.models as models

class CustomObjectDetector(nn.Module):
    def __init__(self, num_classes, num_predictions=10):
        super(CustomObjectDetector, self).__init__()
        self.num_classes = num_classes
        self.num_predictions = num_predictions  # max number of objects per image

        # Load pretrained ResNet18 backbone without the final classification layer
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Output: [B, 512, H/32, W/32]

        # Detection head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),           # [B, 512, 1, 1]
            nn.Flatten(),                           # [B, 512]
            nn.Linear(512, self.num_predictions * (4 + num_classes))  # [B, N*(4+classes)]
        )

    def forward(self, x):
        x = self.backbone(x)             # [B, 512, H/32, W/32]
        x = self.head(x)                 # [B, N*(4 + num_classes)]
        x = x.view(-1, self.num_predictions, 4 + self.num_classes)  # [B, N, 4 + C]

        pred_boxes = x[:, :, :4]         # [B, N, 4]
        pred_class_logits = x[:, :, 4:]  # [B, N, C]
        return pred_boxes, pred_class_logits
