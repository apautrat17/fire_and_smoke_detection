import torch.nn as nn
from ultralytics import YOLO
from src.main import config


class Head(nn.Module):
    def __init__(self, c, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.SiLU(), nn.Conv2d(c, 5 + num_classes, 1)
        )

    def forward(self, x):
        return self.net(x)


class FireSmokeModel(nn.Module):
    def __init__(self, backbone, in_channels, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.head = Head(in_channels, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


def create_fire_smoke_model():

    yolo_model = YOLO("yolov8m.pt")

    yolo_nn = yolo_model.model

    backbone = nn.Sequential(*list(yolo_nn.model[:9]))

    model = FireSmokeModel(
        backbone, in_channels=config.nb_in_channels, num_classes=config.num_classes
    ).to(config.device)
    return model
