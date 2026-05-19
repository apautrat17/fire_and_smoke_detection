import torch.nn as nn
from ultralytics import YOLO
from src.main import config


class FireSmokeModel(nn.Module):
    def __init__(self, backbone, in_channels, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv2d(
            in_channels=in_channels, out_channels=5 + num_classes, kernel_size=1
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


class YOLOFeatureBackbone(nn.Module):
    def __init__(self, yolo_model, hook_key="x", feature_maps={}):
        super().__init__()
        self.yolo_model = yolo_model
        self.hook_key = hook_key
        self.feature_maps = feature_maps

    def forward(self, x):
        _ = self.yolo_model(x)
        return self.feature_maps[self.hook_key]


def create_fire_smoke_model():

    yolo_model = YOLO("yolov8m.pt")

    yolo_nn = yolo_model.model

    feature_maps = {}

    def save_features(module, inputs, output):
        feature_maps["x"] = output

    hook_layer = yolo_nn.model[9]
    hook_layer.register_forward_hook(save_features)

    backbone = YOLOFeatureBackbone(yolo_nn, feature_maps=feature_maps)
    model = FireSmokeModel(
        backbone, in_channels=config.nb_in_channels, num_classes=config.num_classes
    ).to(config.device)
    return model
