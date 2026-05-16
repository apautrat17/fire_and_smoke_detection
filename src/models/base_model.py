import torch.nn as nn

class FireSmokeModel(nn.Module):
    def __init__(self, backbone, in_channels, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv2d(
            in_channels=in_channels,
            out_channels=5 + num_classes,
            kernel_size=1
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)