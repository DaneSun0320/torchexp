from torch import nn
from registry import ModuleRegistry

@ModuleRegistry.register("head")
class Head_Segmentation(nn.Module):
    def __init__(self, num_classes, in_channels=512):
        super(Head_Segmentation, self).__init__()
        self.num_classes = num_classes
        self.head = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x