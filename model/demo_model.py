from torch import nn
from module.backbone.resnet import resnet18
from module.head.seg_head import Head_Segmentation
from registry import ModelRegistry, ModuleRegistry


# 模型定义 示例1
@ModelRegistry.register()
class Demo_Model(nn.Module):
    def __init__(self, num_classes):
        super(Demo_Model, self).__init__()
        self.backbone = resnet18(num_classes=num_classes, include_top=False)
        self.segmentation_head = Head_Segmentation(num_classes=num_classes, in_channels=512)

    def forward(self, x):
        x = self.backbone(x)
        x = self.segmentation_head(x)
        return x

if __name__ == "__main__":
    model = Demo_Model(num_classes=2)
    ModelRegistry.show_modules_tree()