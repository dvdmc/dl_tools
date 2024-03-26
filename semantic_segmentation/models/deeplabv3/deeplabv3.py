import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50, DeepLabV3

class DeepLabV3Model(nn.Module):
    """
    DeepLabV3 model with MCD dropout.
    """

    def __init__(self, num_classes: int) -> None:
        super(DeepLabV3Model, self).__init__()
        self.model = deeplabv3_resnet50(pretrained=False, progress=True, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)['out']