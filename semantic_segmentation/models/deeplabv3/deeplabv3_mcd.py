#####
#
# This file contains an example of a model that can be modified to use MCD dropout.
#
#####
from typing import Any, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch.nn.modules import Sequential
from torchvision.models.segmentation.deeplabv3 import DeepLabV3, _deeplabv3_resnet
from torchvision.models.resnet import BasicBlock, ResNet, Bottleneck

class ResNet50_mcd(ResNet):
    def __init__(self):
        super(ResNet50_mcd, self).__init__(Bottleneck, [3, 4, 6, 3], replace_stride_with_dilation=[False, True, True])
        # self.dropout = nn.Dropout(p=0.3)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> Sequential:
        """ We have to reimplement this method just to add the declaration
            of the dropout layer. Otherwise, the IntermediateLayerGetter
            utility will remove the dropout layer."""
        layer = super()._make_layer(block, planes, blocks, stride, dilate)
        self.dropout = nn.Dropout(p=0.3)
        return layer
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """ Reimplementation adding the dropout layers."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.dropout(x)

        x = self.layer4(x)
        x = self.dropout(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def deeplabv3_resnet50_mcd(num_classes: int) -> DeepLabV3:
    """Instantiates a deeplabv3_resnet50 model with MCD dropout.
        There is no weight loading for this model.

    Args:
        num_classes: Number of classes for the model.

    Returns:
        The modified DeepLabv3 model.
    """
    # The init arguments are taken from the base class instantiations in 
    # torchvision.models.segmentation.deeplabv3_resnet50() and 
    # torchvision.models.resnet50()
    backbone = ResNet50_mcd()

    # THIS FUNCTION USES "IntermediateLayerGetter" UTILITY WHICH IS EVIL. 
    # Therefore we have to declare the dropout in the _make_layer method
    # so Interm...Getter does not remove this the dropout.
    model = _deeplabv3_resnet(backbone, num_classes, None)
    
    return model

class DeepLabV3Model_mcd(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(DeepLabV3Model_mcd, self).__init__()
        self.model = deeplabv3_resnet50_mcd(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)['out']