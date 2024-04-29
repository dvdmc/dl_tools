#####
#
# This file contains an example of a model that can be modified to use MCD dropout.
#
#####
from typing import Any, List, Optional, Type, Union
import torch
import torchvision
print(torchvision.__file__)
import torch.nn as nn
from torch.nn.modules import Sequential
#from torchvision.models.segmentation.deeplabv3 import DeepLabV3, _deeplabv3_resnet
#from torchvision.models.resnet import BasicBlock, ResNet, Bottleneck
from torchvision.models.resnet import ResNet, resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
#import sys
#sys.path.append('/home/loren/Documentos/dl_tools_loren/semantic_segmentation/models/deeplabv3')
from ._utils import IntermediateLayerGetter
from ._utils import _SimpleSegmentationModel
from torch.nn import functional as F

"""
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
        We have to reimplement this method just to add the declaration
            of the dropout layer. Otherwise, the IntermediateLayerGetter
            utility will remove the dropout layer.
        layer = super()._make_layer(block, planes, blocks, stride, dilate)
        self.dropout = nn.Dropout(p=0.3)
        return layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #Reimplementation adding the dropout layers.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        print('applying dropout')

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
"""

class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

def old_deeplabv3_resnet50_mcd(num_classes: int) -> DeepLabV3:
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


def deeplabv3_resnet50_mcd(backbone, num_classes, aux):
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    #aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = DeepLabHead(2048, num_classes)
    return DeepLabV3(backbone, classifier, None)

class DeepLabV3Model_mcd(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(DeepLabV3Model_mcd, self).__init__()
        weights_backbone = ResNet50_Weights.verify(ResNet50_Weights.IMAGENET1K_V1)
        resnet_backbone = resnet50(weights=weights_backbone, replace_stride_with_dilation=[False, True, True])
        self.model = deeplabv3_resnet50_mcd(resnet_backbone, num_classes, aux = False)
        print('Loading DeepLabV3Model_mcd instantiated')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)['out']

class old_DeepLabV3Model_mcd(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(DeepLabV3Model_mcd, self).__init__()
        self.model = deeplabv3_resnet50_mcd(num_classes)
        print('Loading DeepLabV3Model_mcd instantiated')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)['out']
    
#model = DeepLabV3Model_mcd(21)
#x = torch.rand(4, 3, 224, 224)
#output = model(x)