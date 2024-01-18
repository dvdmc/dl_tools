"""
    Code for the Aleatoric implementation of UNet.
    An additional channel in the output for estimating
    the aleatoric uncertainty as a standard deviation.
    From: https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_segmentation.models.unet.unet import double_conv

class AleatoricUNetModel(nn.Module):
    """
    Aleatoric UNet model

    Args:
        num_classes (int): Number of classes

    Returns:
        nn.Module: Aleatoric UNet model
    """
    def __init__(self, num_classes: int) -> nn.Module:
        super().__init__()
        self.num_classes = num_classes

        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, num_classes * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Outputs a tuple of (segmentation, standard deviation)
            TODO (later): can be combined ith UNet to simplify
        """
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        output = self.conv_last(x)

        output_seg, output_std = output.split(self.num_classes, 1)

        return output_seg, output_std
