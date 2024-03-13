"""
    Code for the ERFNet model.
    Extracted from: https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownsamplerBlock(nn.Module):
    """
    Downsampler block for ERFNet

    Args:
        ninput (int): Number of input channels
        noutput (int): Number of output channels

    """

    def __init__(self, ninput: int, noutput: int) -> None:
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    """
    non_bottleneck_1d block for ERFNet

    Args:
        chann (int): Number of input and output channels
        dropprob (float): Dropout probability
        dilated (int): Dilation factor

    """

    def __init__(self, chann: int, dropprob: float, dilated: int) -> None:
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(1 * dilated, 0),
            bias=True,
            dilation=(dilated, 1),
        )

        self.conv1x3_2 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 1 * dilated),
            bias=True,
            dilation=(1, dilated),
        )

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.relu(output + input)


class ERFNetEncoder(nn.Module):
    """
    Encoder for ERFNet

    Args:
        num_classes (int): Number of classes

    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for _ in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for _ in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

    def forward(self, input: torch.Tensor, predict: bool = False) -> torch.Tensor:
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        return output


class UpsamplerBlock(nn.Module):
    """
    Upsampler block for ERFNet

    Args:
        ninput (int): Number of input channels
        noutput (int): Number of output channels

    """

    def __init__(self, ninput: int, noutput: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class ERFNetDecoder(nn.Module):
    """
    Decoder for ERFNet

    Args:
        num_classes (int): Number of classes

    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input

        for layer in self.layers:
            output = layer(output)

        output_seg = self.output_conv(output)

        return output_seg


class ERFNetModel(nn.Module):
    """
    ERFNet

    Args:
        num_classes (int): Number of classes

    Returns:
        nn.Module: ERFNet
    """

    def __init__(self, num_classes):
        super().__init__()

        self.encoder = ERFNetEncoder(num_classes)
        self.decoder = ERFNetDecoder(num_classes)

    def forward(self, input):
        output_enc = self.encoder(input)
        output_seg = self.decoder(output_enc)

        return output_seg
