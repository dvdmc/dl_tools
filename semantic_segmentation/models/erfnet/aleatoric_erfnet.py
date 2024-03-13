"""
    Code for the Aleatoric implementation of ERFNet.
    An additional channel in the output for estimating
    the aleatoric uncertainty as a standard deviation.
    From: https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf
    Bayesian-ERFNet from Jan Weyler.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from semantic_segmentation.models.erfnet.erfnet import UpsamplerBlock, non_bottleneck_1d, ERFNetEncoder


class ERFNetAleatoricDecoder(nn.Module):
    """
    Decoder for aleatoric ERFNet which adds an additional channel for the standard deviation

    Args:
        num_classes (int): Number of classes

    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16, num_classes + 1, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Outputs a tuple of (segmentation, standard deviation)
        """
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)
        output_seg, output_std = output.split(self.num_classes, 1)

        return output_seg, output_std


class AleatoricERFNetModel(nn.Module):
    """
    Aleatoric ERFNet
    Adds one channel to the output to estimate
    aleatoric uncertainty

    Args:
        num_classes (int): Number of classes

    Returns:
        nn.Module: ERFNet
    """

    def __init__(self, num_classes):
        super().__init__()

        self.encoder = ERFNetEncoder(num_classes)
        self.decoder = ERFNetAleatoricDecoder(num_classes)

    def forward(self, input):
        output_enc = self.encoder(input)
        output_seg, output_std = self.decoder(output_enc)

        return output_seg, output_std
