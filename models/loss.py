import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
        weight: torch.Tensor = None,
    ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(
            reduction=reduction, ignore_index=ignore_index, weight=weight
        )

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy loss.

        Args:
            inputs (torch.Tensor): logits of shape [B x C x H x W]
            target (torch.Tensor): ground-truth target tensor of shape [B x H x W]

        Returns:
              torch.Tensor: weighted mean of the output losses.
        """

        loss = self.criterion(inputs, target)

        return loss


class AleatoricLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
        weight: torch.Tensor = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, inputs, target):
        if len(inputs.shape) == 4:
            inputs = inputs.unsqueeze(0)

        T, B, C, H, W = inputs.shape
        inputs = inputs.permute(0, 1, 3, 4, 2)  # T,B,H,W,C
        super_B = B * H * W  # super batch

        inputs = inputs.reshape(T, -1, C)  # T, super_B, C
        target = target.view(-1)
        inputs_indexed = inputs[
            :, torch.arange(super_B).type_as(target), target
        ]  # (T, super_B)

        sum_term = torch.sum(
            torch.exp(
                inputs_indexed - torch.log(torch.sum(torch.exp(inputs), dim=-1) + 1e-8)
            ),
            dim=0,
        )
        log_term = torch.log(sum_term / T + 1e-8)
        loss = -torch.sum(log_term) / super_B
        return loss


class NLLLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
        weight: torch.Tensor = None,
    ):
        super().__init__()
        self.criterion = nn.NLLLoss(
            reduction=reduction, ignore_index=ignore_index, weight=weight
        )

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute NLL loss.

        Args:
            inputs (torch.Tensor): probability vector of shape [B x C x H x W]
            target (torch.Tensor): ground-truth target tensor of shape [B x H x W]

        Returns:
              torch.Tensor: weighted mean of the output losses.
        """

        loss = self.criterion(torch.log(inputs + 1e-8), target)

        return loss
