"""
    This module contains the implementation of different loss functions used for training the model.
    The implementation is based on: https://github.com/dmar-bonn/bayesian_erfnet/
"""

from typing import Optional
import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        ignore_index: int = -100,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction, ignore_index=ignore_index, weight=weight)

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
        weight: Optional[torch.Tensor] = None,
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
        inputs_indexed = inputs[:, torch.arange(super_B).type_as(target), target]  # (T, super_B)

        sum_term = torch.sum(
            torch.exp(inputs_indexed - torch.log(torch.sum(torch.exp(inputs), dim=-1) + 1e-8)),
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
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.criterion = nn.NLLLoss(reduction=reduction, ignore_index=ignore_index, weight=weight)

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


"""
    Evidential DL Losses from: https://github.com/dmar-bonn/bayesian_erfnet/
    Re-implemented from: 
    Sensoy, M., Kaplan, L., & Kandemir, M. (2018). 
    Evidential deep learning to quantify classification uncertainty. 
    Advances in neural information processing systems, 31.
"""


def get_kl_divergence_prior_loss(alpha: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    alpha_kl = targets + (1 - targets) * alpha
    alpha_kl_sum = torch.sum(alpha_kl, dim=1, keepdim=True)
    ones = torch.ones_like(alpha)
    kl_log_term = (
        torch.lgamma(alpha_kl_sum)
        - torch.lgamma(torch.sum(ones, dim=1, keepdim=True))
        - torch.sum(torch.lgamma(alpha_kl), dim=1, keepdim=True)
    )
    kl_digamma_term = torch.sum(
        (alpha_kl - 1) * (torch.digamma(alpha_kl) - torch.digamma(alpha_kl_sum)), dim=1, keepdim=True
    )
    return (kl_log_term + kl_digamma_term).squeeze(dim=1)


class PACType2MLELoss(nn.Module):
    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super(PACType2MLELoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, kl_div_coeff: float) -> torch.Tensor:
        targets_idx = torch.argmax(targets, dim=1)
        msk = (targets_idx != self.ignore_index).float()

        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        pac_type2_mle_loss = torch.sum(targets * (torch.log(S) - torch.log(alpha)), dim=1)

        kl_div_prior_loss = get_kl_divergence_prior_loss(alpha, targets)
        loss = (pac_type2_mle_loss + kl_div_coeff * kl_div_prior_loss) * msk

        if self.reduction == "mean":
            loss = loss.sum() / msk.sum()

        return loss


class CrossEntropyBayesRiskLoss(nn.Module):
    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super(CrossEntropyBayesRiskLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, kl_div_coeff: float) -> torch.Tensor:
        targets_idx = torch.argmax(targets, dim=1)
        msk = (targets_idx != self.ignore_index).float()

        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        xentropy_bayes_risk_loss = torch.sum(targets * (torch.digamma(S) - torch.digamma(alpha)), dim=1)

        kl_div_prior_loss = get_kl_divergence_prior_loss(alpha, targets)
        loss = (xentropy_bayes_risk_loss + kl_div_coeff * kl_div_prior_loss) * msk

        if self.reduction == "mean":
            loss = loss.sum() / msk.sum()

        return loss


class MSEBayesRiskLoss(nn.Module):
    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super(MSEBayesRiskLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, kl_div_coeff: float) -> torch.Tensor:
        targets_idx = torch.argmax(targets, dim=1)
        msk = (targets_idx != self.ignore_index).float()

        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        pred_prob = alpha / S
        error_term = torch.square(targets - pred_prob)
        variance_term = pred_prob * (1 - pred_prob) / (S + 1)
        mse_bayes_risk_loss = torch.sum(error_term + variance_term, dim=1)

        kl_div_prior_loss = get_kl_divergence_prior_loss(alpha, targets)
        loss = (mse_bayes_risk_loss + kl_div_coeff * kl_div_prior_loss) * msk

        if self.reduction == "mean":
            loss = loss.sum() / msk.sum()

        return loss
