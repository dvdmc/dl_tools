"""
    Pytorch Lightning Evidential wrapper from Julius Rückin.
    The implementation is based on:
    https://github.com/dmar-bonn/bayesian_erfnet/
"""

import matplotlib

matplotlib.use("Agg")

import torch
import torch.nn as nn
import torchmetrics

from utils import metrics

from semantic_segmentation.models.base_network import BaseNetwork, NetworkWrapper


class EvidentialNetwork(BaseNetwork):
    """
    Defines the interface for a evidential network.
    """

    def __init__(self, model: nn.Module, cfg: dict) -> None:
        super(EvidentialNetwork, self).__init__(model, cfg)
        # TODO: This should come from the outside
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Used to obtain probabilities from the logits after sampling
        self.softmax = nn.Softmax(dim=1)
        # Used to obtain the standard deviation from the log std
        self.softplus = nn.Softplus(beta=1)
        self.num_mc_aleatoric = self.cfg["train"]["num_mc_aleatoric"]

    def forward(self, x):
        """
        Forward pass with aleatoric uncertainty
        as an additional output. This only returns the logits.
        Post processing (e.g., softplus, aleatoric dist. sampling)
        is done in get_predictions()
        """
        est_seg, est_std = self.model(x)
        return est_seg, est_std

    @torch.no_grad()
    def get_predictions(self, data):
        """
        Function used to get predictions from the model
        This is what you would expect a deployed model to do.
        Depending on the "Network" it might return different data.

        Args:
            data (torch.Tensor): Input data

        Returns:
            torch.Tensor: logit output
            torch.Tensor: probabilities after softmax
            torch.Tensor: predicted labels as argmax of probabilities
            torch.Tensor: uncertainty as categorical entropy
        """
        self.model.eval()
        evidence, hidden_representation = self.model.forward(data)

        ones = torch.ones_like(evidence, device=self.device)
        S = torch.sum(evidence + 1, dim=1, keepdim=True)
        prob = (evidence + 1) / S
        epistemic_unc = (torch.sum(ones, dim=1, keepdim=True) / S).squeeze(1)
        aleatoric_unc = torch.zeros_like(epistemic_unc, device=self.device)

        return (
            prob.cpu().numpy(),
            epistemic_unc.cpu().numpy(),
            aleatoric_unc.cpu().numpy(),
            hidden_representation.cpu().numpy(),
        )


class EvidentialNetworkWrapper(NetworkWrapper):
    """
    Network Wrapper to include Evidential uncertainty as an additional output.
    - We use get_predictions to get the probabilities. This is not required in traning.
    - The uncertainty in this case is the ... (entropy...?) and viualized.
    TODO: clarify extracted from Evidential NN.
    """

    def __init__(self, network: EvidentialNetwork, cfg: dict) -> None:
        super(EvidentialNetworkWrapper, self).__init__(network, cfg)
        self.network = network  # For typing purposes
        self.save_hyperparameters()
        self.vis_interval = self.cfg["train"]["visualization_interval"]

    def training_step(self, batch, batch_idx):
        """
        Pytorch Lightning training step
        One batch pass with the loss and
        log metrics.
        """
        true_label = batch["label"]
        est_seg, est_std = self.network.forward(batch["data"])
        # TODO: This is inconsistent with get_predictions step
        # in get_predictions, softplus is used on std before sampling
        mean_probs = self.network.sample_from_aleatoric_model(est_seg, est_std)
        loss = self.loss_fn(mean_probs, true_label)

        self.log_uncertainty_stats(est_std)
        self.log_gradient_norms()
        self.log("train:loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Pytorch Lightning validation step
        One batch pass that tracks loss and
        validation metrics.
        """
        true_label = batch["label"]
        mean_probs, pred_label, _ = self.network.get_predictions(batch["data"])
        loss = self.loss_fn(mean_probs, true_label)

        confusion_matrix = torchmetrics.functional.confusion_matrix(
            pred_label, true_label, task="multiclass", num_classes=self.network.num_classes, normalize=None
        )
        calibration_info = metrics.compute_calibration_info(mean_probs, true_label, num_bins=50)
        if batch_idx % self.vis_interval == 0:
            self.visualize_step(batch)

        self.log("validation:loss", loss, prog_bar=True)
        outputs = {
            "conf_matrix": confusion_matrix,
            "loss": loss,
            "calibration_info": calibration_info,
        }
        self.val_step_outputs.append(outputs)

        return outputs

    def test_step(self, batch, batch_idx):
        """
        Pytorch Lightning test step
        One batch pass that tracks loss and
        test metrics.
        """
        true_label = batch["label"]
        mean_probs, pred_label, _ = self.network.get_predictions(batch["data"])
        loss = self.loss_fn(mean_probs, true_label)

        confusion_matrix = torchmetrics.functional.confusion_matrix(
            pred_label, true_label, task="multiclass", num_classes=self.network.num_classes, normalize=None
        )
        calibration_info = metrics.compute_calibration_info(mean_probs, true_label, num_bins=50)
        self.log("test:loss", loss, prog_bar=True)
        outputs = {
            "conf_matrix": confusion_matrix,
            "loss": loss,
            "calibration_info": calibration_info,
        }

        self.test_step_outputs.append(outputs)

        return outputs

    def visualize_step(self, batch):
        """
        Visualize the predictions of the model.
        This is specific for aleatoric models.
        """
        self.vis_step += 1
        true_label = batch["label"]
        final_prob, pred_label, aleatoric_unc = self.network.get_predictions(batch["data"])

        # Get a copy of the data in cpu
        img = batch["data"][0].cpu()
        pred_label = pred_label[0].cpu()
        probs = final_prob[0].cpu()
        true_label = true_label[0].cpu()
        unc = aleatoric_unc[0].cpu()

        self.log_prediction_images(
            img,
            true_label,
            pred_label,
            probs,
            uncertainty=unc,
            stage="Validation",
            step=self.vis_step,
        )

    def log_uncertainty_stats(self, std: torch.Tensor):
        """
        Tracks some interesting statistics about the uncertainty

        Args:
            std (torch.Tensor): Standard deviation of the model output
        """
        self.log("Variance/TrainMin", torch.min(std))
        self.log("Variance/TrainMax", torch.max(std))
        self.log("Variance/TrainMean", torch.mean(std))
