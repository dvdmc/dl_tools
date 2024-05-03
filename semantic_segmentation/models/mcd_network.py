"""
    Pytorch Lightning Bayesian training wrapper from Jan Weyler.
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


class MCDNetwork(BaseNetwork):
    """
    Defines the interface for a mcd network.
    """

    def __init__(self, model: nn.Module, cfg: dict) -> None:
        super(MCDNetwork, self).__init__(model, cfg)
        # TODO: This should come from the outside
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Used to obtain probabilities from the logits after sampling
        self.softmax = nn.Softmax(dim=1)
        # Used to obtain the standard deviation from the log std
        self.softplus = nn.Softplus(beta=1)
        self.num_mc_epistemic = self.cfg["train"]["num_mc_epistemic"]

    def forward(self, x):
        """
        Forward pass with epistemic uncertainty
        as an additional output. This only returns the logits.
        Post processing (e.g., softplus, epistemic dist. sampling)
        is done in get_predictions()
        """
        est_seg = self.model(x)
        return est_seg

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
        # We set the dropout layers active during inference!
        # This should ideally be done only once in an deployed model
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        # We sample multiple times to get the epistemic uncertainty
        est_segs = []
        for _ in range(self.num_mc_epistemic):
            logits = self.model(data)
            out = self.softmax(logits)
            est_segs.append(out)
        est_segs = torch.stack(est_segs, dim=0)
        mean_probs = torch.mean(est_segs, dim=0)
        pred_label = torch.argmax(mean_probs, dim=1)
        aleatoric_unc, epistemic_unc = self.compute_uncertainties(est_segs)

        return mean_probs, pred_label, aleatoric_unc, epistemic_unc

    def compute_uncertainties(self, est_segs):
        """
        Compute the uncertainties from the multiple samples
        TODO (Lorenzo): Correct
        """
        mean_probs = torch.mean(est_segs, dim=0)
        aleatoric_unc = torch.mean(mean_probs * (1 - mean_probs), dim=0)
        epistemic_unc = torch.mean(torch.var(est_segs, dim=0), dim=1)
        return aleatoric_unc, epistemic_unc
    
class MCDNetworkWrapper(NetworkWrapper):
    """
    Network Wrapper to include aleatoric and epistemic uncertainty
    - We use get_predictions to get the probabilities. This is not required in traning.
    - The uncertainties are computed according to: TODO (Lorenzo)
    """

    def __init__(self, network: MCDNetwork, cfg: dict) -> None:
        super(MCDNetworkWrapper, self).__init__(network, cfg)
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
        est_seg = self.network.forward(batch["data"])
        loss = self.loss_fn(est_seg, true_label)

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
        mean_probs, pred_label, _, _ = self.network.get_predictions(batch["data"])
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
        final_prob, pred_label, aleatoric_unc, epistemic_unc = self.network.get_predictions(batch["data"])

        # Get a copy of the data in cpu
        img = batch["data"][0].cpu()
        probs = final_prob[0].cpu()
        pred_label = pred_label[0].cpu()
        true_label = true_label[0].cpu()
        aleatoric_unc = aleatoric_unc[0].cpu()

        self.log_prediction_images(
            img,
            true_label,
            pred_label,
            probs,
            uncertainties=[aleatoric_unc, epistemic_unc],
            stage="Validation",
            step=self.vis_step,
        )

