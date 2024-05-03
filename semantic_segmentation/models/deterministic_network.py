from typing import Tuple

import torch
import torch.nn as nn
import torchmetrics

from utils import metrics

from semantic_segmentation.models.base_network import BaseNetwork, NetworkWrapper


class DeterministicNetwork(BaseNetwork):
    """
    Defines the interface for a deterministic network.
    """

    def __init__(self, model: nn.Module, cfg: dict) -> None:
        super(DeterministicNetwork, self).__init__(model, cfg)

        # Common output for all deterministic models
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        This only returns the logits. Post processing (e.g., softmax)
        is done in get_predictions()
        """
        logits = self.model(x)
        return logits

    @torch.no_grad()
    def get_predictions(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        logits = self.forward(data)
        probs: torch.Tensor = self.softmax(logits)
        _, pred_label = torch.max(probs, dim=1)

        unc = -torch.sum(probs * torch.log(probs + 10e-8), dim=1) / torch.log(torch.tensor(self.num_classes))

        return logits, probs, pred_label, unc


class DeterministicNetworkWrapper(NetworkWrapper):
    """
    Network Wrapper
    It is deterministic
    - We use get_predictions to get the probabilities. This is not required in traning.
    - The uncertainty in this case is the ... (entropy...?) and viualized.
    TODO: With this we assume that any classification model offers "some" kind of uncertainty.
            How does this work for regression,  panoptic, foundational...?
    """

    def __init__(self, network: DeterministicNetwork, cfg: dict) -> None:
        super(DeterministicNetworkWrapper, self).__init__(network, cfg)
        self.network = network  # For typing purposes
        self.save_hyperparameters()
        self.vis_interval = self.cfg["train"]["visualization_interval"]  # TODO: For what?

    def training_step(self, batch, batch_idx):
        """
        Pytorch Lightning training step
        One batch pass with the loss and
        log metrics.
        """
        true_label = batch["label"]
        out = self.network.forward(batch["data"])
        loss = self.loss_fn(out, true_label)

        self.log_gradient_norms()
        self.log("train:loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):  # not refactored
        """
        Pytorch Lightning validation step
        One batch pass that tracks loss and
        validation metrics.
        """
        true_label = batch["label"]
        logits, probs, pred_label, _ = self.network.get_predictions(batch["data"])
        loss = self.loss_fn(logits, true_label)

        confusion_matrix = torchmetrics.functional.confusion_matrix(
            pred_label, true_label, task="multiclass", num_classes=self.network.num_classes, normalize=None
        )
        calibration_info = metrics.compute_calibration_info(probs, true_label, num_bins=50)

        # TODO: if this is here, do we also need to visualize on epoch end?
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

    def test_step(self, batch, batch_idx):  # not refactored
        """
        Pytorch Lightning test step
        One batch pass that tracks loss and
        test metrics.
        """
        true_label = batch["label"]
        logits, probs, pred_label, _ = self.network.get_predictions(batch["data"])
        loss = self.loss_fn(logits, true_label)

        print(logits.shape, probs.shape, pred_label.shape, true_label.shape)
        confusion_matrix = torchmetrics.functional.confusion_matrix(
            pred_label, true_label, task="multiclass", num_classes=self.network.num_classes, normalize=None
        )
        calibration_info = metrics.compute_calibration_info(probs, true_label, num_bins=50)
        self.log("test:loss", loss, prog_bar=True)
        outputs = {
            "conf_matrix": confusion_matrix,
            "loss": loss,
            "calibration_info": calibration_info,
        }
        self.test_step_outputs.append(outputs)

        return outputs

    def visualize_step(self, batch):  # not refactored
        """
        Visualize predictions
        This includes the input image, the ground truth label,
        the predicted label, and the uncertainty.
        """
        self.vis_step += 1
        true_label = batch["label"]
        logits, probs, pred_label, unc = self.network.get_predictions(batch["data"])

        # Get a copy of the data in cpu
        img = batch["data"][0].cpu()
        pred_label = pred_label[0].cpu()
        probs = probs[0].cpu()
        true_label = true_label[0].cpu()
        unc = unc[0].cpu()

        self.log_prediction_images(
            img,
            true_label,
            pred_label,
            probs,
            uncertainties=[unc],
            stage="Validation",
            step=self.vis_step,
        )
