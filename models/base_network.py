"""
Code for the NetworkWrapper class.
This class configures the model and pipeline stages.
This is: training, validation, testing, visualization and logging.
It obtains the information from the config file.
Common logging methods that apply to all methods are included here.
When a method varies the pipeline execution (e.g. mc samples, ensembles, etc.)
it is moved to a subclass that modifies the corresponding functions
"""
from typing import Dict, List, Optional, Tuple, Union
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule

import utils.utils as utils
from constants import IGNORE_INDEX
from models.loss import CrossEntropyLoss
from utils import metrics

from models import get_loss_fn

class BaseNetwork:
    """
    This class defines the basic interface to be declared for every network to be used
    in inference. This is agnostic to the training pocess and includes pre and post
    processing steps that depend on the inference method.
    This class only receives cfg as an argument. Implementations should add model
    """
    def __init__(self, model: nn.Module, cfg: dict):
        self.cfg = cfg
        self.num_classes = self.cfg["model"]["num_classes"]

        # Configure model
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model that returns the logits.
        """
        raise NotImplementedError

    @torch.no_grad()
    def get_predictions(self, data: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Function used to get predictions from the model
        This is what you would expect a deployed model to do.
        Depending on the "Network" it might return different data.

        Args:
            data (torch.Tensor): Input data

        Returns:
            Tuple[torch.Tensor]: Predictions. Depends on the network but typically
                                it will be a tuple of (logits, probabilities, uncertainties...)

        Implementation:
            Every subclass that implements this method is intended to be used in evalution.
            Therefore, adding self.model.eval() and @torch.no_grad() as a decorator is 
            required in most cases. This function should use self.forward() to get the logits.
        """
        raise NotImplementedError


class NetworkWrapper(LightningModule):
    """
    Base class for the network wrapper. It implements common methods for all the network types.
    """
    def __init__(self, network: BaseNetwork, cfg: dict): #TODO (later): define a cfg dataclass ?
        super(NetworkWrapper, self).__init__()
        self.cfg = cfg
        self.network = network
        self.model = self.network.model # Lightning requires the model to be an attribute
        self.ignore_index = IGNORE_INDEX[cfg["data"]["name"]]
        self.loss_fn = get_loss_fn(cfg)
        self.vis_step = 0

        # The following is supposed to be stored in val/tet steps 
        # fn respectively in subclasses
        self.val_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        From LightningModule.
        Returns the optimizer based on the config file.
        TODO (later): do we add more optimizers?
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg["train"]["lr"],
            weight_decay=self.cfg["train"]["weight_decay"]
        )
        return optimizer
    
    def training_step(self):
        raise NotImplementedError

    def validation_step(self):
        raise NotImplementedError

    def test_step(self):
        raise NotImplementedError

    def visualize_step(self):
        raise NotImplementedError

    def on_validation_epoch_end(self):
        """
        Log the confusion matrix and calibration info at the end of the validation epoch.
        WARN: This method is on_validation_epoch_end in new Lighting versions. Refactoring requires
              storing outputs manually as an atribute in validation_step.
        WARN: This method asumes that the children will have "conf_matrix" and "calibration_info" 
              in their outputs.
        """
        outputs = self.val_step_outputs
        conf_matrices = [tmp["conf_matrix"] for tmp in outputs]
        calibration_info_list = [tmp["calibration_info"] for tmp in outputs]
        self.log_classification_metrics(
            conf_matrices,
            stage="Validation",
            calibration_info_list=calibration_info_list,
        )
        self.log_confusion_matrix(conf_matrices, stage="Validation")
        self.log_calibration_plots(calibration_info_list)

        fig_ = metrics.compute_calibration_plots(calibration_info_list, num_bins=50)
        self.logger.experiment.add_figure(
            "UncertaintyStats/Calibration", fig_, self.current_epoch
        )

        self.vis_step = 0


    def test_epoch_end(self):
        """
        Log the confusion matrix and calibration info at the end of the evaluation epoch.
        We track... TODO: what?
        WARN: This method is on_validation_epoch_end in new Lighting versions. Refactoring requires
              storing outputs manually as an atribute in validation_step.
        WARN: This method asumes that the children will have "conf_matrix" and "calibration_info" 
              in their outputs.
        """
        outputs = self.test_step_outputs
        conf_matrices = [tmp["conf_matrix"] for tmp in outputs]
        calibration_info_list = [tmp["calibration_info"] for tmp in outputs]

        self.log_classification_metrics(
            conf_matrices,
            stage="Test",
            calibration_info_list=calibration_info_list,
        )
        self.log_confusion_matrix(conf_matrices, stage="Test")
        self.log_calibration_plots(calibration_info_list)

    def log_classification_metrics(  # not refactored
        self,
        conf_matrices: List[torch.Tensor],
        stage: str = "Test",
        calibration_info_list: Optional[List[Dict]] = None,
    ):
        """
        Track the evaluation metrics based on the confusion matrices.

        Args:
            conf_matrices (list): List of confusion matrices.
            stage (str): Stage of the pipeline.
            calibration_info_list (list): List of calibration info.
        """
        miou = metrics.mean_iou_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        per_class_iou = metrics.per_class_iou_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        accuracy = metrics.accuracy_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        precision = metrics.precision_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        recall = metrics.recall_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        f1_score = metrics.f1_score_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )

        ece = -1.0
        if calibration_info_list is not None:
            ece = metrics.ece_from_calibration_info(calibration_info_list, num_bins=50)

        self.log(f"{stage}/Precision", precision)
        self.log(f"{stage}/Recall", recall)
        self.log(f"{stage}/F1-Score", f1_score)
        self.log(f"{stage}/Acc", accuracy)
        self.log(f"{stage}/mIoU", miou)
        self.log(f"{stage}/ECE", ece)

        return {
            f"{stage}/Precision": precision,
            f"{stage}/Recall": recall,
            f"{stage}/F1-Score": f1_score,
            f"{stage}/Acc": accuracy,
            f"{stage}/mIoU": miou,
            f"{stage}/Per-Class-IoU": per_class_iou.tolist(),
            f"{stage}/ECE": ece,
        }

    def log_confusion_matrix(self, conf_matrices: List[torch.Tensor], stage: str="Test"):  # not refactored
        """
        Log the confusion matrix as a figure in seaborn because it looks better.
        """
        aggregated_matrix = metrics.aggregate_confusion_matrices(conf_matrices).cpu().numpy()
        df_cm = pd.DataFrame(
            aggregated_matrix,
            index=range(self.network.num_classes),
            columns=range(self.network.num_classes),
        )

        ax = sns.heatmap(df_cm, annot=True, cmap="Spectral")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground Truth")

        self.logger.experiment.add_figure(
            f"ConfusionMatrix/{stage}", ax.get_figure(), self.current_epoch
        )

        plt.close()
        plt.clf()
        plt.cla()

    def log_calibration_plots(self, calibration_info_list: List[Dict]):  # not refactored
        """
        Log the calibration plots.

        Args:
            calibration_info_list (list): List of calibration info. TODO: Improve definition
        """
        fig_ = metrics.compute_calibration_plots(calibration_info_list, num_bins=50)
        self.logger.experiment.add_figure(
            "UncertaintyStats/Calibration", fig_, self.current_epoch
        )

    def log_gradient_norms(self):  # not refactored
        """
        Log the gradient norms.
        Debug gradient flow. Common problem RNNs.
        If gradients vanish: norm goes to 0. Mean you don't learn more.
        If gradients explode: norm goes to large number. Mean you always "diverge"
        """
        total_grad_norm = 0
        for params in self.network.model.parameters():
            if params.grad is not None:
                total_grad_norm += params.grad.data.norm(2).item()

        self.log(f"LossStats/GradientNorm", torch.tensor(total_grad_norm))

    def log_prediction_images( # not refactored TODO: Move plot predictions to some utils?
        self,
        image: torch.Tensor,
        true_label: torch.Tensor,
        argmax_pred: torch.Tensor,
        prob_pred: torch.Tensor,
        stage: str = "Test",
        step: int = 0,
        uncertainty: Optional[torch.Tensor] = None,
    ):
        """
        Log prediction images.
        All the images should be in CHW format, TODO: check this
        detached from the graph and in CPU.  TODO: check this
        Args:
            image (torch.Tensor): Input image
            true_label (torch.Tensor): Ground truth label
            argmax_pred (torch.Tensor): Predicted label
            prob_pred (torch.Tensor): Predicted probabilities
            stage (str): Stage of the pipeline
            step (int): Step of the pipeline
            uncertainty (torch.Tensor): Uncertainty
        """
        # Plot the input image TODO: check the squeezes in this method
        self.logger.experiment.add_image(
            f"{stage}/Input image", image, step, dataformats="CHW"
        )

        # Plot the output image as a label mask TODO: toOneHot transforms to detach.cpu.numpy(). fix.
        imap_pred = utils.toOneHot(argmax_pred, self.cfg["data"]["name"])
        self.logger.experiment.add_image(
            f"{stage}/Output image",
            torch.from_numpy(imap_pred),
            step,
            dataformats="HWC",
        )

        # Plot the ground truth label as a label mask
        imap_true = utils.toOneHot(true_label, self.cfg["data"]["name"])
        self.logger.experiment.add_image(
            f"{stage}/Annotation",
            torch.from_numpy(imap_true),
            step,
            dataformats="HWC",
        )

        # Plot the error image with the cross entropy loss 
        # This needs the images to be in batch format
        # TODO: Change to configured loss?
        prob_pred_batch = prob_pred.unsqueeze(0)
        true_label_batch = true_label.unsqueeze(0)

        cross_entropy_fn = CrossEntropyLoss(reduction="none")
        sample_error_img = cross_entropy_fn(
            prob_pred_batch, true_label_batch
        ).squeeze()
        sizes = imap_pred.shape # TODO: what is this for?
        px = 1 / plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(px * sizes[1], px * sizes[0]))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.imshow(sample_error_img.cpu().numpy(), cmap="gray")
        self.logger.experiment.add_figure(f"{stage}/Error image", fig, step)

        fig.add_axes(ax)
        plt.cla()
        plt.clf()

        # Uncertainty
        if uncertainty is not None:
            unc_np = uncertainty.numpy()
            fig = plt.figure(figsize=(px * sizes[1], px * sizes[0]))
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            ax.imshow(unc_np, cmap="plasma")
            fig.add_axes(ax)
            self.logger.experiment.add_figure(f"{stage}/Uncertainty", fig, step)
            plt.cla()
            plt.clf()
