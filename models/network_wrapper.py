"""
Code for the NetworkWrapper class.
This class configures the model and pipeline stages.
This is: training, validation, testing, visualization and logging.
It obtains the information from the config file.
Common logging methods that apply to all methods are included here.
When a method varies the pipeline execution (e.g. mc samples, ensembles, etc.)
it is moved to a subclass that modifies the corresponding functions
"""
from typing import List
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning.core.lightning import LightningModule

import utils.utils as utils
from constants import Losses, IGNORE_INDEX
from models.loss import CrossEntropyLoss, NLLLoss, AleatoricLoss
from utils import metrics

class NetworkWrapper(LightningModule):
    """
    Base class for the network wrapper. It implements common methods for all the methods.
    """
    def __init__(self, cfg: dict): #TODO (later): define a cfg dataclass ?
        super().__init__()

        self.cfg = cfg
        self.num_classes = self.cfg["model"]["num_classes"]
        self.ignore_index = IGNORE_INDEX[self.cfg["data"]["name"]]
        self.vis_step = 0
        self.loss_fn = self.get_loss_fn()

        # If class frequencies are provided, use it to weight the losses.
        if "class_frequencies" in self.cfg["model"]:
            class_frequencies = torch.Tensor(self.cfg["model"]["class_frequencies"])
            self.inv_class_frequencies = class_frequencies.sum() / class_frequencies

        # The following is supposed to be stored in val/tet steps fn respectively
        # in subclasses
        self.val_step_outputs = []
        self.test_step_outputs = []
    def get_loss_fn(self) -> nn.Module:
        """
        Returns the loss function based on the config file.
        """
        loss_name = self.cfg["model"]["loss"]

        if loss_name == Losses.CROSS_ENTROPY:
            return CrossEntropyLoss(
                weight=self.inv_class_frequencies,
                ignore_index=self.ignore_index
                if self.ignore_index is not None
                else -100,
            )
        elif loss_name == Losses.NLL:
            return NLLLoss(
                weight=self.inv_class_frequencies,
                ignore_index=self.ignore_index
                if self.ignore_index is not None
                else -100,
            )
        elif loss_name == Losses.ALEATORIC:
            return AleatoricLoss(
                weight=self.inv_class_frequencies,
                ignore_index=self.ignore_index
                if self.ignore_index is not None
                else -100,
            )
        else:
            raise RuntimeError(f"Loss {loss_name} not available!")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
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


    def validation_epoch_end(self):
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
        self.track_evaluation_metrics(
            conf_matrices,
            stage="Validation",
            calibration_info_list=calibration_info_list,
        )
        self.track_confusion_matrix(conf_matrices, stage="Validation")

        fig_ = metrics.compute_calibration_plots(outputs, num_bins=50)
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

        # TODO: Refactor to log evaluation metrics from conf. matrix.
        self.track_evaluation_metrics(
            conf_matrices,
            stage="Test",
            calibration_info_list=calibration_info_list,
        )

        self.track_confusion_matrix(conf_matrices, stage="Test")

        # TODO: Refactor?
        fig_ = metrics.compute_calibration_plots(outputs, num_bins=50)
        self.logger.experiment.add_figure(
            "UncertaintyStats/Calibration", fig_, self.current_epoch
        )

    def track_evaluation_metrics(  # not refactored
        self,
        conf_matrices: List[torch.Tensor],
        stage: str = "Test",
        calibration_info_list: List[dict] = None,
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

    def track_confusion_matrix(self, conf_matrices: List[torch.Tensor], stage: str="Test"):  # not refactored
        """
        Track the confusion matrix as a figure in seaborn because it looks better.
        """
        # TODO: Refactor, look into this.
        total_conf_matrix = metrics.total_conf_matrix_from_conf_matrices(conf_matrices)
        df_cm = pd.DataFrame(
            total_conf_matrix.cpu().numpy(),
            index=range(self.num_classes),
            columns=range(self.num_classes),
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

    def track_gradient_norms(self):  # not refactored
        """
        Track the gradient norms.
        Debug gradient flow. Common problem RNNs.
        If gradients vanish: norm goes to 0. Mean you don't learn more.
        If gradients explode: norm goes to large number. Mean you always "diverge"
        """
        total_grad_norm = 0
        for params in self.model.parameters():
            if params.grad is not None:
                total_grad_norm += params.grad.data.norm(2).item()

        self.log(f"LossStats/GradientNorm", torch.tensor(total_grad_norm))

    def plot_predictions( # not refactored TODO: Move plot predictions to some utils?
        self,
        images: torch.Tensor,
        hard_predictions: torch.Tensor,
        prob_predictions: torch.Tensor,
        targets: torch.Tensor,
        stage: str = "Test",
        step: int = 0,
        uncertainties=None, # TODO: Unc. are not general.
    ):
        # TODO: Refactor this method to either take one image or sample from the batch. 
        # or support other kind of batch plotting
        sample_img_out = hard_predictions[:1]
        sample_img_out = utils.toOneHot(sample_img_out, self.cfg["data"]["name"])
        self.logger.experiment.add_image(
            f"{stage}/Output image",
            torch.from_numpy(sample_img_out),
            step,
            dataformats="HWC",
        )
        sample_img_in = images[:1]
        sample_anno = targets[:1]

        self.logger.experiment.add_image(
            f"{stage}/Input image", sample_img_in.squeeze(), step, dataformats="CHW"
        )

        sample_prob_prediction = prob_predictions[:1]
        cross_entropy_fn = CrossEntropyLoss(reduction="none")
        sample_error_img = cross_entropy_fn(
            sample_prob_prediction, sample_anno
        ).squeeze()

        sizes = sample_img_out.shape
        px = 1 / plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(px * sizes[1], px * sizes[0]))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.imshow(sample_error_img.cpu().numpy(), cmap="gray")
        fig.add_axes(ax)
        self.logger.experiment.add_figure(f"{stage}/Error image", fig, step)
        plt.cla()
        plt.clf()

        sample_anno = utils.toOneHot(sample_anno, self.cfg["data"]["name"])
        self.logger.experiment.add_image(
            f"{stage}/Annotation",
            torch.from_numpy(sample_anno),
            step,
            dataformats="HWC",
        )

        if uncertainties is not None:
            sample_al_unc_out = uncertainties.cpu().numpy()[0, :, :]
            fig = plt.figure(figsize=(px * sizes[1], px * sizes[0]))
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            ax.imshow(sample_al_unc_out, cmap="plasma")
            fig.add_axes(ax)
            self.logger.experiment.add_figure(f"{stage}/Uncertainty", fig, step)
            plt.cla()
            plt.clf()
