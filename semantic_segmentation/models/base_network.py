"""
    Code for the NetworkWrapper class.
    This class configures the model and pipeline stages.
    This is: training, validation, testing, visualization and logging.
    It obtains the information from the config file.
    Common logging methods that apply to all methods are included here.
    When a method varies the pipeline execution (e.g. mc samples, ensembles, etc.)
    it is moved to a subclass that modifies the corresponding functions.
    The code is based on: https://github.com/dmar-bonn/bayesian_erfnet/
    and discussed with Julius Rückin and Liren Jin.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import semantic_segmentation.utils.utils as utils
from semantic_segmentation.constants import IGNORE_INDEX
from semantic_segmentation.constants import LABELS
from semantic_segmentation.models import get_loss_fn
from semantic_segmentation.models.loss import CrossEntropyLoss
from semantic_segmentation.utils import metrics
import wandb
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from semantic_segmentation.models import NetworkType

from torch.optim.lr_scheduler import _LRScheduler, StepLR



class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
                for base_lr in self.base_lrs]

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

    def __init__(self, network: NetworkType, cfg: dict):  # TODO (later): define a cfg dataclass ?
        super(NetworkWrapper, self).__init__()
        self.cfg = cfg
        self.network = network
        
        self.model = self.network.model  # Lightning requires the model to be an attribute
        self.ignore_index = IGNORE_INDEX[cfg["data"]["name"]]
        self.loss_fn = get_loss_fn(cfg)
        self.vis_step = 0

        # The following is supposed to be stored in val/tet steps
        # fn respectively in subclasses
        self.val_step_outputs = []
        self.test_step_outputs = []
        
        #Visualize the calibration plots
        self.LABELS_COLORS = LABELS[self.cfg["data"]["name"]]
        for c in self.LABELS_COLORS:
            color = self.LABELS_COLORS[c]['color']
            self.LABELS_COLORS[c]['color'] = tuple([c / 255.0 for c in color])

    def load_pretrained_weights(self, checkpoint_url):
        """
        Load pretrained weights from checkpoint_url
        """
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url, progress=True)
        self.model.load_state_dict(checkpoint)
        print('Pretrained weights loaded from:', checkpoint_url)
    


    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        From LightningModule.
        Returns the optimizer based on the config file.
        TODO (later): do we add more optimizers?
        """
        #optimizer = torch.optim.AdamW(
        #    self.parameters(), lr=self.cfg["train"]["lr"], weight_decay=self.cfg["train"]["weight_decay"]
        #)
        #param_groups = [
        #    {"params": self.model.backbone.parameters(), "lr": self.cfg["train"]["lr"]},
        #    {"params": self.model.classifier.parameters(), "lr": self.cfg["train"]["lr"] * 10},
        #]
        #optimizer = torch.optim.SGD(
        #    param_groups, weight_decay=self.cfg["train"]["weight_decay"], momentum = 0.9, nesterov = False,
        #)
        #optimizer = torch.optim.Adam(params = [
        #{'params': self.model.backbone.parameters(), 'lr': 1e-5},
        #{'params': self.model.classifier.parameters(), 'lr': 1e-5}], lr = 1e-5, weight_decay = 5e-4) #Antes estaba en 1e-4
        optimizer = torch.optim.SGD(params=[{'params': self.model.backbone.parameters(), 'lr': 1 * 0.01},
                                            {'params': self.model.classifier.parameters(), 'lr': 0.01},], 
                                            lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = PolyLR(optimizer, 30, power=0.9)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer=my_optimizer, step_size=10, gamma=0.1)
        #return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # "step" para por paso, "epoch" para por época
                "frequency": 1,       # Frecuencia de actualización
                "reduce_on_plateau": False,  # Si es para ReduceLROnPlateau
                # "monitor": "val_loss",  # Metrica a monitorear si reduce_on_plateau es True
            }
        }

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
        loss_epoch = torch.stack([tmp["loss"] for tmp in outputs]).mean()
        self.log("Validation_epoch/loss", loss_epoch)
        self.log_classification_metrics(
            conf_matrices,
            stage="Validation",
            calibration_info_list=calibration_info_list,
        )
        #self.log_confusion_matrix(conf_matrices, stage="Validation")
        #self.log_calibration_plots(calibration_info_list)

        #fig_ = metrics.compute_calibration_plots(calibration_info_list, num_bins=50)
        #self.logger.experiment.add_figure("UncertaintyStats/Calibration", fig_, self.current_epoch)

        self.vis_step = 0
    
    def max_in_tensor_list(self, tensor_list):
        if tensor_list:  # Asegurarse de que la lista no esté vacía
            # Concatenar todos los tensores en la lista a lo largo de una nueva dimensión
            combined_tensor = torch.cat([t.flatten() for t in tensor_list])
            # Calcular el máximo
            return combined_tensor.max().item()
        else:
            return float('nan')  # Retorna NaN si la lista está vacía

    def on_test_epoch_end(self):
        """
        Log the confusion matrix and calibration info at the end of the evaluation epoch.
        We track... TODO: what?
        WARN: This method is on_validation_epoch_end in new Lighting versions. Refactoring requires
              storing outputs manually as an atribute in validation_step.
        WARN: This method asumes that the children will have "conf_matrix" and "calibration_info"
              in their outputs.
        """
        outputs = self.test_step_outputs
        
    
        for tmp in outputs:
            cal = tmp['calibration_info']
        conf_matrices = [tmp["conf_matrix"] for tmp in outputs]
        calibration_info_list = [tmp["calibration_info"] for tmp in outputs]

        self.log_classification_metrics(
            conf_matrices,
            stage="Test",
            calibration_info_list=calibration_info_list,
        )
        self.show_uncertainty_diagram(self.plot_epist_TP, 'TP_val')
        self.show_uncertainty_diagram(self.plot_epist_FP, 'FP_val')
        #self.show_uncertainty_diagram(self.plot_epist_TN, 'TN_val')
        self.show_uncertainty_diagram(self.plot_epist_FN, 'FN_val')
        self.log_calibration_plots(calibration_info_list)
    
    def show_uncertainty_diagram(self, uncertainty, name):
        
        # Configurar el tamaño de la figura
        plt.figure(figsize=(10, 6))

        # Número de bins para el histograma
        bins = 30

        x = np.linspace(0, 0.25, 100)
        for class_name, values in uncertainty.items():
            sorted_values = np.sort(values)
            cumulative_frequency = np.searchsorted(sorted_values, x, side='right') / len(values)
            cumulative_frequency *= 100
            plt.plot(x, cumulative_frequency, label=class_name, color=self.LABELS_COLORS[class_name]['color'])

        plt.title(name + ' histogram')
        plt.xlabel('Epistemic uncertainty')
        plt.ylabel('Percentage of pixels')
        plt.legend()  # Añadir leyenda para identificar cada clase
        
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        plt.close()
        
        self.logger.experiment.log({
            f"Epistemic_uncertainty/{name}": wandb.Image(image),
            "epoch": self.current_epoch
        })

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
        #miou = metrics.mean_iou_from_conf_matrices(conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]])
        miou = metrics.mean_iou_from_conf_matrices(conf_matrices, ignore_index=None)
        per_class_iou = metrics.per_class_iou_from_conf_matrices(
            conf_matrices, ignore_index=None
        )
        accuracy = metrics.accuracy_from_conf_matrices(
            conf_matrices, ignore_index=None
        )
        precision = metrics.precision_from_conf_matrices(
            conf_matrices, ignore_index=None
        )
        recall = metrics.recall_from_conf_matrices(conf_matrices, ignore_index=None)
        f1_score = metrics.f1_score_from_conf_matrices(
            conf_matrices, ignore_index=None
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

    def log_confusion_matrix(self, conf_matrices: List[torch.Tensor], stage: str = "Test"):  # not refactored
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

        self.logger.experiment.add_figure(f"ConfusionMatrix/{stage}", ax.get_figure(), self.current_epoch)

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
        #if not isinstance(fig_, list):
        #    fig_ = [fig_]
        fig_wandb = wandb.Image(fig_)
        self.logger.experiment.log({f"UncertaintyStats/Calibration": fig_wandb})
        #self.logger.experiment.log("UncertaintyStats/Calibration", fig_wandb, self.current_epoch)

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

    def log_prediction_images(  # not refactored TODO: Move plot predictions to some utils?
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
        img_denorm = utils.denormalize_image(image.cpu())
        img_wandb = wandb.Image(img_denorm.squeeze().permute(1, 2, 0).numpy())
        #self.logger.experiment.add_image(f"{stage}/Input image", img_denorm, step, dataformats="CHW")
        self.logger.experiment.log({f"{stage}/Input image": img_wandb})

        # Plot the output image as a label mask TODO: toOneHot transforms to detach.cpu.numpy(). fix.
        imap_pred = utils.toOneHot(argmax_pred, self.cfg["data"]["name"])
        imap_pred_wandb = wandb.Image(imap_pred)
        self.logger.experiment.log({f"{stage}/Output image": imap_pred_wandb})
        
        #self.logger.experiment.add_image(
        #    f"{stage}/Output image",
        #    torch.from_numpy(imap_pred),
        #    step,
        #    dataformats="HWC",
        #)

        # Plot the ground truth label as a label mask
        imap_true = utils.toOneHot(true_label, self.cfg["data"]["name"])
        imap_true_wandb = wandb.Image(imap_true)
        self.logger.experiment.log({f"{stage}/Annotation": imap_true_wandb})
        #self.logger.experiment.add_image(
        #    f"{stage}/Annotation",
        #    torch.from_numpy(imap_true),
        #    step,
        #    dataformats="HWC",
        #)
        
        """
        # Plot the error image with the cross entropy loss
        # This needs the images to be in batch format
        # TODO: Change to configured loss?
        prob_pred_batch = prob_pred.unsqueeze(0)
        true_label_batch = true_label.unsqueeze(0)

        cross_entropy_fn = CrossEntropyLoss(reduction="none")
        sample_error_img = cross_entropy_fn(prob_pred_batch, true_label_batch).squeeze()
        sizes = imap_pred.shape  # TODO: what is this for?
        px = 1 / plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(px * sizes[1], px * sizes[0]))
        ax = plt.Axes(fig, (0.0, 0.0, 1.0, 1.0))
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
            ax = plt.Axes(fig, (0.0, 0.0, 1.0, 1.0))
            ax.set_axis_off()
            ax.imshow(unc_np, cmap="plasma")
            fig.add_axes(ax)
            self.logger.experiment.add_figure(f"{stage}/Uncertainty", fig, step)
            plt.cla()
            plt.clf()        
        """

        