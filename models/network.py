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
from models.erfnet import ERFNetModel, AleatoricERFNetModel
from models.unet import UNetModel, AleatoricUNetModel


class NetworkWrapper(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_classes = self.cfg["model"]["num_classes"]
        self.ignore_index = IGNORE_INDEX[self.cfg["data"]["name"]]
        self.vis_step = 0
        self.loss_fn = self.get_loss_fn()

    def get_loss_fn(self):
        loss_name = self.cfg["model"]["loss"]

        if loss_name == Losses.CROSS_ENTROPY:
            return CrossEntropyLoss(
                weight=torch.tensor([0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                ignore_index=self.ignore_index
                if self.ignore_index is not None
                else -100,
            )
        elif loss_name == Losses.NLL:
            return NLLLoss(
                weight=torch.tensor([0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                ignore_index=self.ignore_index
                if self.ignore_index is not None
                else -100,
            )
        elif loss_name == Losses.ALEATORIC:
            return AleatoricLoss(
                weight=torch.tensor([0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                ignore_index=self.ignore_index
                if self.ignore_index is not None
                else -100,
            )
        else:
            raise RuntimeError(f"Loss {loss_name} not available!")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg["train"]["lr"],
            weight_decay=self.weight_decay,
        )
        return optimizer

    def validation_epoch_end(self, outputs):
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

    def test_epoch_end(self, outputs):
        conf_matrices = [tmp["conf_matrix"] for tmp in outputs]
        calibration_info_list = [tmp["calibration_info"] for tmp in outputs]

        self.track_evaluation_metrics(
            conf_matrices,
            stage="Test",
            calibration_info_list=calibration_info_list,
        )

        self.track_confusion_matrix(conf_matrices, stage="Test")

        fig_ = metrics.compute_calibration_plots(outputs, num_bins=50)
        self.logger.experiment.add_figure(
            "UncertaintyStats/Calibration", fig_, self.current_epoch
        )

    def track_evaluation_metrics(
        self,
        conf_matrices,
        stage="Test",
        calibration_info_list=None,
    ):
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

    def track_confusion_matrix(self, conf_matrices, stage="Validation"):
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

    def track_gradient_norms(self):
        total_grad_norm = 0
        for params in self.model.parameters():
            if params.grad is not None:
                total_grad_norm += params.grad.data.norm(2).item()

        self.log(f"LossStats/GradientNorm", torch.tensor(total_grad_norm))

    def plot_predictions(
        self,
        images,
        hard_predictions,
        prob_predictions,
        targets,
        stage="Train",
        step=0,
        uncertainties=None,
    ):
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

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def test_step(self):
        pass

    def visualize_step(self):
        pass

    @property
    def weight_decay(self):
        return self.cfg["train"]["weight_decay"]


class Network(NetworkWrapper):
    def __init__(self, name, cfg):
        super(Network, self).__init__(cfg)

        self.save_hyperparameters()
        self.num_mc_aleatoric = self.cfg["train"]["num_mc_aleatoric"]
        self.vis_interval = self.cfg["train"]["visualization_interval"]
        self.softmax = nn.Softmax(dim=1)

        if name == "erfnet":
            self.model = ERFNetModel(self.num_classes)
        elif name == "unet":
            self.model = UNetModel(self.num_classes)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        true_label = batch["anno"]
        logits = self.forward(batch["data"])
        loss = self.loss_fn(logits, true_label)

        self.track_gradient_norms()
        self.log("train:loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        true_label = batch["anno"]
        logits, probs, pred_label, _ = self.get_predictions(batch["data"])
        loss = self.loss_fn(logits, true_label)

        confusion_matrix = torchmetrics.functional.confusion_matrix(
            pred_label, true_label, num_classes=self.num_classes, normalize=None
        )
        calibration_info = metrics.compute_calibration_info(
            probs, true_label, num_bins=50
        )

        if batch_idx % self.vis_interval == 0:
            self.visualize_step(batch)

        self.log("validation:loss", loss, prog_bar=True)
        return {
            "conf_matrix": confusion_matrix,
            "loss": loss,
            "calibration_info": calibration_info,
        }

    def test_step(self, batch, batch_idx):
        true_label = batch["anno"]
        logits, probs, pred_label, _ = self.get_predictions(batch["data"])
        loss = self.loss_fn(logits, true_label)

        confusion_matrix = torchmetrics.functional.confusion_matrix(
            pred_label, true_label, num_classes=self.num_classes, normalize=None
        )
        calibration_info = metrics.compute_calibration_info(
            probs, true_label, num_bins=50
        )
        self.log("test:loss", loss, prog_bar=True)
        return {
            "conf_matrix": confusion_matrix,
            "loss": loss,
            "calibration_info": calibration_info,
        }

    def visualize_step(self, batch):
        self.vis_step += 1
        true_label = batch["anno"]
        logits, probs, pred_label, unc = self.get_predictions(batch["data"])

        self.plot_predictions(
            batch["data"],
            pred_label,
            probs,
            true_label,
            stage="Validation",
            step=self.vis_step,
            uncertainties=unc,
        )

    @torch.no_grad()
    def get_predictions(self, data):
        self.model.eval()
        logits = self.forward(data)
        probs = self.softmax(logits)
        _, pred_label = torch.max(probs, dim=1)

        unc = -torch.sum(probs * torch.log(probs + 10 ** (-8)), dim=1) / torch.log(
            torch.tensor(self.num_classes)
        )

        return logits, probs, pred_label, unc


class AleatoricNetwork(NetworkWrapper):
    def __init__(self, name, cfg):
        super(AleatoricNetwork, self).__init__(cfg)

        self.save_hyperparameters()
        self.num_mc_aleatoric = self.cfg["train"]["num_mc_aleatoric"]
        self.vis_interval = self.cfg["train"]["visualization_interval"]
        self.softmax = nn.Softmax(dim=1)

        if name == "erfnet":
            self.model = AleatoricERFNetModel(self.num_classes)
        elif name == "unet":
            self.model = AleatoricUNetModel(self.num_classes)

    def forward(self, x):
        est_seg, est_std = self.model(x)
        return est_seg, est_std

    def sample_from_aleatoric_model(self, seg, std):
        sampled_probs = torch.zeros(
            (self.num_mc_aleatoric, *seg.size()), device=self.device
        )
        noise_mean = torch.zeros(seg.size(), device=self.device)
        noise_std = torch.ones(seg.size(), device=self.device)
        dist = torch.distributions.normal.Normal(noise_mean, noise_std)
        for i in range(self.num_mc_aleatoric):
            epsilon = dist.sample()
            sampled_logits = seg + torch.mul(std, epsilon)
            sampled_probs[i] = self.softmax(sampled_logits)
        mean_probs = torch.mean(sampled_probs, dim=0)
        return mean_probs

    def training_step(self, batch, batch_idx):
        true_label = batch["anno"]
        est_seg, est_std = self.forward(batch["data"])
        mean_probs = self.sample_from_aleatoric_model(est_seg, est_std)
        loss = self.loss_fn(mean_probs, true_label)

        self.track_uncertainty_stats(est_std)
        self.track_gradient_norms()
        self.log("train:loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        true_label = batch["anno"]
        mean_probs, pred_label, _ = self.get_predictions(batch["data"])
        loss = self.loss_fn(mean_probs, true_label)

        confusion_matrix = torchmetrics.functional.confusion_matrix(
            pred_label, true_label, num_classes=self.num_classes, normalize=None
        )
        calibration_info = metrics.compute_calibration_info(
            mean_probs, true_label, num_bins=50
        )
        if batch_idx % self.vis_interval == 0:
            self.visualize_step(batch)

        self.log("validation:loss", loss, prog_bar=True)
        return {
            "conf_matrix": confusion_matrix,
            "loss": loss,
            "calibration_info": calibration_info,
        }

    def test_step(self, batch, batch_idx):
        true_label = batch["anno"]
        mean_probs, pred_label, _ = self.get_predictions(batch["data"])
        loss = self.loss_fn(mean_probs, true_label)

        confusion_matrix = torchmetrics.functional.confusion_matrix(
            pred_label, true_label, num_classes=self.num_classes, normalize=None
        )
        calibration_info = metrics.compute_calibration_info(
            mean_probs, true_label, num_bins=50
        )
        self.log("test:loss", loss, prog_bar=True)
        return {
            "conf_matrix": confusion_matrix,
            "loss": loss,
            "calibration_info": calibration_info,
        }

    def visualize_step(self, batch):
        self.vis_step += 1
        true_label = batch["anno"]
        final_prob, pred_label, aleatoric_unc = self.get_predictions(batch["data"])

        self.plot_predictions(
            batch["data"],
            pred_label,
            final_prob,
            true_label,
            stage="Validation",
            step=self.vis_step,
            uncertainties=aleatoric_unc,
        )

    @torch.no_grad()
    def get_predictions(self, data):
        self.model.eval()
        est_seg, est_std = self.forward(data)
        mean_probs = self.sample_from_aleatoric_model(est_seg, est_std)

        _, pred_label = torch.max(mean_probs, dim=1)

        aleatoric_unc = -torch.sum(
            mean_probs * torch.log(mean_probs + 10 ** (-8)), dim=1
        ) / torch.log(torch.tensor(self.num_classes))

        return mean_probs, pred_label, aleatoric_unc

    def track_uncertainty_stats(self, std):
        self.log("Variance/TrainMin", torch.min(std))
        self.log("Variance/TrainMax", torch.max(std))
        self.log("Variance/TrainMean", torch.mean(std))
