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
from semantic_segmentation.constants import LABELS

class MCDNetwork(BaseNetwork):
    """
    Defines the interface for a aleatoric network.
    """

    def __init__(self, model: nn.Module, cfg: dict) -> None:
        super(MCDNetwork, self).__init__(model, cfg)
        # TODO: This should come from the outside
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Used to obtain probabilities from the logits after sampling
        self.softmax = nn.Softmax(dim=1)
        # Used to obtain the standard deviation from the log std
        self.softplus = nn.Softplus(beta=1)
        self.MC_samples = self.cfg["train"]["MC_samples"]
        self.N_classes = self.cfg["model"]["num_classes"]
        
        self.uncertainty_measure = "trace" #"trace" // "covariance" // "winning_class"
        

    def forward(self, x):
        """
        Forward pass with aleatoric uncertainty
        as an additional output. This only returns the logits.
        Post processing (e.g., softplus, aleatoric dist. sampling)
        is done in get_predictions()
        """
        all_p = torch.zeros((self.MC_samples, self.N_classes, *x.size()[2:]), device=self.device) # (MC_samples, N_classes, H, W)
        for n in range(self.MC_samples):
            all_p[n] = self.softmax(self.model(x))
        p_mean = torch.mean(all_p, dim=0)
        p_var = torch.var(all_p, dim=0, unbiased=False)

        
        if self.uncertainty_measure == "trace":
            p_var = torch.var(all_p, dim=0, unbiased=False) # (N_classes, H, W)
            p_trace = torch.sum(p_var, dim=0) # (H, W)
            output_var = p_trace
            
        elif self.uncertainty_measure == "winning_class":
            p_var = torch.var(all_p, dim=0, unbiased=False) # (N_classes, H, W)
            winning_class = torch.argmax(p_mean, dim=0).unsqueeze(0) # (H, W)
            var_winning_class = torch.gather(p_var, 0, winning_class).squeeze() # (H, W)
            output_var = var_winning_class
            
        
        elif self.uncertainty_measure == "covariance":
            cov_matrix = torch.zeros((*x.size()[2:], self.N_classes, self.N_classes), device=self.device) # (H, W)
            for n in range(self.MC_samples):
                v_e = torch.permute((all_p[n, :, :, :] - p_mean), (1, 2, 0))
                term_1 = torch.unsqueeze(v_e, axis = 3)
                term_2 = torch.unsqueeze(v_e, axis = 2)
                epist_n = torch.matmul(term_1, term_2)
                cov_matrix += epist_n
            cov_matrix = cov_matrix / self.MC_samples
            epist_trace = torch.diagonal(cov_matrix, dim1 = 2, dim2 = 3).permute(2, 0, 1)
            
        return p_mean, output_var


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
        #This only wotks with batch size == 1
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                module.train() #We activate dropout layers
                
        p_mean, var = self.forward(data)
        _, pred_label = torch.max(p_mean, dim=0)
        return p_mean.unsqueeze(0), pred_label.unsqueeze(0), var.unsqueeze(0)


class MCDNetworkWrapper(NetworkWrapper):
    """
    Network Wrapper to include aleatoric uncertainty as an additional output.
    - We use get_predictions to get the probabilities. This is not required in traning.
    - The uncertainty in this case is the ... (entropy...?) and viualized.
    TODO: clarify extracted from aleatoric NN.
    """

    def __init__(self, network: MCDNetwork, cfg: dict) -> None:
        super(MCDNetworkWrapper, self).__init__(network, cfg)
        self.network = network  # For typing purposes
        self.save_hyperparameters()
        self.vis_interval = self.cfg["train"]["visualization_interval"]
        
        self.class_names = LABELS[self.cfg["data"]["name"]].keys()
        self.output_classes = ["background", "sofa", "pottedplant", "bottle", "chair", "diningtable", "tvmonitor"]
        self.plot_epist_TP = {class_name: [] for class_name in self.class_names}
        self.plot_epist_FP = {class_name: [] for class_name in self.class_names}
        self.plot_epist_TN = {class_name: [] for class_name in self.class_names}
        self.plot_epist_FN = {class_name: [] for class_name in self.class_names}

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
        mean_probs, pred_label, variance = self.network.get_predictions(batch["data"])
        loss = self.loss_fn(mean_probs, true_label)
        
        for c in range(self.network.N_classes):  
            TP_pixels_c = (true_label == c) & (true_label == pred_label)
            TN_pixels_c = (true_label != c) & (true_label != pred_label)
            FN_pixels_c = (true_label == c) & (true_label != pred_label)
            FP_pixels_c = (pred_label == c) & (true_label != pred_label)

            class_name = list(self.class_names)[c]
            if class_name in self.output_classes:
   
                if TP_pixels_c.any():
                    self.plot_epist_TP[class_name].extend(variance[TP_pixels_c].tolist())
                if FN_pixels_c.any():
                    self.plot_epist_FN[class_name].extend(variance[FN_pixels_c].tolist())
                #if TN_pixels_c.any():
                #    self.plot_epist_TN[class_name].extend(variance[TN_pixels_c].tolist())
                if FP_pixels_c.any():
                    self.plot_epist_FP[class_name].extend(variance[FP_pixels_c].tolist())

        confusion_matrix = torchmetrics.functional.confusion_matrix(
            pred_label, true_label, task="multiclass", num_classes=self.network.num_classes, normalize=None)
        
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
