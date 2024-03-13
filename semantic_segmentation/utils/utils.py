from typing import Dict, Tuple
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchvision import transforms
from semantic_segmentation.constants import LABELS, THEMES


def imap2rgb(imap, channel_order, theme):
    """converts an iMap label image into a RGB Color label image,
    following label colors/ids stated in the "labels" dict.

    Arguments:
        imap {numpy with shape (h,w)} -- label image containing label ids [int]
        channel_order {str} -- channel order ['hwc' for shape(h,w,3) or 'chw' for shape(3,h,w)]
        theme {str} -- label theme

    Returns:
        float32 numpy with shape (channel_order) -- rgb label image containing label colors from dict (int,int,int)
    """
    assert channel_order == "hwc" or channel_order == "chw"
    assert len(imap.shape) == 2
    assert theme in LABELS.keys()

    rgb = np.zeros((imap.shape[0], imap.shape[1], 3), np.float32)
    for _, cl in LABELS[theme].items():  # loop each class label
        if cl["color"] == (0, 0, 0):
            continue  # skip assignment of only zeros
        mask = imap == cl["id"]
        rgb[:, :, 0][mask] = cl["color"][0]
        rgb[:, :, 1][mask] = cl["color"][1]
        rgb[:, :, 2][mask] = cl["color"][2]
    if channel_order == "chw":
        rgb = np.moveaxis(rgb, -1, 0)  # convert hwc to chw
    return rgb


# TODO: Fix this
def toOneHot(tensor, dataset_name):
    img = tensor
    if len(img.shape) == 3:
        img = np.transpose(img, (1, 2, 0))
        img = np.argmax(img, axis=-1)

    img = imap2rgb(img, channel_order="hwc", theme=THEMES[dataset_name])
    return img.astype(np.uint8)


def enable_dropout(model: nn.Module):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


# def compute_prediction_stats(predictions, normalized=False):
#     mean_predictions = torch.mean(predictions, dim=0)
#     class_num = mean_predictions.shape[1]
#     variance_predictions = torch.var(predictions, dim=0)
#     if variance_predictions.shape[1] == 1:
#         variance_predictions = variance_predictions.squeeze(dim=1)
#     if normalized:
#         entropy_predictions = -torch.sum(
#             mean_predictions * torch.log(mean_predictions + 10 ** (-8)), dim=1
#         ) / torch.log(torch.tensor(class_num))

#         mutual_info_predictions = entropy_predictions - torch.mean(
#             torch.sum(-predictions * torch.log(predictions + 10 ** (-8)), dim=2)
#             / torch.log(torch.tensor(class_num)),
#             dim=0,
#         )
#     else:
#         entropy_predictions = -torch.sum(
#             mean_predictions * torch.log(mean_predictions + 10 ** (-8)), dim=1
#         )
#         mutual_info_predictions = entropy_predictions - torch.mean(
#             torch.sum(-predictions * torch.log(predictions + 10 ** (-8)), dim=2), dim=0
#         )
#     return (
#         mean_predictions,
#         variance_predictions,
#         entropy_predictions,
#         mutual_info_predictions,
#     )


# def get_predictions(
#     model,
#     batch,
#     num_mc_aleatoric=50,
#     device=None,
# ):
#     use_mc_dropout = num_mc_dropout > 1
#     num_mc_dropout = num_mc_dropout if num_mc_dropout > 1 else 1
#     num_predictions = num_mc_dropout

#     softmax = nn.Softmax(dim=1)
#     prob_predictions = []
#     aleatoric_unc_predictions = []

#     single_model = model.to(device)
#     single_model.eval()
#     if use_mc_dropout:
#         enable_dropout(single_model)

#     for i in range(num_predictions):
#         with torch.no_grad():
#             if aleatoric_model:
#                 (
#                     est_prob,
#                     est_aleatoric_unc,
#                 ) = sample_from_aleatoric_classification_model(
#                     single_model,
#                     batch,
#                     num_mc_aleatoric=num_mc_aleatoric,
#                     device=device,
#                 )
#             else:
#                 est_prob, _ = single_model.forward(batch["data"])
#                 est_prob, est_aleatoric_unc = softmax(est_prob), torch.zeros_like(
#                     est_prob[:, 0, :, :], device=device
#                 )

#             prob_predictions.append(est_prob)
#             aleatoric_unc_predictions.append(est_aleatoric_unc.squeeze(1))

#     prob_predictions = torch.stack(prob_predictions)
#     aleatoric_unc_predictions = torch.stack(aleatoric_unc_predictions)

#     (
#         prob_predictions,
#         epistemic_variance_predictions,
#         epistemic_entropy_predictions,
#         epistemic_mutual_info_predictions,
#     ) = compute_prediction_stats(prob_predictions)
#     aleatoric_unc_predictions = torch.mean(aleatoric_unc_predictions, dim=0)

#     epistemic_unc_predictions = (
#         epistemic_mutual_info_predictions
#         if use_mc_dropout
#         else epistemic_entropy_predictions
#     )

#     return (
#         prob_predictions,
#         epistemic_unc_predictions,
#         aleatoric_unc_predictions,
#     )


# def sample_from_aleatoric_classification_model(
#     model: LightningModule,
#     batch: Dict,
#     num_mc_aleatoric: int = 50,
#     device: torch.device = None,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     softmax = nn.Softmax(dim=1)
#     est_seg, est_std, _ = model.forward(batch["data"])

#     sampled_predictions = torch.zeros(
#         (num_mc_aleatoric, *est_seg.size()), device=device
#     )
#     noise_mean = torch.zeros(est_seg.size(), device=device)
#     noise_std = torch.ones(est_seg.size(), device=device)
#     dist = torch.distributions.normal.Normal(noise_mean, noise_std)
#     for j in range(num_mc_aleatoric):
#         epsilon = dist.sample()
#         sampled_seg = est_seg + torch.mul(est_std, epsilon)
#         sampled_predictions[j] = softmax(sampled_seg)
#     mean_predictions, _, entropy_predictions, _ = compute_prediction_stats(
#         sampled_predictions
#     )
#     return (
#         mean_predictions,
#         entropy_predictions,
#     )


# def infer_anno_and_epistemic_uncertainty_from_image(
#     model: LightningModule,
#     image: np.array,
#     num_mc_epistemic: int = 25,
#     resize_image: bool = False,
#     aleatoric_model: bool = True,
#     num_mc_aleatoric: int = 50,
#     ensemble_model: bool = False,
#     task: str = "classification",
# ) -> Tuple[np.array, np.array, np.array]:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     to_normalized_tensor = transforms.ToTensor()
#     image_tensor = to_normalized_tensor(image)

#     if resize_image:
#         image_tensor = resize(
#             image_tensor,
#             width=344,
#             height=None,
#             interpolation=1,
#             keep_aspect_ratio=True,
#         )

#     image_batch = {"data": image_tensor.float().unsqueeze(0).to(device)}
#     mean_predictions, uncertainty_predictions, hidden_representations = get_predictions(
#         model,
#         image_batch,
#         num_mc_dropout=num_mc_epistemic,
#         aleatoric_model=aleatoric_model,
#         num_mc_aleatoric=num_mc_aleatoric,
#         ensemble_model=ensemble_model,
#         device=device,
#         task=task,
#     )

#     return (
#         np.squeeze(mean_predictions, axis=0),
#         np.squeeze(uncertainty_predictions, axis=0),
#         np.squeeze(hidden_representations, axis=0),
#     )
