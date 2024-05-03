from typing import Dict, Tuple
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchvision import transforms
from semantic_segmentation.constants import LABELS, THEMES
import cv2
from PIL import Image
import os

#Dictionary in BGR colors
nyu40_colors = {1: [102, 179, 92], # wall
                2: [14, 106, 71], # floor
                3: [188, 20, 102], # cabinet
                4: [121, 210, 214], # bed
                5: [74, 202, 87], # chair
                6: [116, 99, 103], # sofa
                7: [151, 130, 149], # table
                8: [52, 1, 87], # door
                9: [235, 157, 37], # window
                10: [129, 191, 187], # bookshelf
                11: [20, 160, 203], # picture
                12: [57, 21, 252], # counter
                13: [235, 88, 48], # blinds
                14: [218, 58, 254], # desk
                15: [169, 255, 219], # shelves
                16: [187, 207, 14], # curtain
                17: [189, 189, 174], # dresser
                18: [189, 50, 107], # pillow
                19: [54, 243, 63], # mirror
                20: [248, 130, 228], # floor mat
                21: [50, 134, 20], # clothes
                22: [72, 166, 17], # ceiling
                23: [131, 88, 59], # books
                24: [13, 241, 249], # refrigerator
                25: [8, 89, 52], # television
                26: [129, 83, 91], # paper
                27: [110, 187, 198], # towel
                28: [171, 252, 7], # shower curtain
                29: [174, 34, 205], # box
                30: [80, 163, 49], # whiteboard
                31: [103, 131, 1], # person
                32: [253, 133, 53], # night stand
                33: [105, 3, 53], # toilet
                34: [220, 190, 145], # sink
                35: [217, 43, 161], # lamp
                36: [201, 189, 227], # bathtub
                37: [13, 94, 47], # bag
                38: [14, 199, 205], # otherstructure
                39: [214, 251, 248], # otherfurniture
                40: [189, 39, 212], # otherprop
                }

rgb_mean = np.array([0.485, 0.456, 0.406])
rgb_std = np.array([0.229, 0.224, 0.225])

def paint_labels_and_image(img, label, sample_number):
    show_dir = '/home/ego_exo4d/Documents/dl_tools_loren/semantic_segmentation/show_dir'
    label = label.squeeze(0).numpy()
    label = np.uint8(label)
    label_bgr = np.zeros((label.shape[0], label.shape[1], 3), np.uint8)
    for i in range(1, 41):
        mask = label == i
        label_bgr[:, :, 0][mask] = nyu40_colors[i][0]
        label_bgr[:, :, 1][mask] = nyu40_colors[i][1]
        label_bgr[:, :, 2][mask] = nyu40_colors[i][2]
    cv2.imwrite(os.path.join(show_dir, f'label_{sample_number}.png'), label_bgr)

    img = img.permute(1, 2, 0).numpy()
    img = img * rgb_std + rgb_mean
    img = np.uint8(img * 255)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(show_dir, f'image_{sample_number}.png'), img_bgr)
    

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

def denormalize_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    mean_torch = torch.tensor(mean).view(-1, 1, 1)
    std_torch = torch.tensor(std).view(-1, 1, 1)
    image = image * std_torch + mean_torch
    return image

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
