"""
    This file works as a module factory.
    It returns the correct modules based on the config file.
    In our implementation, the different modules are:
    - Model: The network architecture
    - Network: The type of network  wrapper (e.g., deterministic, aleatoric)
    - Loss: The loss function
"""

from typing import Union
import torch
from pytorch_lightning import LightningModule

from semantic_segmentation.constants import Losses, IGNORE_INDEX
from semantic_segmentation.models.loss import CrossEntropyLoss, NLLLoss, AleatoricLoss

def get_loss_fn(cfg) -> torch.nn.Module:
        """
        Returns the loss function based on the config file.
        """
        loss_name = cfg["model"]["loss"]

        # If class frequencies are provided, use it to weight the losses.
        if "class_frequencies" in cfg["model"]:
            class_frequencies = torch.Tensor(cfg["model"]["class_frequencies"])
            inv_class_frequencies = class_frequencies.sum() / class_frequencies
        else:
            inv_class_frequencies = None
            
        ignore_index = IGNORE_INDEX[cfg["data"]["name"]]

        if loss_name == Losses.CROSS_ENTROPY:
            return CrossEntropyLoss(
                weight=inv_class_frequencies,
                ignore_index=ignore_index
                if ignore_index is not None
                else -100,
            )
        elif loss_name == Losses.NLL:
            return NLLLoss(
                weight=inv_class_frequencies,
                ignore_index=ignore_index
                if ignore_index is not None
                else -100,
            )
        elif loss_name == Losses.ALEATORIC:
            return AleatoricLoss(
                weight=inv_class_frequencies,
                ignore_index=ignore_index
                if ignore_index is not None
                else -100,
            )
        else:
            raise RuntimeError(f"Loss {loss_name} not available!")

from semantic_segmentation.models.deterministic_network import DeterministicNetwork, DeterministicNetworkWrapper
from semantic_segmentation.models.aleatoric_network import AleatoricNetwork, AleatoricNetworkWrapper
from semantic_segmentation.models.erfnet.erfnet import ERFNetModel
from semantic_segmentation.models.erfnet.aleatoric_erfnet import AleatoricERFNetModel
from semantic_segmentation.models.unet.unet import UNetModel
from semantic_segmentation.models.unet.aleatoric_unet import AleatoricUNetModel

from torchvision.models.segmentation.deeplabv3 import deeplabv3_resnet50

NetworkWrapperType = Union['DeterministicNetworkWrapper', 'AleatoricNetworkWrapper']
NetworkType = Union['DeterministicNetwork', 'AleatoricNetwork']
ModelType = Union['ERFNetModel', 'AleatoricERFNetModel', 'UNetModel', 'AleatoricUNetModel']
LossType = Union['CrossEntropyLoss', 'NLLLoss', 'AleatoricLoss']

# TODO: Probably, the concept of model should change to "pipeline" or something similar.
network_wrapper = {
    "deterministic": DeterministicNetworkWrapper,
    "aleatoric": AleatoricNetworkWrapper,
}

networks = {
    "deterministic": DeterministicNetwork,
    "aleatoric": AleatoricNetwork,
}

deterministic_models = {
    "erfnet": ERFNetModel,
    "unet": UNetModel,
}

aleatoric_models = {
    "erfnet": AleatoricERFNetModel,
    "unet": AleatoricUNetModel,
}

models_dict = {
    "deterministic": deterministic_models,
    "aleatoric": aleatoric_models,
}

losses = {
    "cross_entropy": CrossEntropyLoss,
    "nll": NLLLoss,
    "aleatoric": AleatoricLoss,
}

def get_model(cfg) -> LightningModule:
    network_type = cfg["model"]["type"]

    if isinstance(cfg, dict):
        model_name = cfg["model"]["name"]
        network_type = cfg["model"]["type"]
        
        models_for_type = models_dict[network_type]        
        model = models_for_type[model_name](cfg["model"]["num_classes"])        
        network = networks[network_type](model, cfg)        
        network_wrapper_class = network_wrapper[network_type]        
        
        return network_wrapper_class(network, cfg)
    
    else:
        raise RuntimeError(f"{type(cfg)} not a valid config")
