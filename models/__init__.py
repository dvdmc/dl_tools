"""
    This file works as a module factory.
    It returns the correct modules based on the config file.
    In our implementation, the different modules are:
    - Model: The network architecture
    - Network: The type of network  wrapper (e.g., deterministic, aleatoric)
    - Loss: The loss function
"""

import torch
from models.deterministic_network import DeterministicNetwork
from models.aleatoric_network import AleatoricNetwork
from pytorch_lightning import LightningModule

from constants import Losses, IGNORE_INDEX
from models.deterministic_network import DeterministicNetwork
from models.aleatoric_network import AleatoricNetwork
from models.erfnet.erfnet import ERFNetModel
from models.erfnet.aleatoric_erfnet import AleatoricERFNetModel
from models.unet.unet import UNetModel
from models.unet.aleatoric_unet import AleatoricUNetModel
from models.loss import CrossEntropyLoss, NLLLoss, AleatoricLoss

# TODO: Probably, the concept of model should change to "pipeline" or something similar.
networks = {
    "deterministic": DeterministicNetwork,
    "aleatoric": AleatoricNetwork,
}

models = {
    "erfnet": ERFNetModel,
    "unet": UNetModel,
}

aleatoric_models = {
    "erfnet": AleatoricERFNetModel,
    "unet": AleatoricUNetModel,
}

losses = {
    "cross_entropy": CrossEntropyLoss,
    "nll": NLLLoss,
    "aleatoric": AleatoricLoss,
}

# TODO: We have to refactor the network/model naming.
#       - The "model" (ERFNet, UNet) should include the architecture (model)
#       and a few utilities like "get_predictions", "visualize?"
#      - The "network" (Deterministic, Aleatoric) should include the training loop
#       logs, loss...
def get_model(cfg) -> LightningModule:
    network_type = cfg["model"]["type"]

    if isinstance(cfg, dict):
        try:
            model_name = cfg["model"]["name"]
            if network_type == "deterministic":
                return networks[network_type](models[model_name](cfg), cfg)
            elif network_type == "aleatoric":
                return networks[network_type](aleatoric_models[model_name](cfg), cfg)
            else:
                raise RuntimeError(f"Model {model_name} not implemented")            
        except KeyError:
            raise RuntimeError(f"Model {model_name} not implemented")
    else:
        raise RuntimeError(f"{type(cfg)} not a valid config")


def get_loss_fn(cfg) -> torch.nn.Module:
        """
        Returns the loss function based on the config file.
        """
        loss_name = cfg["model"]["loss"]

        # If class frequencies are provided, use it to weight the losses.
        if "class_frequencies" in cfg["model"]:
            class_frequencies = torch.Tensor(cfg["model"]["class_frequencies"])
            inv_class_frequencies = class_frequencies.sum() / class_frequencies

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