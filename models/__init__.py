import torch
from models.network import Network, AleatoricNetwork
from pytorch_lightning.core.lightning import LightningModule


def get_model(cfg) -> LightningModule:
    network_type = cfg["model"]["type"]
    network_name = cfg["model"]["name"]

    if isinstance(cfg, dict):
        if network_type == "network":
            return Network(network_name, cfg)
        elif network_type == "aleatoric_network":
            return AleatoricNetwork(network_name, cfg)
        else:
            raise RuntimeError(f"{network_type} not implemented")
    else:
        raise RuntimeError(f"{type(cfg)} not a valid config")


def load_pretrained_model(checkpoint_path: str) -> LightningModule:
    cfg = torch.load(checkpoint_path)["hyper_parameters"]["cfg"]
    return get_model(cfg).load_from_checkpoint(checkpoint_path).eval()
