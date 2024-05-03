# This scripts loads a model and saves only its weights.
# This avoids the need to have the same imports when loading the model.

import tyro
from os.path import join, dirname, abspath
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import yaml
import torch
from semantic_segmentation.datasets import get_data_module
from semantic_segmentation.models import get_model


def main(checkpoint: str):
    """
    Test a model

    Args:
        checkpoint (str): path to checkpoint file (.ckpt)
    """
    ckpt = torch.load(checkpoint)

    # Get the state dict
    sd = ckpt["state_dict"]

    # Change all the keys to remove the initial "model."
    new_sd = {}
    for k, v in sd.items():
        new_sd[k.replace("model.", "")] = v

    # Save only the weights
    torch.save(new_sd, checkpoint.replace(".ckpt", "_weights.ckpt"))

if __name__ == "__main__":
    tyro.cli(main)