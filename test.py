import tyro
from os.path import join, dirname, abspath
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import yaml
import torch
from datasets import get_data_module
from models import get_model


def main(checkpoint: str):
    """
    Test a model

    Args:
        checkpoint (str): path to checkpoint file (.ckpt)
    """
    cfg = torch.load(checkpoint)["hyper_parameters"]["cfg"]

    # Load data and model
    data = get_data_module(cfg)
    model = get_model(cfg)

    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg["experiment"]["id"], default_hp_metric=False
    )

    # Setup trainer
    trainer = Trainer(logger=tb_logger, devices=cfg["train"]["n_gpus"])

    # Test!
    trainer.test(model, data, ckpt_path=checkpoint)


if __name__ == "__main__":
    tyro.cli(main)
