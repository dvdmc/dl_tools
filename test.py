import click
from os.path import join, dirname, abspath
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import yaml
import torch
from datasets import get_data_module
from models import load_pretrained_model


@click.command()
### Add your options here
@click.option(
    "--checkpoint",
    "-ckpt",
    type=str,
    help="path to checkpoint file (.ckpt)",
    required=True,
)
def main(checkpoint):
    cfg = torch.load(checkpoint)["hyper_parameters"]["cfg"]

    # Load data and model
    data = get_data_module(cfg)
    model = load_pretrained_model(checkpoint)

    tb_logger = pl_loggers.TensorBoardLogger(
        "experiments/" + cfg["experiment"]["id"], default_hp_metric=False
    )

    # Setup trainer
    trainer = Trainer(logger=tb_logger, gpus=cfg["train"]["n_gpus"])

    # Test!
    trainer.test(model, data)


if __name__ == "__main__":
    main()
