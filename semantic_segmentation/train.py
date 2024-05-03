from os.path import abspath, dirname, join
from typing import Optional
import tyro
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from semantic_segmentation.datasets import get_data_module
from semantic_segmentation.models import get_model
import warnings

warnings.filterwarnings("ignore")


def main(
    config: str = join(dirname(abspath(__file__))),
    weights: Optional[str] = None,  # TODO: add possibility to load weights
    checkpoint: Optional[str] = None,
):
    """
    Train a model

    Args:
        config (str): path to config file (.yaml)
        weights (str): path to pretrained weights (.ckpt)
        checkpoint (str): path to checkpoint file (.ckpt) to resume training.
    """
    with open(config, "r") as config_file:
        cfg = yaml.safe_load(config_file)

    # Load data and model
    data = get_data_module(cfg)
    model = get_model(cfg)
    print(model)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        monitor="Validation/mIoU",
        filename=cfg["experiment"]["id"] + "_{epoch:02d}_{iou:.2f}",
        mode="max",
        save_last=True,
        save_weights_only=True, # This is required to looad models without having the same imports
    )

    tb_logger = pl_loggers.TensorBoardLogger(f"experiments/{cfg['experiment']['id']}", default_hp_metric=False)

    # Setup trainer
    trainer = Trainer(
        devices=cfg["train"]["n_gpus"],
        logger=tb_logger,
        max_epochs=cfg["train"]["max_epoch"],
        callbacks=[lr_monitor, checkpoint_saver],
    )

    # Train!
    trainer.fit(model, data, ckpt_path=checkpoint)


if __name__ == "__main__":
    tyro.cli(main)
