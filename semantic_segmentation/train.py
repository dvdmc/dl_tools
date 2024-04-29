import torch
# Set matrix multiplication precision to 'medium' for better performance
torch.set_float32_matmul_precision('medium')

from os.path import abspath, dirname, join
from typing import Optional
import tyro
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import sys
sys.path.append('/home/ego_exo4d/Documents/dl_tools_loren')
from semantic_segmentation.datasets import get_data_module
from semantic_segmentation.models import get_model
from utils.utils import paint_labels_and_image
import warnings
import os

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
    data.setup(stage = 'fit')
    

    # Load data and model
    #data = get_data_module(cfg)
    #data.setup(stage = 'fit')
    #train_loader = data.train_dataloader()
    
    model = get_model(cfg)
    print(model)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver = ModelCheckpoint(
        monitor="Validation/mIoU",
        filename=cfg["experiment"]["id"] + "_{epoch:02d}_{iou:.2f}",
        mode="max",
        save_last=True,
    )

    log_dir = os.path.join('/home/ego_exo4d/Documents/dl_tools_loren/experiments', cfg["experiment"]["id"])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_logger = pl_loggers.TensorBoardLogger(log_dir)

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
