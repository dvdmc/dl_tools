import tyro
from os.path import join, dirname, abspath
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import yaml
import torch
import sys
sys.path.append('/home/ego_exo4d/Documents/dl_tools_loren')
from semantic_segmentation.datasets import get_data_module
from semantic_segmentation.models import get_model
import os

import wandb
from pytorch_lightning.loggers import WandbLogger

def main(checkpoint: str, exp: str):
    """
    Test a model

    Args:
        checkpoint (str): path to checkpoint file (.ckpt)
    """
    #cfg = torch.load(checkpoint)["hyper_parameters"]["cfg"]
    config = '/home/ego_exo4d/Documents/dl_tools_loren/config/voc12_MCD_inference.yaml'
    with open(config, "r") as config_file:
        cfg = yaml.safe_load(config_file)
    
    #model_weights = torch.load(checkpoint)['model_state']
    #new_model_weights = {}
    #for key in model_weights.keys():
    #    if key.startswith('backbone.'):
    #        new_key = key.replace('backbone.', 'model.backbone.')
    #    elif key.startswith('classifier.classifier.'):
    #        new_key = key.replace('classifier.classifier.', 'model.classifier.classifier.')
    #    new_model_weights[new_key] = model_weights[key]

    
    # Load data and model
    data = get_data_module(cfg)
    model = get_model(cfg)
    #model.load_state_dict(new_model_weights)

    #tb_logger = pl_loggers.TensorBoardLogger("experiments/" + cfg["experiment"]["id"], default_hp_metric=False)
    output_dir = '/home/ego_exo4d/Documents/dl_tools_loren/wandb_logs'
    wandb_dir = os.path.join(output_dir, exp)
    if not(os.path.exists(wandb_dir)):
        os.makedirs(wandb_dir)
    n_subexperiments = os.listdir(wandb_dir)
    subdirs = [item for item in os.listdir(wandb_dir) if os.path.isdir(os.path.join(wandb_dir, item))]
    name_exp = exp + '_' + str(len(subdirs) + 1)
    wb_logger = WandbLogger(project="SEM_mapping_test", entity="affordances", name=name_exp, dir=wandb_dir)

    # Setup trainer
    trainer = Trainer(logger=wb_logger, devices=cfg["train"]["n_gpus"])

    # Test!
    trainer.test(model, data, ckpt_path=checkpoint)


if __name__ == "__main__":
    tyro.cli(main)
