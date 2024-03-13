"""
    This file works as a module factory.
    It returns the correct modules based on the config file.
    In our implementation, the different modules are:
    - DataModule: The dataset
    TODO: Maybe here can be the bridges
"""

from datasets.shapenet import ShapenetDataModule

def get_data_module(cfg):
    name = cfg["data"]["name"]
    if name == "shapenet":
        return ShapenetDataModule(cfg)
    else:
        raise ValueError(f"Dataset '{name}' not found!")
