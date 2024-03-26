"""
    This file works as a module factory.
    It returns the correct modules based on the config file.
    In our implementation, the different modules are:
    - DataModule: The dataset
    TODO: Maybe here can be the bridges
"""

def get_data_module(cfg):
    name = cfg["data"]["name"]
    if name == "shapenet":
        from datasets.shapenet import ShapenetDataModule
        
        return ShapenetDataModule(cfg)
    elif name == "pascal_voc":
        from datasets.voc_pascal import PascalVOCDataModule

        return PascalVOCDataModule(cfg)
    else:
        raise ValueError(f"Dataset '{name}' not found!")
