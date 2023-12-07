from datasets.shapenet import ShapenetDataModule


def get_data_module(cfg):
    name = cfg["data"]["name"]
    if name == "shapenet":
        return ShapenetDataModule(cfg)
    else:
        raise ValueError(f"Dataset '{name}' not found!")
