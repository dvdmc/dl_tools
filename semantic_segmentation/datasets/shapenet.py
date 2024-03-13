import os
import cv2
import numpy as np
import torch
from PIL import Image
from utils.transformations import get_transformations
from utils.utils import LABELS
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob


class ShapenetDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage):
        path_to_dataset = os.path.join(self.cfg["data"]["path_to_dataset"])

        # Assign datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self._train = ShapenetDataset(
                path_to_dataset,
                "train",
                transformations=get_transformations(self.cfg, "train"),
            )
            self._val = ShapenetDataset(
                path_to_dataset,
                "val",
                transformations=get_transformations(self.cfg, "val"),
            )

        if stage == "test":
            self._test = ShapenetDataset(
                path_to_dataset,
                "test",
                transformations=get_transformations(self.cfg, "test"),
            )

    def train_dataloader(self):
        shuffle = self.cfg["data"]["train_shuffle"]
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(
            self._train, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers
        )

        return loader

    def val_dataloader(self):
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(
            self._val, batch_size=batch_size, num_workers=n_workers, shuffle=True
        )

        return loader

    def test_dataloader(self):
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(
            self._test, batch_size=batch_size, num_workers=n_workers, shuffle=False
        )

        return loader


class ShapenetDataset(Dataset):
    def __init__(self, data_rootdir, mode, transformations):
        super().__init__()

        assert os.path.exists(data_rootdir)
        split_file = os.path.join(data_rootdir, f"{mode}.lst")
        assert os.path.exists(split_file)
        with open(split_file, "r") as f:
            scene_id_list = [x.strip() for x in f.readlines()]

        scene_path = [
            os.path.join(data_rootdir, "scene" + scene_id) for scene_id in scene_id_list
        ]

        # Check if scenes exist. Print warning if not.
        for scene in scene_path:
            if not os.path.exists(scene):
                print(f"Warning: {scene} does not exist!")

        self.image_files = []
        self.label_files = []
        for scene in scene_path:
            image_path = os.path.join(scene, "images")
            label_path = os.path.join(scene, "semantics")
            scene_images = [
                x
                for x in glob.glob(os.path.join(image_path, "*"))
                if (x.endswith(".jpg") or x.endswith(".png"))
            ]
            scene_images.sort()
            scene_labels = [
                x
                for x in glob.glob(os.path.join(label_path, "*"))
                if (x.endswith(".jpg") or x.endswith(".png"))
            ]
            scene_images.sort()
            scene_labels.sort()
            self.image_files += scene_images
            self.label_files += scene_labels
        print(f"Found {len(self.image_files)} images for {mode} mode")
        self.img_to_tensor = transforms.ToTensor()
        self.transformations = transformations

    def __getitem__(self, idx):
        path_to_current_img = self.image_files[idx]
        img_pil = Image.open(path_to_current_img)
        img = self.img_to_tensor(img_pil)

        path_to_current_label = self.label_files[idx]
        label = self.get_label(path_to_current_label)

        # apply a set of transformations to the raw_image, image and label
        for transformer in self.transformations:
            img_pil, img, label = transformer(img_pil, img, label)

        label = self.remap_label(label.numpy())

        return {"data": img, "image": img, "label": label, "index": idx}

    def __len__(self):
        return len(self.image_files)

    def get_label(self, path_to_current_label):
        label = cv2.imread(path_to_current_label)
        label = label.astype(np.int64)  # torch does not support conversion of uint16
        label = np.moveaxis(label, -1, 0)  # now in CHW mode
        return torch.from_numpy(label).long()

    @staticmethod
    def remap_label(label):
        dims = label.shape
        assert len(dims) == 3, "wrong matrix dimension!!!"
        assert dims[0] == 3, "label must have 3 channels!!!"

        shapenet_labels = LABELS["shapenet"]
        remapped_label = (
            np.ones((dims[1], dims[2])) * shapenet_labels["background"]["id"]
        )

        for label_key, label_info in shapenet_labels.items():
            if label_key == "background":
                continue

            label_color = np.flip(np.array(label_info["color"])).reshape((3, 1, 1))
            remapped_label[(label == label_color).all(axis=0)] = label_info["id"]

        return torch.from_numpy(remapped_label).long()
