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
import yaml
import pandas as pd


class VOC12DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage):
        path_to_dataset = os.path.join(self.cfg["data"]["path_to_dataset"])

        # Assign datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self._train = VOC12Dataset(
                path_to_dataset,
                "train",
                transformations=get_transformations(self.cfg, "train"),
            )
            self._val = VOC12Dataset(
                path_to_dataset,
                "val",
                transformations=get_transformations(self.cfg, "val"),
            )

        if stage == "test":
            self._test = VOC12Dataset(
                path_to_dataset,
                "test",
                transformations=get_transformations(self.cfg, "test"),
            )

    def train_dataloader(self):
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(self._train, batch_size=batch_size, num_workers=n_workers, shuffle=True)

        return loader

    def val_dataloader(self):
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(self._val, batch_size=batch_size, num_workers=n_workers, shuffle=False)

        return loader

    def test_dataloader(self):
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(self._test, batch_size=batch_size, num_workers=n_workers, shuffle=False)

        return loader


class VOC12Dataset(Dataset):
    def __init__(self, data_rootdir, mode, transformations):
        super().__init__()

        assert os.path.exists(data_rootdir)
        if mode == "train":
            split_file = '/home/ego_exo4d/VOCdevkit/VOC2012/ImageSets/Segmentation/train_aug.txt'
        elif mode == "val":
            split_file = '/home/ego_exo4d/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
        assert os.path.exists(split_file)
        with open(split_file, "r") as f:
            image_id_list = [x.strip() for x in f.readlines()]

        self.image_files = [os.path.join(data_rootdir, "JPEGImages", x + ".jpg") for x in image_id_list]
        self.label_files = [os.path.join(data_rootdir, "SegmentationClassAug", x + ".png") for x in image_id_list]
        print(f"Found {len(self.image_files)} images for {mode} mode")
        print(f"Found {len(self.label_files)} labels for {mode} mode")
        print(self.image_files[0], self.label_files[0])
    
        assert len(self.image_files) == len(self.label_files)
        self.img_to_tensor = transforms.ToTensor()
        self.transformations = transformations
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_label(self, label_path):
        label_map = np.array(Image.open(label_path))
        label_map = torch.Tensor(label_map).unsqueeze(0)
        return label_map

    def __getitem__(self, idx):
        path_to_current_img = self.image_files[idx]
        img_pil = Image.open(path_to_current_img)
        img = self.img_to_tensor(img_pil) #(3, H, W) Tensor
        img = self.normalize_img(img)
        
        path_to_current_label = self.label_files[idx]
        label = self.get_label(path_to_current_label) #(1, H, W) Tensor
        label[label == 255] = 0
        
        # apply a set of transformations to the raw_image, image and label
        for transformer in self.transformations:
            img, label, img_pil = transformer(img, label, img_pil)

        return {"data": img, "label": label.squeeze().long(), "index": idx}

    def __len__(self):
        return len(self.image_files)
