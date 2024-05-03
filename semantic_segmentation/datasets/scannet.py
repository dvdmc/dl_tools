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


class ScanNetDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage):
        path_to_dataset = os.path.join(self.cfg["data"]["path_to_dataset"])

        # Assign datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self._train = ScanNetDataset(
                path_to_dataset,
                "train",
                transformations=get_transformations(self.cfg, "train"),
            )
            self._val = ScanNetDataset(
                path_to_dataset,
                "val",
                transformations=get_transformations(self.cfg, "val"),
            )

        if stage == "test":
            self._test = ScanNetDataset(
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


class ScanNetDataset(Dataset):
    def __init__(self, data_rootdir, mode, transformations):
        super().__init__()

        assert os.path.exists(data_rootdir), "Error: data_rootdir is not found"
        split_file = os.path.join(data_rootdir, 'splits', f"{mode}.txt")
        assert os.path.exists(split_file), f"Error: {split_file} is not found"

        with open(split_file, "r") as f:
            scene_id_list = [x.strip() for x in f.readlines()]

        scene_path = [os.path.join(data_rootdir, "data", "scans", scene_id) for scene_id in scene_id_list if scene_id.endswith("_00")]

        # Check if scenes exist. Print warning if not.
        for scene in scene_path:
            if not os.path.exists(scene):
                print(f"Warning: {scene} does not exist!")

        self.image_files = []
        self.label_files = []
        for scene in scene_path:
            image_path = os.path.join(scene, "color")
            label_path = os.path.join(scene, "semantic_labels_filt")
            
            #Images
            scene_images = [
                x for x in glob.glob(os.path.join(image_path, "*")) if (x.endswith(".jpg") or x.endswith(".png"))
            ]
            scene_images = [image for image in scene_images
                            if int(image.split('/')[-1].split('.')[0]) % 20 == 0] #We reduce the dataset size and avoid redundant images
            scene_images.sort()
            
            #Sementic label maps
            scene_labels = [
                x for x in glob.glob(os.path.join(label_path, "*")) if (x.endswith(".jpg") or x.endswith(".png"))
            ]
            scene_labels = [label for label in scene_labels 
                            if int(label.split('/')[-1].split('.')[0]) % 20 == 0]
            scene_labels.sort()
            
            self.image_files += scene_images
            self.label_files += scene_labels
        print(f"Found {len(self.image_files)} images for {mode} mode")
        print(f"Found {len(self.label_files)} images for {mode} mode")
        assert len(self.image_files) == len(self.label_files)
        self.img_to_tensor = transforms.ToTensor()
        self.transformations = transformations
        self.normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.mapping_from_ScanNet_to_NYU40 = self.label_indexer()

        
    def label_indexer(self):
        tsv_file = '/home/ego_exo4d/Documents/dl_tools_loren/semantic_segmentation/datasets/scannetv2-labels.combined.tsv'
        label_mapping = pd.read_csv(tsv_file, sep='\t')
        NYU_classes = label_mapping['eigen13id'].values
        id_classes = label_mapping['id'].values
        #Make a dict with the mapping between ScanNet classes and NYUv2 classes
        return {id_classes[i]: NYU_classes[i].astype(int) for i in range(len(NYU_classes))}
        
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

        # apply a set of transformations to the raw_image, image and label
        for transformer in self.transformations:
            img, label, img_pil = transformer(img, label, img_pil)
        label = self.remap_label(label).squeeze(0)
        return {"data": img, "label": label, "index": idx}

    def __len__(self):
        return len(self.image_files)

    def remap_label(self, ScanNet_label):
        label_40 = np.zeros_like(ScanNet_label)
        for key, value in self.mapping_from_ScanNet_to_NYU40.items():
            label_40[np.where(ScanNet_label == key)] = value
        return torch.from_numpy(label_40).long()
    
