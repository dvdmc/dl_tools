import os
import numpy as np
import torch
from PIL import Image
from utils.transformations import get_transformations
from utils.utils import LABELS
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class PascalVOCDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage):
        path_to_dataset = os.path.join(self.cfg["data"]["path_to_dataset"])
        
        # Assign datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if "aug" in self.cfg["data"]:
                aug = self.cfg["data"]["aug"]
            else:
                aug = False

            train_split = "train"
            val_split = "val"
            test_split = "test"
            if aug:
                train_split = "train_aug"
                val_split = "train_aug_val"

            self._train = PascalVOCDataset(
                path_to_dataset,
                "train",
                split=train_split,
                transformations=get_transformations(self.cfg, "train"),
            )
            self._val = PascalVOCDataset(
                path_to_dataset,
                "val",
                split=val_split,
                transformations=get_transformations(self.cfg, "val"),
            )

        if stage == "test":
            self._test = PascalVOCDataset(
                path_to_dataset,
                "test",
                split=test_split,
                transformations=get_transformations(self.cfg, "test"),
            )

    def train_dataloader(self):
        shuffle = self.cfg["data"]["train_shuffle"]
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(self._train, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)

        return loader

    def val_dataloader(self):
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(self._val, batch_size=batch_size, num_workers=n_workers, shuffle=True)

        return loader

    def test_dataloader(self):
        batch_size = self.cfg["data"]["batch_size"]
        n_workers = self.cfg["data"]["num_workers"]

        loader = DataLoader(self._test, batch_size=batch_size, num_workers=n_workers, shuffle=False)

        return loader

class PascalVOCDataset(Dataset):
    """Data loader for the Pascal VOC semantic segmentation dataset.

    Source: https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/voc12/README.md
    
    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        trainval: The combination of `train` and `val` - 2913 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_aug_val: The original VOC 2012 validation data minus the images
                   present in `train_aug` (This is done with the same logic as
                   the validation set used in FCN PAMI paper, but with VOC 2012
                   rather than VOC 2011) - 904 images
    """

    def __init__(
        self,
        data_rootdir,
        mode,
        split="train_aug",
        transformations=None,
        img_norm=True,
    ):
        self.data_rootdir = data_rootdir
        self.mode = mode
        self.split = split
        self.transformations = transformations
        self.img_norm = img_norm
        self.n_classes = 21

        assert os.path.exists(self.data_rootdir), f"Error: {data_rootdir} is not found"
        split_file = os.path.join(self.data_rootdir, "ImageSets/SegmentationAug", split + ".txt")
        assert os.path.exists(split_file), "Error: split is not found"
        with open(split_file, "r") as f:
            items_id_list = [x.split(" ") for x in f.readlines()]
        
        self.image_files = []
        self.label_files = []
        for item_id in items_id_list:
            image_path = self.data_rootdir + item_id[0]
            label_path = self.data_rootdir + item_id[1].rstrip()
            self.image_files.append(image_path)
            self.label_files.append(label_path)
        if img_norm:
            self.norm_img_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.img_to_tensor_transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img = Image.open(self.image_files[index])
        label = Image.open(self.label_files[index])

        img_tensor = self.img_to_tensor_transform(img)

        if self.img_norm:
            img_tensor = self.norm_img_transform(img_tensor)

        if self.transformations is not None:
            for transforation in self.transformations:
                img, label = transforation(img_tensor, label, img)
            img, label = self.transformations(img, label)


        label = np.array(label)
        label[label == 255] = 0
        label = torch.from_numpy(label).long()

        return {"data": img, "image": img, "label": label, "index": index}