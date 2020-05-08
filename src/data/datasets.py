import os

import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as albu
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import WeightedRandomSampler

from src.data.augmentations import get_aug
from src.utils import ToCudaLoader


def get_dataloaders(
    root='data/interim',
    augmentation='light', 
    fold=0,
    pos_weight=0.5,
    batch_size=4, 
    size=512,
    val_size=768,
    workers=6):
    """
    Args:
        root (str): Path to folder with data
        aumentation (str): Type of aug defined in `src.data.augmentations.py`
        fold (int): Number of KFold validation split. Default: 0
        pos_weight (float): Proportion of positive examples. Default: 0.5 - balanced sampling
        batch_size (int): Number of images in stack
        size (int): Crop size to take from original image
        size (int): Crop size to take from original image
        workers (int): Number of CPU threads used to load images
    Returns:
        train_dataloader, val_dataloader
    """

    # Get augmentations
    train_aug = get_aug(augmentation, size=size)
    val_aug = get_aug("val", size=val_size)

    # Get dataset
    train_dataset = PneumothoraxDataset(
        root=root,
        fold=fold,
        train=True,
        transform=train_aug, 
    )

    val_dataset = PneumothoraxDataset(
        root=root,
        fold=fold,
        train=False,
        transform=val_aug, 
    )

    # Fix class inbalance
    # Distribution of classes in the dataset 
    label_to_weight = {
        0: 1 - pos_weight,
        1: pos_weight
    }
    weights = [label_to_weight[k] for k in train_dataset.classes]

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True),
        num_workers=workers, 
        drop_last=True, 
        pin_memory=True)
        
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True)

    train_loader = ToCudaLoader(train_loader)
    val_loader = ToCudaLoader(val_loader)

    print(f"\nUsing fold: {fold}. Train size: {len(train_dataset)},", 
        f"Validation size: {len(val_dataset)}")
    return train_loader, val_loader


def get_test_dataloader(
    root="data/interim",
    batch_size=8, 
    size=768,
    workers=6,
    ):
    """
    Args:
        root (str): Path to folder with data
        batch_size (int): Number of images in stack
        size (int): Crop size to take from original image
        workers (int): Number of CPU threads used to load images
    Returns:
        test_loader, test_filenames
    """
    aug = get_aug("test", size=size)

    test_dataset = PneumothoraxTestDataset(root=root, transform=aug)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers)
    test_loader = ToCudaLoader(test_loader)

    print(f"\nTest size: {len(test_dataset)}")
    return test_loader, test_dataset.images


class PneumothoraxDataset(torch.utils.data.Dataset):
    "Dataset for SIIM-ACR Pneumothorax Segmentation challenge"
    def __init__(
        self, 
        root="data/interim", 
        train_val_csv_path="data/interim/train_val.csv",
        fold=0,
        train=True,
        transform=None):
        """
        Args:
            root (str): Path to folder with all training data
            train_val_csv_path (str): Path to file with indexes split
            fold (int): KFold validation split to use
            train (bool): Flag to split dataset into train and validation
            transform (albu.Compose): albumentation transformation for images
        """
        # Read DF, convert columns to right dtype and filter by `train` and `fold`
        df = pd.read_csv(train_val_csv_path)
        df = df.astype({'Index': str, 'Class': int, 'Fold': int, 'Train': int})
        df = df.astype({'Train': bool})
        df = df[(df["Train"] == train) & (df["Fold"] == fold)]

        self.images = list(root + "/images/" + df["Index"] + ".png")
        self.masks = list(root + "/masks/" + df["Index"] + ".png")

        # 0 - No pneumothorax, 1 - Pneumothorax
        self.classes = list(df["Class"])

        self.transform = albu.Compose([]) if transform is None else transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, image_mask, target) where target is the image segmentation.
        """
        image_path = self.images[index]
        mask_path = self.masks[index]
        image = cv2.imread(image_path) # , cv2.IMREAD_GRAYSCALE
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Greyscale -> PseudoRGB
        # image = np.stack([image, image, image], axis=2)
        # np.repeat()

        # Apply Albumentation transform
        trfm = self.transform(image=image, mask=mask)
        image, mask = trfm["image"], trfm["mask"]

        return image, mask.unsqueeze(0) / 255.0


class PneumothoraxTestDataset(torch.utils.data.Dataset):
    "Test dataset for SIIM-ACR Pneumothorax Segmentation challenge"
    def __init__(self, root="data/interim", transform=None):
        """
        Args:
            root (str): Path to folder with all training data
            transform (albu.Compose): albumentation transformation for images
        """
        self.root = root
        self.images = sorted(os.listdir(root + '/test_images'))  # Order files
        self.transform = albu.Compose([]) if transform is None else transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, filename)
        """
        image_path = self.root + "/test_images/" + self.images[index]
        image = cv2.imread(image_path) #, cv2.IMREAD_GRAYSCALE
        # Greyscale -> PseudoRGB
        # image = np.stack([image, image, image], axis=2)
        image = self.transform(image=image)["image"]
        return image


# class ImbalancedBinarySampler(torch.utils.data.sampler.Sampler):
#     """Samples elements randomly from a given list of indices for imbalanced dataset
#     Args:
#         dataset: Dataset with unbalanced binary classes
#         indices (list, optional): a list of indices
#         # num_samples (int, optional): Number of samples to draw. Default: len(dataset)
#         pos_weight (float): Proportion of positive examples. Default: 0.5 - balanced sampling
#     """
#     def __init__(self, dataset, pos_weight=0.5):

#         self.indices = list(range(len(dataset)))
#         self.num_samples = len(dataset)
            
#         # distribution of classes in the dataset 
#         label_to_weight = {
#             0: 1 - pos_weight,
#             1: pos_weight
#         }

#         # weight for each sample
#         weights = [label_to_weight[dataset.classes[idx]] for idx in self.indices]
#         self.weights = torch.DoubleTensor(weights)

#     def __iter__(self):
#         return (self.indices[i] for i in torch.multinomial(
#             self.weights, self.num_samples, replacement=True))

#     def __len__(self):
#         return self.num_samples