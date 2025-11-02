import os
import pandas as pd
from typing import Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .augmentations import get_transforms


class TrashBinaryDataset(Dataset):
    def __init__(self, csv_path: str, split: str, image_size: int = 224, aug_strength: str = "medium"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.split = split
        train_tfms, val_tfms = get_transforms(image_size=image_size, strength=aug_strength)
        self.tfms = train_tfms if split == "train" else val_tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        x = self.tfms(img)
        y = torch.tensor(row["label"], dtype=torch.float32)
        return x, y


def get_loaders(csv_path: str, image_size: int, batch_size: int, num_workers: int, aug_strength: str) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    train_ds = TrashBinaryDataset(csv_path, split="train", image_size=image_size, aug_strength=aug_strength)
    val_ds = TrashBinaryDataset(csv_path, split="val", image_size=image_size, aug_strength=aug_strength)
    test_ds = None
    if (pd.read_csv(csv_path)["split"] == "test").any():
        test_ds = TrashBinaryDataset(csv_path, split="test", image_size=image_size, aug_strength="none")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers) if test_ds else None
    return train_loader, val_loader, test_loader
