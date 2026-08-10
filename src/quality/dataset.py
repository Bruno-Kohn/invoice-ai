"""PyTorch Dataset for document quality classification."""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_TO_IDX = {"ready": 0, "marginal": 1, "not_ready": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


class QualityDataset(Dataset):
    """Dataset that loads synthetic images with CER-based quality labels.

    Args:
        labels_path: Path to labels.json.
        synthetic_dir: Root directory of synthetic images.
        transform: Optional torchvision transform to apply.
        split: One of "train", "val", "test". Uses stratified split.
        split_seed: Random seed for reproducible splits.
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
    """

    def __init__(
        self,
        labels_path: Path = Path("data/synthetic/labels.json"),
        synthetic_dir: Path = Path("data/synthetic"),
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
        split_seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        self.synthetic_dir = Path(synthetic_dir)
        self.transform = transform

        with open(labels_path) as f:
            all_labels = json.load(f)

        # Filter out entries where image doesn't exist
        valid_labels = [
            entry for entry in all_labels
            if (self.synthetic_dir / entry["image_path"]).exists()
        ]

        # Stratified split
        self.samples = self._split(valid_labels, split, split_seed, train_ratio, val_ratio)

    def _split(self, labels, split, seed, train_ratio, val_ratio):
        """Stratified split by label."""
        rng = np.random.default_rng(seed)

        by_label = {"ready": [], "marginal": [], "not_ready": []}
        for entry in labels:
            by_label[entry["label"]].append(entry)

        train, val, test = [], [], []
        for label_entries in by_label.values():
            indices = rng.permutation(len(label_entries))
            n_train = int(len(label_entries) * train_ratio)
            n_val = int(len(label_entries) * val_ratio)

            for i, idx in enumerate(indices):
                if i < n_train:
                    train.append(label_entries[idx])
                elif i < n_train + n_val:
                    val.append(label_entries[idx])
                else:
                    test.append(label_entries[idx])

        if split == "train":
            return train
        elif split == "val":
            return val
        else:
            return test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        entry = self.samples[idx]
        img_path = self.synthetic_dir / entry["image_path"]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            img = default_transform()(img)

        label = LABEL_TO_IDX[entry["label"]]
        return img, label


def default_transform(image_size: int = 224):
    """Default transform for quality CNN (MobileNetV2 input)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def train_transform(image_size: int = 224):
    """Training transform with augmentation."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
