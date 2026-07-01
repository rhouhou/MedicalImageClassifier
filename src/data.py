from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(image_size: int, augment: bool = True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int, image_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tfms = build_transforms(image_size, augment=True)
    eval_tfms = build_transforms(image_size, augment=False)
    root = Path(data_dir)
    train_ds = datasets.ImageFolder(root / 'train', transform=train_tfms)
    val_ds = datasets.ImageFolder(root / 'val', transform=eval_tfms)
    test_ds = datasets.ImageFolder(root / 'test', transform=eval_tfms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader