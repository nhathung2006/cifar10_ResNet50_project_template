import random
from zipfile import BadZipFile
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    CIFAR10_MEAN,
    CIFAR10_STD,
    DATA_DIR,
    NUM_WORKERS,
    SEED,
    SPLIT_PATH,
)


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_prediction_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _read_split(dataset: datasets.CIFAR10) -> tuple[list[int], list[int]]:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy fixed split: {SPLIT_PATH}. "
            "Hãy chạy python create_fixed_split.py trước."
        )

    try:
        with np.load(SPLIT_PATH, allow_pickle=False) as split:
            train_indices = np.asarray(split["train_indices"])
            val_indices = np.asarray(split["val_indices"])
    except (BadZipFile, KeyError, OSError, ValueError) as error:
        raise ValueError(f"File split không hợp lệ: {SPLIT_PATH}") from error

    if train_indices.ndim != 1 or val_indices.ndim != 1:
        raise ValueError("train_indices và val_indices phải là mảng 1 chiều.")
    if not np.issubdtype(train_indices.dtype, np.integer) or not np.issubdtype(
        val_indices.dtype, np.integer
    ):
        raise ValueError("Các index trong split phải là số nguyên.")

    train_indices = train_indices.astype(np.int64, copy=False)
    val_indices = val_indices.astype(np.int64, copy=False)
    all_indices = np.concatenate((train_indices, val_indices))
    expected_indices = np.arange(len(dataset), dtype=np.int64)

    if len(train_indices) != 45_000 or len(val_indices) != 5_000:
        raise ValueError(
            "Split phải có 45.000 train và 5.000 validation, "
            f"nhưng nhận được {len(train_indices)} và {len(val_indices)}."
        )
    if len(np.unique(train_indices)) != len(train_indices) or len(
        np.unique(val_indices)
    ) != len(val_indices):
        raise ValueError("Split chứa index bị trùng trong cùng một tập.")
    if len(np.intersect1d(train_indices, val_indices)) != 0:
        raise ValueError("Train và validation có index trùng nhau.")
    if not np.array_equal(np.sort(all_indices), expected_indices):
        raise ValueError("Split bị thiếu index hoặc có index ngoài phạm vi.")

    for name, indices in (("train", train_indices), ("validation", val_indices)):
        counts = np.bincount(
            np.asarray(dataset.targets)[indices], minlength=10
        )
        expected_count = 4_500 if name == "train" else 500
        if not np.all(counts == expected_count):
            raise ValueError(
                f"Split {name} không stratified đúng: {counts.tolist()}"
            )

    return train_indices.tolist(), val_indices.tolist()


def _loader_options(device: torch.device) -> dict:
    return {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
        "worker_init_fn": seed_worker,
    }


def create_dataloaders(
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    split_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        transform=None,
        download=False,
    )
    train_indices, val_indices = _read_split(split_dataset)

    full_train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        transform=get_train_transform(),
        download=False,
    )
    full_val_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        transform=get_eval_transform(),
        download=False,
    )
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        transform=get_eval_transform(),
        download=False,
    )
    if len(test_dataset) != 10_000:
        raise ValueError(f"CIFAR-10 test phải có 10.000 ảnh, nhận được {len(test_dataset)}.")

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)
    loader_generator = torch.Generator().manual_seed(SEED)
    options = _loader_options(device)

    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=loader_generator, **options
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **options)
    test_loader = DataLoader(test_dataset, shuffle=False, **options)
    return train_loader, val_loader, test_loader


def create_test_loader(device: torch.device) -> DataLoader:
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        transform=get_eval_transform(),
        download=False,
    )
    if len(test_dataset) != 10_000:
        raise ValueError(f"CIFAR-10 test phải có 10.000 ảnh, nhận được {len(test_dataset)}.")

    return DataLoader(
        test_dataset,
        shuffle=False,
        **_loader_options(device),
    )
