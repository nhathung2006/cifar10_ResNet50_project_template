import random
from zipfile import BadZipFile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE, CLASS_NAMES, EVAL_RESIZE_SIZE, IMAGE_MEAN, IMAGE_SIZE,
    IMAGE_STD, NUM_WORKERS, SEED, SPLIT_PATH, TEST_DATA_DIR, TRAIN_DATA_DIR,
    validate_data_dir,
)


def _validate_class_mapping(dataset: datasets.ImageFolder, split_name: str) -> None:
    expected = list(CLASS_NAMES)
    if dataset.classes != expected:
        raise ValueError(f"Class mapping của {split_name} không đúng: {dataset.classes}; mong đợi {expected}.")


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.80, 1.0), ratio=(0.80, 1.25)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.12)),
    ])


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(EVAL_RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])


def get_prediction_transform() -> transforms.Compose:
    return get_eval_transform()


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _read_split(dataset: datasets.ImageFolder) -> tuple[list[int], list[int]]:
    split_path = SPLIT_PATH
    if not split_path.is_file():
        raise FileNotFoundError("Không tìm thấy fixed split. Hãy chạy python create_fixed_split.py trước.")
    try:
        with np.load(split_path, allow_pickle=False) as split:
            train = np.asarray(split["train_indices"])
            val = np.asarray(split["val_indices"])
    except (BadZipFile, KeyError, OSError, ValueError) as error:
        raise ValueError(f"File split không hợp lệ: {split_path}") from error

    if train.ndim != 1 or val.ndim != 1 or not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(val.dtype, np.integer):
        raise ValueError("train_indices và val_indices phải là mảng số nguyên 1 chiều.")
    train, val = train.astype(np.int64), val.astype(np.int64)
    combined = np.concatenate((train, val))
    expected = np.arange(len(dataset), dtype=np.int64)
    if len(np.unique(train)) != len(train) or len(np.unique(val)) != len(val):
        raise ValueError("Split chứa index bị trùng trong cùng một tập.")
    if np.intersect1d(train, val).size:
        raise ValueError("Train và validation có index trùng nhau.")
    if not np.array_equal(np.sort(combined), expected):
        raise ValueError("Split bị thiếu index hoặc có index ngoài phạm vi.")
    targets = np.asarray(dataset.targets)
    for name, indices in (("train", train), ("validation", val)):
        counts = np.bincount(targets[indices], minlength=len(CLASS_NAMES))
        expected_ratio = 1.0 - 0.1 if name == "train" else 0.1
        expected_counts = np.bincount(targets, minlength=len(CLASS_NAMES)) * expected_ratio
        if not np.allclose(counts, expected_counts, atol=1):
            raise ValueError(f"Split {name} không stratified đúng: {counts.tolist()}")
    return train.tolist(), val.tolist()


def _loader_options(device: torch.device) -> dict:
    return {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS, "pin_memory": device.type == "cuda", "persistent_workers": NUM_WORKERS > 0, "worker_init_fn": seed_worker}


def create_dataloaders(device: torch.device) -> tuple[DataLoader, DataLoader, DataLoader]:
    validate_data_dir()
    split_dataset = datasets.ImageFolder(str(TRAIN_DATA_DIR), transform=None)
    test_dataset = datasets.ImageFolder(str(TEST_DATA_DIR), transform=get_eval_transform())
    _validate_class_mapping(split_dataset, "training")
    _validate_class_mapping(test_dataset, "test")
    train_indices, val_indices = _read_split(split_dataset)
    train_base = datasets.ImageFolder(str(TRAIN_DATA_DIR), transform=get_train_transform())
    val_base = datasets.ImageFolder(str(TRAIN_DATA_DIR), transform=get_eval_transform())
    options = _loader_options(device)
    generator = torch.Generator().manual_seed(SEED)
    print(f"Thứ tự class      : {split_dataset.classes}")
    print(f"Số ảnh train      : {len(train_indices)}")
    print(f"Số ảnh validation : {len(val_indices)}")
    print(f"Số ảnh test       : {len(test_dataset)}")
    return (
        DataLoader(Subset(train_base, train_indices), shuffle=True, generator=generator, **options),
        DataLoader(Subset(val_base, val_indices), shuffle=False, **options),
        DataLoader(test_dataset, shuffle=False, **options),
    )


def create_test_loader(device: torch.device) -> DataLoader:
    validate_data_dir()
    dataset = datasets.ImageFolder(str(TEST_DATA_DIR), transform=get_eval_transform())
    _validate_class_mapping(dataset, "test")
    return DataLoader(dataset, shuffle=False, **_loader_options(device))
