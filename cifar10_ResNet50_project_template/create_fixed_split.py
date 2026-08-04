import numpy as np
from torchvision.datasets import CIFAR10

from config import (
    DATA_DIR,
    LOCAL_SPLIT_PATH,
    SEED,
    VAL_RATIO,
    validate_data_dir,
)


def main() -> None:
    if LOCAL_SPLIT_PATH.exists():
        print(f"Split đã tồn tại, không ghi đè: {LOCAL_SPLIT_PATH}")
        return

    validate_data_dir()
    dataset = CIFAR10(root=DATA_DIR, train=True, download=False)
    if len(dataset) != 50_000:
        raise ValueError(f"CIFAR-10 train phải có 50.000 ảnh, nhận được {len(dataset)}.")

    targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(SEED)
    train_indices: list[int] = []
    val_indices: list[int] = []
    val_count = int(5_000 * VAL_RATIO)

    for class_id in range(10):
        class_indices = np.flatnonzero(targets == class_id)
        if len(class_indices) != 5_000:
            raise ValueError(
                f"Lớp {class_id} phải có 5.000 ảnh, nhận được {len(class_indices)}."
            )
        rng.shuffle(class_indices)
        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train_indices_array = np.asarray(train_indices, dtype=np.int64)
    val_indices_array = np.asarray(val_indices, dtype=np.int64)
    all_indices = np.concatenate((train_indices_array, val_indices_array))

    if len(train_indices_array) != 45_000 or len(val_indices_array) != 5_000:
        raise ValueError("Kích thước split không đúng.")
    if len(np.unique(all_indices)) != 50_000:
        raise ValueError("Split có index trùng hoặc thiếu index.")
    if not np.array_equal(np.sort(all_indices), np.arange(50_000)):
        raise ValueError("Split có index ngoài phạm vi hoặc thiếu index.")

    train_counts = np.bincount(targets[train_indices_array], minlength=10)
    val_counts = np.bincount(targets[val_indices_array], minlength=10)
    if not np.all(train_counts == 4_500) or not np.all(val_counts == 500):
        raise ValueError("Split không giữ đúng 4.500/500 ảnh cho mỗi lớp.")

    LOCAL_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        LOCAL_SPLIT_PATH,
        train_indices=train_indices_array,
        val_indices=val_indices_array,
    )
    print(f"Đã tạo fixed split: {LOCAL_SPLIT_PATH}")


if __name__ == "__main__":
    main()
