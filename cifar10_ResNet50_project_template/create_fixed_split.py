import numpy as np
from torchvision.datasets import ImageFolder

from config import CLASS_NAMES, SEED, SPLIT_OUTPUT_PATH, TRAIN_DATA_DIR, VAL_RATIO, validate_data_dir


def main() -> None:
    if SPLIT_OUTPUT_PATH.exists():
        print(f"Split đã tồn tại, không ghi đè: {SPLIT_OUTPUT_PATH}")
        return
    validate_data_dir()
    dataset = ImageFolder(str(TRAIN_DATA_DIR))
    if dataset.classes != list(CLASS_NAMES):
        raise ValueError(f"Thứ tự class không đúng: {dataset.classes}; mong đợi {list(CLASS_NAMES)}")
    targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(SEED)
    train_indices, val_indices = [], []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_indices = np.flatnonzero(targets == class_id)
        if len(class_indices) < 2:
            raise ValueError(f"Lớp {class_name} phải có ít nhất 2 ảnh.")
        rng.shuffle(class_indices)
        val_count = max(1, int(round(len(class_indices) * VAL_RATIO)))
        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())
        print(f"{class_name}: train={len(class_indices) - val_count}, validation={val_count}")
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train_array = np.asarray(train_indices, dtype=np.int64)
    val_array = np.asarray(val_indices, dtype=np.int64)
    combined = np.concatenate((train_array, val_array))
    if np.intersect1d(train_array, val_array).size or len(np.unique(combined)) != len(dataset):
        raise ValueError("Split có index trùng hoặc thiếu index.")
    if not np.array_equal(np.sort(combined), np.arange(len(dataset))):
        raise ValueError("Split có index ngoài phạm vi hoặc thiếu index.")
    train_counts = np.bincount(targets[train_array], minlength=len(CLASS_NAMES))
    val_counts = np.bincount(targets[val_array], minlength=len(CLASS_NAMES))
    if np.any(val_counts < 1) or np.any(train_counts < 1):
        raise ValueError("Mỗi class phải xuất hiện trong cả train và validation.")
    SPLIT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SPLIT_OUTPUT_PATH, train_indices=train_array, val_indices=val_array)
    print(f"Đã tạo fixed split: {SPLIT_OUTPUT_PATH}")
    print(f"Tổng train={len(train_array)}, validation={len(val_array)}")


if __name__ == "__main__":
    main()
