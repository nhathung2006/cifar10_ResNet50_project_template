from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
KAGGLE_DATA_ROOT = Path("/kaggle/input/cat-and-dog")
LOCAL_DATA_ROOT = PROJECT_ROOT / "data"

SEED = 42
VAL_RATIO = 0.10
BATCH_SIZE = 64
NUM_WORKERS = 2

NUM_CLASSES = 2
CLASS_NAMES = ("cats", "dogs")
IMAGE_SIZE = 160
EVAL_RESIZE_SIZE = 176
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

MODEL_NAME = "ResNet50-CatDog"
EPOCHS = 160
LEARNING_RATE = 0.03
MOMENTUM = 0.90
WEIGHT_DECAY = 4e-5
MIN_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 25

SPLIT_FILE_NAME = "catdog_seed42_val10.npz"
LOCAL_SPLIT_PATH = PROJECT_ROOT / "splits" / SPLIT_FILE_NAME
KAGGLE_WORKING_SPLIT_PATH = Path("/kaggle/working/splits") / SPLIT_FILE_NAME

if Path("/kaggle/working").is_dir():
    SPLIT_OUTPUT_PATH = KAGGLE_WORKING_SPLIT_PATH
else:
    SPLIT_OUTPUT_PATH = LOCAL_SPLIT_PATH


def _contains_classes(path: Path) -> bool:
    return all((path / class_name).is_dir() for class_name in CLASS_NAMES)


def _find_split_dir(root: Path, split_name: str) -> Path | None:
    if not root.is_dir():
        return None
    preferred = root / split_name
    if _contains_classes(preferred):
        return preferred
    candidates = sorted(
        (path for path in root.rglob("*") if path.is_dir() and _contains_classes(path)),
        key=lambda path: (len(path.parts), str(path)),
    )
    return candidates[0] if candidates else None


def find_dataset_dirs() -> tuple[Path | None, Path | None]:
    roots = []
    if KAGGLE_DATA_ROOT.exists():
        roots.append(KAGGLE_DATA_ROOT)
    if Path("/kaggle/input").is_dir():
        roots.append(Path("/kaggle/input"))
    roots.append(LOCAL_DATA_ROOT)

    for root in roots:
        train_dir = _find_split_dir(root, "training_set")
        test_dir = _find_split_dir(root, "test_set")
        if train_dir is not None and test_dir is not None:
            return train_dir, test_dir
    return None, None


TRAIN_DATA_DIR, TEST_DATA_DIR = find_dataset_dirs()
DATA_DIR = TRAIN_DATA_DIR


def validate_data_dir() -> None:
    train_dir, test_dir = find_dataset_dirs()
    missing = []
    if train_dir is None:
        missing.append("thư mục training_set trực tiếp chứa cats/ và dogs/")
    if test_dir is None:
        missing.append("thư mục test_set trực tiếp chứa cats/ và dogs/")
    if missing:
        raise FileNotFoundError(
            "Không tìm thấy dữ liệu Cat/Dog. Thiếu: " + "; ".join(missing) + ". "
            "Hãy Add Input dataset Cat and Dog hoặc đặt dữ liệu vào project/data."
        )
    print(f"Training dataset: {train_dir}")
    print(f"Test dataset    : {test_dir}")


def find_split_path() -> Path:
    search_roots = []
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        search_roots.extend(sorted(kaggle_input.rglob(SPLIT_FILE_NAME)))
    search_roots.append(LOCAL_SPLIT_PATH)
    if KAGGLE_WORKING_SPLIT_PATH not in search_roots:
        search_roots.append(KAGGLE_WORKING_SPLIT_PATH)
    for path in search_roots:
        if path.is_file():
            return path
    return SPLIT_OUTPUT_PATH


SPLIT_PATH = find_split_path()

if Path("/kaggle/working").is_dir():
    OUTPUT_DIR = Path("/kaggle/working/catdog_resnet50_outputs")
else:
    OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_PATH = OUTPUT_DIR / "best_resnet50_catdog.pt"
HISTORY_CSV_PATH = OUTPUT_DIR / "training_history.csv"
LOSS_PLOT_PATH = OUTPUT_DIR / "loss_history.png"
ACCURACY_PLOT_PATH = OUTPUT_DIR / "accuracy_history.png"
TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
