from pathlib import Path


# =========================================================
# Đường dẫn project
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# =========================================================
# Đường dẫn CIFAR-10
# =========================================================

KAGGLE_DATA_DIR = Path(
    "/kaggle/input/datasets/alifrahman/"
    "cifar10-python-dataset"
)

# Khi chạy trên Windows, torchvision sẽ tìm:
# data/cifar-10-batches-py
LOCAL_DATA_DIR = PROJECT_ROOT / "data"

DATA_DIR = (
    KAGGLE_DATA_DIR
    if KAGGLE_DATA_DIR.exists()
    else LOCAL_DATA_DIR
)
def validate_data_dir() -> None:
    cifar_folder = DATA_DIR / "cifar-10-batches-py"

    if not cifar_folder.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu CIFAR-10 tại: {cifar_folder}"
        )

# =========================================================
# Fixed split
# =========================================================

LOCAL_SPLIT_PATH = (
    PROJECT_ROOT
    / "splits"
    / "cifar10_seed42_val10.npz"
)

KAGGLE_SPLIT_PATH = Path(
    "/kaggle/input/datasets/trannhathung2006/"
    "cifar10-fixed-split-v1/"
    "cifar10_seed42_val10.npz"
)

SPLIT_PATH = (
    KAGGLE_SPLIT_PATH
    if KAGGLE_SPLIT_PATH.exists()
    else LOCAL_SPLIT_PATH
)


# =========================================================
# Kết quả huấn luyện
# =========================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_PATH = (
    OUTPUT_DIR / "best_resnet50_cifar10.pt"
)

HISTORY_CSV_PATH = OUTPUT_DIR / "training_history.csv"
LOSS_PLOT_PATH = OUTPUT_DIR / "loss_history.png"
ACCURACY_PLOT_PATH = OUTPUT_DIR / "accuracy_history.png"
TEST_METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"


# =========================================================
# Cấu hình dữ liệu
# =========================================================

SEED = 42
VAL_RATIO = 0.10
BATCH_SIZE = 128
NUM_WORKERS = 2


# =========================================================
# Cấu hình mô hình
# =========================================================

NUM_CLASSES = 10

CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# =========================================================
# Early stopping
# =========================================================

EARLY_STOPPING_PATIENCE = 20