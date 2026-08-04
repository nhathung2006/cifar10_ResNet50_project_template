from pathlib import Path


# ============================================================
# 1. Đường dẫn gốc của project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================
# 2. Đường dẫn dữ liệu CIFAR-10
# ============================================================

# Dataset CIFAR-10 đã Add Input trên Kaggle
# Bên trong thư mục này phải có:
# cifar-10-batches-py
KAGGLE_DATA_DIR = Path(
    "/kaggle/input/datasets/alifrahman/"
    "cifar10-python-dataset"
)

# Đường dẫn dùng khi chạy trên Windows.
# Cấu trúc yêu cầu:
# project/
# └── data/
#     └── cifar-10-batches-py/
LOCAL_DATA_DIR = PROJECT_ROOT / "data"

# Tự động chọn đường dẫn:
# - Nếu đang chạy trên Kaggle: dùng KAGGLE_DATA_DIR
# - Nếu chạy trên Windows: dùng LOCAL_DATA_DIR
DATA_DIR = (
    KAGGLE_DATA_DIR
    if KAGGLE_DATA_DIR.exists()
    else LOCAL_DATA_DIR
)


# ============================================================
# 3. Đường dẫn fixed split
# ============================================================

SPLIT_FILE_NAME = "cifar10_seed42_val10.npz"

# File split nằm trong project.
# create_fixed_split.py sẽ tạo file tại đây.
LOCAL_SPLIT_PATH = (
    PROJECT_ROOT
    / "splits"
    / SPLIT_FILE_NAME
)

# Sau khi upload fixed split thành Kaggle Dataset,
# file sẽ được đọc tại đường dẫn này.
KAGGLE_SPLIT_PATH = Path(
    "/kaggle/input/datasets/trannhathung2006/"
    "cifar10-fixed-split-v1/"
    "cifar10_seed42_val10.npz"
)

# Nếu đã Add Input fixed split trên Kaggle thì dùng file Kaggle.
# Nếu chưa có thì sử dụng file split trong project.
SPLIT_PATH = (
    KAGGLE_SPLIT_PATH
    if KAGGLE_SPLIT_PATH.exists()
    else LOCAL_SPLIT_PATH
)


# ============================================================
# 4. Đường dẫn kết quả huấn luyện
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_PATH = (
    OUTPUT_DIR / "best_resnet50_cifar10.pt"
)

HISTORY_CSV_PATH = (
    OUTPUT_DIR / "training_history.csv"
)

LOSS_PLOT_PATH = (
    OUTPUT_DIR / "loss_history.png"
)

ACCURACY_PLOT_PATH = (
    OUTPUT_DIR / "accuracy_history.png"
)

TEST_METRICS_PATH = (
    OUTPUT_DIR / "test_metrics.json"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR / "confusion_matrix.png"
)


# ============================================================
# 5. Cấu hình dữ liệu
# ============================================================

# Seed dùng để cố định kết quả chia dữ liệu
SEED = 42

# 10% của 50.000 ảnh train được dùng làm validation
VAL_RATIO = 0.10

BATCH_SIZE = 128
NUM_WORKERS = 2


# ============================================================
# 6. Cấu hình mô hình
# ============================================================

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


# ============================================================
# 7. Giá trị chuẩn hóa CIFAR-10
# ============================================================

CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)


# ============================================================
# 8. Cấu hình huấn luyện
# ============================================================

EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# ============================================================
# 9. Learning-rate scheduler
# ============================================================

# Giảm learning rate khi validation loss không cải thiện
LR_FACTOR = 0.5
LR_PATIENCE = 3
MIN_LEARNING_RATE = 1e-6


# ============================================================
# 10. Early stopping
# ============================================================

# Dừng huấn luyện nếu validation loss không cải thiện
EARLY_STOPPING_PATIENCE = 17