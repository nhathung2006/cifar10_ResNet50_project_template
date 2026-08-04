# Phân loại CIFAR-10 bằng ResNet-50

Project sử dụng ResNet-50 cho ảnh CIFAR-10 kích thước `32 x 32`. Dữ liệu train và validation dùng một fixed split stratified, để các project model khác có thể đánh giá trên cùng một tập dữ liệu.

## 1. Cấu trúc chính

```text
config.py
create_fixed_split.py
data.py
model.py
engine.py
train.py
evaluate.py
predict.py
utils.py
splits/
└── cifar10_seed42_val10.npz
outputs/
```

`DATA_DIR` mặc định là `/kaggle/input/cifar10-python`. Dataset này phải chứa CIFAR-10 dạng gốc để `torchvision.datasets.CIFAR10` đọc được, và project luôn dùng `download=False`.

## 2. Tạo fixed split dùng chung

Chạy một lần trong project:

```powershell
python create_fixed_split.py
```

Script đọc 50.000 ảnh train, chia theo `SEED = 42` và `VAL_RATIO = 0.10`, tạo:

- 45.000 ảnh train, 4.500 ảnh mỗi lớp.
- 5.000 ảnh validation, 500 ảnh mỗi lớp.
- Không trùng, không thiếu và không có index ngoài phạm vi.

Nếu file đã tồn tại, script không ghi đè.

## 3. Dùng split trên Kaggle

1. Upload `splits/cifar10_seed42_val10.npz` thành Kaggle Dataset có tên `cifar10-fixed-split-v1`.
2. Trong mỗi project model, chọn **Add Input** cho dataset này.
3. Project sẽ ưu tiên đọc:
   `/kaggle/input/cifar10-fixed-split-v1/cifar10_seed42_val10.npz`.
4. Nếu đường dẫn Kaggle không tồn tại, project đọc file local tại `splits/cifar10_seed42_val10.npz`.

Các project model khác cần dùng đúng file này, không tự chia lại train/validation.

## 4. Cài đặt và chạy

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python create_fixed_split.py
python train.py
python evaluate.py
```

Không chạy `train.py` nếu chỉ muốn kiểm tra hoặc tạo split. Lệnh dự đoán ảnh:

```powershell
python predict.py --image "duong_dan_anh.png" --top-k 5
```

Train có augmentation; validation và test chỉ dùng `ToTensor` và Normalize CIFAR-10. Tập test luôn dùng `train=False`, gồm 10.000 ảnh và `shuffle=False`.

Checkpoint ResNet-50 và các kết quả được lưu trong `outputs/` theo tên đã cấu hình trong `config.py`.
