# Phân loại CIFAR-10 bằng ResNet-50 trên Kaggle

Project này chỉ chạy trên Kaggle, sử dụng ResNet-50 cho ảnh CIFAR-10 `32 x 32`. Không tải CIFAR-10 từ Internet; tất cả dataset đều được đọc với `download=False`.

## 1. Add Input

Trong Kaggle Notebook, chọn **Add Input** cho dataset:

```text
cifar-10 python dataset của Alif Rahman
```

Dataset phải có thư mục:

```text
/kaggle/input/cifar10-python-dataset/cifar-10-batches-py/
├── batches.meta
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
└── test_batch
```

Nếu thư mục này không tồn tại, project dừng với thông báo yêu cầu Add Input dataset trên.

## 2. Copy project sang working

`/kaggle/input` là read-only. Copy project sang `/kaggle/working` trước khi tạo split hoặc chạy train:

```python
!cp -r /kaggle/input/<project-dataset>/cifar10_ResNet50_project_template /kaggle/working/
%cd /kaggle/working/cifar10_ResNet50_project_template
```

Thay `<project-dataset>` bằng tên dataset chứa project đã upload.

## 3. Tạo fixed split

Chạy một lần trong thư mục project tại `/kaggle/working`:

```python
!python create_fixed_split.py
```

Script đọc 50.000 ảnh train bằng `train=True`, chia stratified với `SEED = 42` thành:

- 45.000 ảnh train, 4.500 ảnh mỗi lớp.
- 5.000 ảnh validation, 500 ảnh mỗi lớp.

Script kiểm tra index không trùng, không thiếu và không ngoài phạm vi, sau đó lưu:

```text
splits/cifar10_seed42_val10.npz
```

File đã tồn tại sẽ không bị ghi đè.

## 4. Upload split dùng chung

Upload file sau thành Kaggle Dataset có tên:

```text
cifar10-fixed-split-v1
```

```text
splits/cifar10_seed42_val10.npz
```

Các project model khác cần **Add Input** dataset `cifar10-fixed-split-v1`. Project sẽ ưu tiên đọc:

```text
/kaggle/input/cifar10-fixed-split-v1/cifar10_seed42_val10.npz
```

Nếu chưa Add Input split dataset, project dùng file local trong `splits/` của bản copy tại `/kaggle/working`.

## 5. Chạy project

```python
!python train.py
!python evaluate.py
```

Tập train dùng augmentation. Validation và test chỉ dùng `ToTensor` và Normalize CIFAR-10. Test dùng `train=False`, có 10.000 ảnh và `shuffle=False`.

Không chạy huấn luyện nếu chỉ cần tạo fixed split. Checkpoint ResNet-50 giữ nguyên tên đã cấu hình trong `config.py` và tất cả output được ghi vào `/kaggle/working/.../outputs`.

Sau khi `train.py` hoàn tất, phần `THÔNG SỐ TÀI NGUYÊN` in ra:

- Tổng Parameters và số trainable Parameters.
- Tổng thời gian huấn luyện.
- Tổng thời gian suy luận trên tập test.
- PEAK VRAM đã cấp phát bởi PyTorch; CPU/MPS sẽ hiển thị `N/A`.

Các giá trị này cũng được lưu trong `outputs/test_metrics.json`. Có thể chạy bài test kiểm tra model và phần in thông số bằng:

```python
!python -m unittest discover -s tests -v
```
