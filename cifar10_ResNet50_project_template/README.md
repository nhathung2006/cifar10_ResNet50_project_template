# ResNet50 Cat/Dog trên Kaggle

Project phân loại ảnh mèo và chó bằng ResNet50 torchvision, huấn luyện từ đầu và không dùng pretrained weights.

## Dataset

Trong Kaggle chọn **Add Input** dataset Cat and Dog. Project ưu tiên `/kaggle/input/cat-and-dog/` và tự tìm các thư mục trực tiếp chứa `cats/` và `dogs/`, kể cả cấu trúc:

```text
training_set/training_set/cats
training_set/training_set/dogs
test_set/test_set/cats
test_set/test_set/dogs
```

Training set được chia stratified 90% train và 10% validation với seed 42. Toàn bộ test set được giữ độc lập. Split là `catdog_seed42_val10.npz`; script không ghi đè file đã tồn tại.

## Chạy trên Kaggle

```python
!python create_fixed_split.py
!python model.py
!python -m unittest discover -s tests -v
!python train.py
!python evaluate.py
```

Ví dụ dự đoán:

```python
!python predict.py --image /kaggle/working/example.jpg --top-k 2
```

Xuất ảnh dự đoán nhầm:

```python
!python export_confusion_examples.py
```

Model nhận input `N x 3 x 160 x 160` và trả về 2 logits: `0 = cats`, `1 = dogs`. Validation, test và prediction dùng resize 176, center crop 160, không augmentation. Model giữ nguyên stem chuẩn ResNet50 (7x7 stride 2 và maxpool 3x3 stride 2).

## Output

Kaggle lưu tại `/kaggle/working/catdog_resnet50_outputs/`; local lưu tại `outputs/`. Bao gồm checkpoint `best_resnet50_catdog.pt`, history, biểu đồ, metrics và confusion matrix. Metrics có parameters, thời gian train/inference, peak VRAM, MACs, FLOPs (`2 x MACs`) và kích thước state_dict. Hãy Save Version hoặc tải checkpoint trước khi kết thúc Kaggle session.
