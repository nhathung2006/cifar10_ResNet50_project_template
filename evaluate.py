import json

import torch
from torch import nn

from config import CHECKPOINT_PATH, CLASS_NAMES, IMAGE_SIZE, NUM_CLASSES, TEST_METRICS_PATH
from data import create_test_loader
from engine import evaluate_detailed
from metrics import count_macs, count_parameters, model_size_mb, peak_vram_mb, reset_peak_vram, start_timer, stop_timer
from model import ResNet50CatDog
from utils import ensure_directories, get_device, load_checkpoint, save_confusion_matrix, save_json


def main() -> None:
    ensure_directories()
    device = get_device()
    print(f"Thiết bị đang sử dụng: {device}")
    test_loader = create_test_loader(device)
    model = ResNet50CatDog(num_classes=NUM_CLASSES).to(device)
    total_parameters, trainable_parameters = count_parameters(model)
    macs = count_macs(model, (1, 3, IMAGE_SIZE, IMAGE_SIZE))
    checkpoint = load_checkpoint(CHECKPOINT_PATH, model, device)
    reset_peak_vram(device)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, confusion_matrix, class_correct, class_total = evaluate_detailed(model, test_loader, criterion, device, NUM_CLASSES)
    second_image, _ = test_loader.dataset[1]
    second_image = second_image.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
    model.eval()
    with torch.inference_mode():
        start = start_timer(device)
        model(second_image)
        inference_time = stop_timer(start, device)
    vram = peak_vram_mb(device)
    previous_metrics = {}
    if TEST_METRICS_PATH.is_file():
        with TEST_METRICS_PATH.open(encoding="utf-8") as file:
            previous_metrics = json.load(file)
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        per_class[name] = 100.0 * int(class_correct[index]) / int(class_total[index]) if class_total[index] else 0.0
        print(f"{name}: {per_class[name]:.2f}%")
    save_confusion_matrix(confusion_matrix, CLASS_NAMES)
    state_dict_size = model_size_mb(model)
    training_time = previous_metrics.get("training_time_seconds")
    metrics = {"best_epoch": int(checkpoint["epoch"]), "best_val_loss": float(checkpoint["val_loss"]), "best_val_accuracy": float(checkpoint["val_accuracy"]), "test_loss": float(test_loss), "test_accuracy": float(test_accuracy), "total_parameters": total_parameters, "trainable_parameters": trainable_parameters, "training_time_seconds": training_time, "inference_time_seconds": inference_time, "peak_vram_mb": vram, "macs": macs, "flops": 2 * macs, "model_size_mb": state_dict_size, "classes": list(CLASS_NAMES), "per_class_accuracy": per_class, "confusion_matrix": confusion_matrix.tolist()}
    save_json(TEST_METRICS_PATH, metrics)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.2f}%")
    print(f"Parameters: {total_parameters:,} (trainable: {trainable_parameters:,})")
    print(f"Training time: {training_time if training_time is not None else 'N/A'} seconds")
    print(f"Inference time: {inference_time:.3f} seconds | Peak VRAM: {vram if vram is not None else 'N/A'} MB")
    print(f"MACs: {macs:,} | FLOPs: {2 * macs:,} | State_dict: {state_dict_size:.2f} MB")
    print(f"Confusion matrix shape: {tuple(confusion_matrix.shape)}")


if __name__ == "__main__":
    main()
