import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import *
from data import create_dataloaders
from engine import evaluate, train_one_epoch
from metrics import count_macs, count_parameters, model_size_mb, peak_vram_mb, reset_peak_vram, start_timer, stop_timer
from model import ResNet50CatDog
from utils import ensure_directories, get_device, load_checkpoint, plot_history, save_checkpoint, save_history_csv, save_json, set_seed


def should_stop_early(epoch: int, epochs_without_improvement: int) -> bool:
    return (
        epoch >= MIN_EARLY_STOPPING_EPOCH
        and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
    )


def main() -> None:
    ensure_directories()
    set_seed(SEED)
    device = get_device()
    print(f"Thiết bị đang sử dụng: {device}")
    train_loader, val_loader, test_loader = create_dataloaders(device)
    model = ResNet50CatDog(num_classes=NUM_CLASSES).to(device)
    total_parameters, trainable_parameters = count_parameters(model)
    macs = count_macs(model, (1, 3, IMAGE_SIZE, IMAGE_SIZE))
    flops = 2 * macs
    state_dict_size = model_size_mb(model)
    reset_peak_vram(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LEARNING_RATE)
    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    training_start = start_timer(device)
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": float(epoch), "learning_rate": float(current_lr), "train_loss": float(train_loss), "train_accuracy": float(train_accuracy), "val_loss": float(val_loss), "val_accuracy": float(val_accuracy)})
        print(f"Epoch {epoch:02d}/{EPOCHS} | LR: {current_lr:.7f} | Train loss: {train_loss:.4f} | Train acc: {train_accuracy:.2f}% | Val loss: {val_loss:.4f} | Val acc: {val_accuracy:.2f}%")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(CHECKPOINT_PATH, model, optimizer, epoch, val_loss, val_accuracy)
            print(f"  -> Đã lưu model tốt nhất: {CHECKPOINT_PATH.name}")
        else:
            epochs_without_improvement += 1
            print(f"  -> Validation loss chưa cải thiện {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE} epoch.")
            if should_stop_early(epoch, epochs_without_improvement):
                print("Dừng sớm để hạn chế overfitting.")
                break
    training_time_seconds = stop_timer(training_start, device)
    save_history_csv(history)
    plot_history(history)
    checkpoint = load_checkpoint(CHECKPOINT_PATH, model, device)
    inference_start = start_timer(device)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    inference_time_seconds = stop_timer(inference_start, device)
    peak_vram = peak_vram_mb(device)
    test_metrics = {"best_epoch": int(checkpoint["epoch"]), "best_val_loss": float(checkpoint["val_loss"]), "best_val_accuracy": float(checkpoint["val_accuracy"]), "test_loss": float(test_loss), "test_accuracy": float(test_accuracy), "total_parameters": total_parameters, "trainable_parameters": trainable_parameters, "training_time_seconds": training_time_seconds, "inference_time_seconds": inference_time_seconds, "peak_vram_mb": peak_vram, "macs": macs, "flops": flops, "model_size_mb": state_dict_size, "classes": list(CLASS_NAMES)}
    save_json(TEST_METRICS_PATH, test_metrics)
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.2f}%")
    print(f"MACs: {macs:,} | FLOPs: {flops:,} | State_dict: {state_dict_size:.2f} MB")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
