"""Runtime metrics that do not change the model architecture."""

from time import perf_counter

import torch
from torch import nn


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def start_timer(device: torch.device) -> float:
    """Synchronize CUDA before starting a wall-clock measurement."""
    synchronize(device)
    return perf_counter()


def stop_timer(start_time: float, device: torch.device) -> float:
    """Synchronize CUDA and return elapsed seconds."""
    synchronize(device)
    return perf_counter() - start_time


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_vram(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_vram_mb(device: torch.device) -> float | None:
    """Return peak allocated CUDA memory in MiB, or None without CUDA."""
    if device.type != "cuda":
        return None
    synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def format_vram(value_mb: float | None) -> str:
    return f"{value_mb:.2f} MB" if value_mb is not None else "N/A (CPU/MPS)"


def print_training_metrics(
    total_parameters: int,
    trainable_parameters: int,
    training_time_seconds: float,
    inference_time_seconds: float,
    peak_vram: float | None,
) -> None:
    """Print resource and runtime metrics."""
    print("\n===== THÔNG SỐ TÀI NGUYÊN =====")
    print(
        f"1. Parameters      : {total_parameters:,} "
        f"(trainable: {trainable_parameters:,})"
    )
    print(f"2. Training time   : {training_time_seconds:.3f} giây")
    print(f"3. Inference time  : {inference_time_seconds:.3f} giây")
    print(f"4. PEAK VRAM       : {format_vram(peak_vram)}")


def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Count multiply-accumulate operations for Conv2d and Linear layers."""
    total = 0
    hooks = []

    def hook(layer: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if isinstance(layer, nn.Conv2d):
            output_elements = output.numel()
            total += output_elements * (layer.kernel_size[0] * layer.kernel_size[1] * (layer.in_channels // layer.groups))
        elif isinstance(layer, nn.Linear):
            total += output.numel() * layer.in_features

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(input_shape, device=device))
    if was_training:
        model.train()
    for handle in hooks:
        handle.remove()
    return total


def model_size_mb(model: nn.Module) -> float:
    """Return the serialized state_dict size in MiB."""
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / (1024**2)
