from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from config import CLASS_NAMES, IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, OUTPUT_DIR, SEED
from data import create_dataloaders
from utils import get_device, set_seed

IMAGE_COUNT = 50


def main() -> None:
    set_seed(SEED)
    train_loader, _, _ = create_dataloaders(get_device())
    output_dir = OUTPUT_DIR / "training_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    grid_images, grid_labels = [], []
    saved_count = 0
    for images, labels in train_loader:
        for batch_index in range(images.size(0)):
            if saved_count >= IMAGE_COUNT:
                break
            input_tensor = images[batch_index].cpu()
            if tuple(input_tensor.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Kích thước tensor không đúng: {tuple(input_tensor.shape)}")
            image_array = ((input_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy())
            class_name = CLASS_NAMES[int(labels[batch_index])]
            output_path = output_dir / f"{saved_count + 1:03d}_{class_name}.png"
            Image.fromarray(image_array, mode="RGB").save(output_path)
            if Image.open(output_path).size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError("PNG xuất ra không đúng kích thước.")
            grid_images.append(image_array)
            grid_labels.append(class_name)
            saved_count += 1
        if saved_count >= IMAGE_COUNT:
            break
    if saved_count != IMAGE_COUNT:
        raise RuntimeError(f"Chỉ xuất được {saved_count}/{IMAGE_COUNT} ảnh training.")
    figure, axes = plt.subplots(5, 10, figsize=(10, 5))
    for axis, image, name in zip(axes.flat, grid_images, grid_labels):
        axis.imshow(image)
        axis.set_title(name, fontsize=6)
        axis.axis("off")
    figure.tight_layout(pad=0.2)
    figure.savefig(OUTPUT_DIR / "training_inputs_grid.png", dpi=100)
    plt.close(figure)


if __name__ == "__main__":
    main()
