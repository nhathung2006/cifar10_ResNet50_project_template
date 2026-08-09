import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from config import BATCH_SIZE, CHECKPOINT_PATH, CLASS_NAMES, NUM_CLASSES, NUM_WORKERS, OUTPUT_DIR, TEST_DATA_DIR, validate_data_dir
from data import get_eval_transform, seed_worker
from model import ResNet50CatDog
from utils import get_device, load_checkpoint

EXAMPLES_PER_PAIR = 9
CONFUSION_PAIRS = (("cats", "dogs"), ("dogs", "cats"))


def main() -> None:
    validate_data_dir()
    device = get_device()
    test_dataset = ImageFolder(str(TEST_DATA_DIR), transform=get_eval_transform())
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda", persistent_workers=NUM_WORKERS > 0, worker_init_fn=seed_worker)
    model = ResNet50CatDog(num_classes=NUM_CLASSES).to(device)
    load_checkpoint(CHECKPOINT_PATH, model, device)
    model.eval()
    counts = {pair: 0 for pair in CONFUSION_PAIRS}
    sample_index = 0
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images.to(device, non_blocking=device.type == "cuda")).argmax(dim=1).cpu()
            for batch_index, label in enumerate(labels):
                pair = (CLASS_NAMES[int(label)], CLASS_NAMES[int(predictions[batch_index])])
                if pair in counts and counts[pair] < EXAMPLES_PER_PAIR:
                    source_path, _ = test_dataset.samples[sample_index]
                    output_dir = OUTPUT_DIR / "confusion_examples" / f"{pair[0]}_{pair[1]}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    Image.open(source_path).convert("RGB").save(output_dir / f"{counts[pair] + 1:02d}.png")
                    counts[pair] += 1
                sample_index += 1
    for pair, count in counts.items():
        print(f"{pair[0]} -> {pair[1]}: {count}/{EXAMPLES_PER_PAIR} ảnh")


if __name__ == "__main__":
    main()
