import torch
from torch import nn
from torchvision.models import resnet50

from config import IMAGE_SIZE, NUM_CLASSES


class ResNet50CatDog(nn.Module):
    """Standard torchvision ResNet50 with a two-logit Cat/Dog classifier."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.network = resnet50(weights=None)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == "__main__":
    model = ResNet50CatDog(num_classes=NUM_CLASSES)
    sample = torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE)
    logits = model(sample)
    print(f"Input shape : {tuple(sample.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
