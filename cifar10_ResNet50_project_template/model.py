import torch
from torch import nn
from torchvision.models import resnet50

class RESNET50CIFAR10(nn.Module):
    """
    RESNET50-BN được điều chỉnh cho ảnh CIFAR-10 32 x 32.
    convolution ban đầu: chuyển kernal 7x7 -> 3x3, stride = 1
    bỏ maxpooling đầu tiên
    """

    def __init__(self, num_classes: int = 10,) -> None:
        super().__init__()

        self.network = resnet50(weights = None)

        self.network.conv1 = nn.Conv2d(
        in_channels = 3,
        out_channels = 64,
        kernel_size = 3,
        stride = 1,
        bias = False,
        )

        #không giảm 32x32 xuống 8x8 quá sớm
        self.network.maxpool = nn.Identity()

        #Thay classifier ImageNet 1000 lớp thành Cifar10
        in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(
            in_features=in_features,
            out_features=num_classes,
        )

        # Khởi tạo lại convolution đầu vừa thay
        nn.init.kaiming_normal_(
            self.network.conv1.weight,
            mode= "fan_out",
            nonlinearity='relu',
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


if __name__ == "__main__":
    model = RESNET50CIFAR10(num_classes=10)
    sample = torch.randn(8, 3, 32, 32)
    logits = model(sample)

    print(model)
    print(f"Input shape : {tuple(sample.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
