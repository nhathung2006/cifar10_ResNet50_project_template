import io
import unittest
from contextlib import redirect_stdout

import torch
from torch import nn

from metrics import count_parameters, peak_vram_mb, print_training_metrics
from model import RESNET50CIFAR10


class MetricsTest(unittest.TestCase):
    def test_model_forward_and_parameter_metrics(self) -> None:
        model = RESNET50CIFAR10(num_classes=10).eval()
        total, trainable = count_parameters(model)

        with torch.inference_mode():
            output = model(torch.randn(1, 3, 32, 32))

        self.assertEqual(tuple(output.shape), (1, 10))
        self.assertGreater(total, 0)
        self.assertEqual(total, trainable)

    def test_metrics_are_printed(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            print_training_metrics(123, 100, 1.25, 0.05, None)

        text = output.getvalue()
        self.assertIn("Parameters", text)
        self.assertIn("Training time", text)
        self.assertIn("Inference time", text)
        self.assertIn("PEAK VRAM", text)

    def test_cpu_peak_vram_is_not_reported_as_zero(self) -> None:
        self.assertIsNone(peak_vram_mb(torch.device("cpu")))


if __name__ == "__main__":
    unittest.main()
