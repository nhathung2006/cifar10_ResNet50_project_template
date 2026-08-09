import io
import unittest
from contextlib import redirect_stdout

import torch

from config import IMAGE_SIZE, NUM_CLASSES
from metrics import count_macs, count_parameters, model_size_mb, peak_vram_mb, print_training_metrics
from model import ResNet50CatDog


class MetricsTest(unittest.TestCase):
    def test_model_forward_and_metrics(self) -> None:
        model = ResNet50CatDog(num_classes=NUM_CLASSES).eval()
        total, trainable = count_parameters(model)
        with torch.inference_mode():
            output = model(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
        macs = count_macs(model, (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        flops = 2 * macs
        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertGreater(total, 0)
        self.assertGreater(trainable, 0)
        self.assertEqual(total, trainable)
        self.assertGreater(macs, 0)
        self.assertEqual(flops, 2 * macs)
        self.assertGreater(model_size_mb(model), 0)

    def test_metrics_are_printed(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            print_training_metrics(123, 100, 1.25, 0.05, None)
        text = output.getvalue()
        for name in ("Parameters", "Training time", "Inference time", "PEAK VRAM"):
            self.assertIn(name, text)

    def test_cpu_peak_vram_is_none(self) -> None:
        self.assertIsNone(peak_vram_mb(torch.device("cpu")))


if __name__ == "__main__":
    unittest.main()
