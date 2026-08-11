import tempfile
import unittest
from pathlib import Path

from config import MIN_EARLY_STOPPING_EPOCH, _find_split_dir
from train import should_stop_early


class DatasetLookupTest(unittest.TestCase):
    def test_find_nested_training_and_test_split_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_dir = root / "training_set" / "training_set"
            test_dir = root / "test_set" / "test_set"
            for split_dir in (train_dir, test_dir):
                (split_dir / "cats").mkdir(parents=True)
                (split_dir / "dogs").mkdir()

            found_train_dir = _find_split_dir(root, "training_set")
            found_test_dir = _find_split_dir(root, "test_set")

            self.assertEqual(found_train_dir, train_dir)
            self.assertEqual(found_test_dir, test_dir)
            self.assertNotEqual(found_train_dir, found_test_dir)

    def test_early_stopping_cannot_trigger_before_epoch_110(self) -> None:
        self.assertEqual(MIN_EARLY_STOPPING_EPOCH, 110)
        self.assertFalse(should_stop_early(109, 10_000))
        self.assertTrue(should_stop_early(110, 25))
        self.assertFalse(should_stop_early(110, 24))


if __name__ == "__main__":
    unittest.main()
