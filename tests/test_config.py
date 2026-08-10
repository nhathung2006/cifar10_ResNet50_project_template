import tempfile
import unittest
from pathlib import Path

from config import _find_split_dir


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


if __name__ == "__main__":
    unittest.main()
