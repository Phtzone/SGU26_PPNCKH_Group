"""Catch channel/label misalignment, subject leakage and holdout-fitted scaling."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from har_project import data
except ImportError:
    data = None


CHANNELS = (
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(data, "The data preparation module is not implemented yet")

    def make_split(self, root, split="train"):
        folder = root / split
        (folder / "Inertial Signals").mkdir(parents=True)
        for channel_index, channel in enumerate(CHANNELS):
            values = np.tile(np.arange(128), (4, 1)) + channel_index * 1000
            values += np.arange(4)[:, None] * 100000
            np.savetxt(folder / "Inertial Signals" / f"{channel}_{split}.txt", values)
        np.savetxt(folder / f"y_{split}.txt", [6, 1, 4, 2], fmt="%d")
        np.savetxt(folder / f"subject_{split}.txt", [1, 5, 1, 11], fmt="%d")

    def test_loading_preserves_sample_time_channel_and_label_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_split(root)
            x, y, subjects = data.load_split(root, "train")
            self.assertEqual(x.shape, (4, 128, 9))
            self.assertEqual(x.dtype, np.float32)
            self.assertEqual(float(x[2, 7, 3]), 203007.0)
            np.testing.assert_array_equal(y, [5, 0, 3, 1])
            np.testing.assert_array_equal(subjects, [1, 5, 1, 11])

    def test_loading_rejects_channel_with_missing_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_split(root)
            np.savetxt(root / "train/Inertial Signals/body_acc_y_train.txt", np.zeros((3, 128)))
            with self.assertRaisesRegex(ValueError, "shape|rows|aligned"):
                data.load_split(root, "train")

    def test_subject_split_keeps_all_windows_of_a_person_together(self):
        fit, validation = data.subject_indices(np.array([1, 5, 1, 11, 2, 5]), (5, 11))
        np.testing.assert_array_equal(fit, [0, 2, 4])
        np.testing.assert_array_equal(validation, [1, 3, 5])

    def test_unknown_validation_person_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing|absent"):
            data.subject_indices(np.array([1, 5, 1]), (5, 11))

    def test_normalization_uses_fit_train_not_holdout(self):
        train = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
        holdout = np.array([[[100.0, 200.0]]], dtype=np.float32)
        mean, std = data.fit_normalizer(train)
        np.testing.assert_allclose(mean, [[[2.0, 3.0]]])
        np.testing.assert_allclose(std, [[[1.0, 1.0]]])
        scaled = data.normalize(holdout, mean, std)
        np.testing.assert_allclose(scaled, [[[98.0, 197.0]]])
        holdout[:] = -999
        data.normalize(holdout, mean, std)
        np.testing.assert_allclose(mean, [[[2.0, 3.0]]])
        np.testing.assert_allclose(std, [[[1.0, 1.0]]])

    def test_constant_channel_remains_finite(self):
        train = np.full((2, 128, 1), 7.0, dtype=np.float32)
        mean, std = data.fit_normalizer(train)
        self.assertTrue(np.isfinite(data.normalize(train, mean, std)).all())
        np.testing.assert_array_equal(data.normalize(train, mean, std), np.zeros_like(train))


if __name__ == "__main__":
    unittest.main()
