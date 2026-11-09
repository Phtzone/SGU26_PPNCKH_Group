"""Catch mislabeled metrics, a holdout-selected baseline and wrong model outputs."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from har_project import experiment
except ImportError:
    experiment = None


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(experiment, "The experiment module is not implemented yet")

    def test_metrics_include_missing_classes_in_macro_average(self):
        result = experiment.classification_metrics(np.array([0, 1, 2]), np.array([0, 2, 2]))
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 5 / 18)
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), 3)
        self.assertEqual(result["report"]["LAYING"]["support"], 0)

    def test_baseline_class_is_chosen_from_training_labels(self):
        result = experiment.baseline_validation(np.array([2, 2, 0]), np.array([0, 0, 2]))
        self.assertEqual(result["predicted_class"], 2)
        self.assertAlmostEqual(result["accuracy"], 1 / 3)

    def test_cnn_b_returns_six_normalized_probabilities(self):
        model = experiment.build_model("cnn_b")
        probabilities = np.asarray(model(np.zeros((2, 128, 9), np.float32), training=False))
        self.assertEqual(probabilities.shape, (2, 6))
        np.testing.assert_allclose(probabilities.sum(axis=1), [1, 1], atol=1e-6)
        self.assertTrue((probabilities >= 0).all())

    def test_existing_run_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cnn_b_seed42"
            output.mkdir()
            (output / "evidence.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                experiment.train_run(Path(tmp) / "missing.npz", output)
            self.assertEqual((output / "evidence.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
