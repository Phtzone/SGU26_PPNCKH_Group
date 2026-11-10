"""Exercise notebook model code and output guards, not a wrapper around the CLI."""
import tempfile
import unittest
from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/HAR_Research.ipynb"


class TrainingNotebookTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(NOTEBOOK.is_file(), "The cell-by-cell training notebook is not implemented yet")
        self.notebook = nbformat.read(NOTEBOOK, as_version=4)

    def cell(self, tag):
        return next(c.source for c in self.notebook.cells if tag in c.metadata.get("tags", []))

    def guards(self):
        namespace = {"Path": Path}
        exec(compile(self.cell("output-guards"), "notebook_output_guards", "exec"), namespace)
        return namespace

    def test_training_cannot_write_into_original_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guard = self.guards()["validate_output"]
            for candidate in (root / "results", root / "results/nested"):
                with self.assertRaises(ValueError):
                    guard(candidate, root / "notebook_runs", root / "results")

    def test_output_name_cannot_escape_notebook_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                self.guards()["validate_output"](root / "elsewhere", root / "notebook_runs", root / "results")

    def test_existing_run_keeps_all_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "cnn_b_seed42"
            folder.mkdir()
            (folder / "evidence.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                self.guards()["create_run_folder"](folder)
            self.assertEqual((folder / "evidence.txt").read_text(), "keep")

    def test_sensor_only_run_is_rejected_before_architecture_output_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "notebook_runs"
            run = base / "v1"
            sensor = run / "sensors"
            sensor.mkdir(parents=True)
            (sensor / "evidence.txt").write_text("keep")
            guard = self.guards().get("reserve_research_run")
            self.assertIsNotNone(guard, "Need shared run reservation before either training loop")
            with self.assertRaises(FileExistsError):
                guard(run / "architectures", sensor, base, root / "results")
            self.assertFalse((run / "architectures").exists())
            self.assertEqual((sensor / "evidence.txt").read_text(), "keep")

    def test_invalid_seeds_are_rejected_before_creating_output(self):
        validate = self.guards()["validate_seeds"]
        for seeds in ([], [-1], [2**32], ["42/../../results"], [True], [42, 42]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                validate(seeds)

    def test_inline_model_returns_probabilities_for_each_architecture(self):
        import numpy as np
        import tensorflow as tf
        namespace = {"tf": tf, "LEARNING_RATE": 0.001, "DROPOUT": 0.3}
        exec(compile(self.cell("model-definition"), "notebook_model_definition", "exec"), namespace)
        for name in ("cnn_a", "cnn_b", "cnn_c"):
            model = namespace["build_cnn"](name)
            probabilities = np.asarray(model(np.zeros((2, 128, 9), np.float32), training=False))
            self.assertEqual(probabilities.shape, (2, 6))
            np.testing.assert_allclose(probabilities.sum(axis=1), [1, 1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
