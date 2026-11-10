"""Exercise the notebook's real data cell without launching a kernel or training."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np

from har_project.data import file_hash


class NotebookInspectionTests(unittest.TestCase):
    def data_cell(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/HAR_Research.ipynb"
        notebook = nbformat.read(path, as_version=4)
        return next(cell.source for cell in notebook.cells if "load-data" in cell.metadata.get("tags", []))

    def fixture(self, root):
        result = root / "results"
        (result / "prepared").mkdir(parents=True)
        np.savez(result / "prepared/dataset.npz", x_train=np.ones((2, 128, 9)))
        np.savez(result / "normalization.npz", mean=np.zeros(9), std=np.ones(9))
        (result / "split.json").write_text('{"validation_subjects": [5]}')
        source = root / "Dataset/UCI HAR Dataset/train/y_train.txt"
        source.parent.mkdir(parents=True)
        source.write_text("1\n2\n")
        summary = {key: {"shape": [2, 128, 9], "subjects": [5]} for key in ("fit_train", "validation", "test")}
        summary.update(subject_overlap=[], prepared_sha256=file_hash(result / "prepared/dataset.npz"), input_sha256={"train/y_train.txt": file_hash(source)})
        (result / "data_summary.json").write_text(json.dumps(summary))
        return result, source

    def test_inspect_never_rewrites_preparation_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, _ = self.fixture(root)
            files = [result / p for p in ("prepared/dataset.npz", "normalization.npz", "split.json", "data_summary.json")]
            before = [(file_hash(p), p.stat().st_mtime_ns) for p in files]
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(self.data_cell(), "notebook_data_cell", "exec"), {"ROOT": root, "MODE": "inspect", "json": json})
            self.assertEqual(before, [(file_hash(p), p.stat().st_mtime_ns) for p in files])

    def test_inspect_rejects_changed_source_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source = self.fixture(root)
            source.write_text("6\n6\n")
            with self.assertRaises(AssertionError):
                exec(compile(self.data_cell(), "notebook_data_cell", "exec"), {"ROOT": root, "MODE": "inspect", "json": json})


if __name__ == "__main__":
    unittest.main()
