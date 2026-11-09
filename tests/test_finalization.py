"""Prevent test-driven selection, inconsistent seed groups and mutable test locks."""
import json
import tempfile
import unittest
from pathlib import Path

try:
    from har_project import finalization
except ImportError:
    finalization = None


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(finalization, "Finalization is not implemented yet")

    def records(self):
        rows = []
        for name, score, parameters in (("cnn_a", 0.91, 100), ("cnn_b", 0.95, 200), ("cnn_c", 0.94, 300)):
            for seed in (42, 123, 2026):
                rows.append({"model": name, "seed": seed, "split": "validation", "kind": "official", "macro_f1": score, "parameters": parameters})
        return rows

    def test_model_selection_uses_validation_not_test(self):
        rows = self.records()
        for row in rows:
            row["test_macro_f1"] = 1.0 if row["model"] == "cnn_a" else 0.0
        self.assertEqual(finalization.select_model(rows), "cnn_b")

    def test_four_decimal_tie_prefers_fewer_parameters(self):
        rows = self.records()
        for row in rows:
            row["macro_f1"] = {"cnn_a": 0.95001, "cnn_b": 0.95004, "cnn_c": 0.94}[row["model"]]
        self.assertEqual(finalization.select_model(rows), "cnn_a")

    def test_incomplete_seed_group_is_not_ranked(self):
        with self.assertRaisesRegex(ValueError, "seed|nine|9"):
            finalization.select_model(self.records()[:-1])

    def test_test_split_cannot_be_used_for_selection(self):
        rows = self.records()
        rows[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation"):
            finalization.select_model(rows)

    def test_missing_lock_blocks_test_before_tensorflow_or_data_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "lock"):
                finalization.evaluate_frozen(Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_changed_checkpoint_is_detected_by_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "runs/cnn_b_seed42/model.keras"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"original checkpoint")
            from har_project.data import file_hash
            hashes = {"runs/cnn_b_seed42/model.keras": file_hash(model)}
            finalization.verify_file_hashes(root, hashes)
            model.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "SHA|changed|hash"):
                finalization.verify_file_hashes(root, hashes)

    def test_lock_cannot_reference_files_outside_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "results"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "outside|escape"):
                finalization.verify_file_hashes(root, {"../outside": "bad"})

    def test_verification_requires_completed_test_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "completed"):
                finalization.verify_results(root)

    def test_verification_detects_changed_result_before_using_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from har_project.data import file_hash
            result = root / "metrics_summary.csv"
            result.write_text("model,accuracy_mean\ncnn_b,0.9\n")
            hashes = {"metrics_summary.csv": file_hash(result)}
            result.write_text("model,accuracy_mean\ncnn_b,1.0\n")
            with self.assertRaisesRegex(ValueError, "changed|SHA|hash"):
                finalization.verify_file_hashes(root, hashes)

    def test_incomplete_lock_is_rejected_before_test_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "protocol_lock.json").write_text(json.dumps({"schema_version": 1, "files": {}}))
            with self.assertRaisesRegex(ValueError, "inventory|schema|complete"):
                finalization.evaluate_frozen(root)
            self.assertFalse((root / "test_evaluation_status.json").exists())

    def test_swapped_checkpoint_predictions_are_rejected(self):
        import numpy as np
        saved = np.array([[0.9, 0.1, 0, 0, 0, 0]])
        swapped = np.array([[0.1, 0.9, 0, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "checkpoint|validation"):
            finalization.verify_checkpoint_predictions(saved, swapped)


if __name__ == "__main__":
    unittest.main()
