"""Protect scientific invariants at the notebook's callable cell boundaries."""
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/HAR_Research.ipynb'


class PracticalNotebookTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(NOTEBOOK.exists(), 'Practical research notebook is missing')
        self.nb = nbformat.read(NOTEBOOK, as_version=4)
        self.ns = dict(np=np, pd=pd, Path=Path, accuracy_score=accuracy_score,
                       f1_score=f1_score, confusion_matrix=confusion_matrix,
                       classification_report=classification_report)
        for tag in ('analysis-helpers', 'output-guards'):
            source = next(c.source for c in self.nb.cells if tag in c.metadata.get('tags', []))
            exec(compile(source, tag, 'exec'), self.ns)

    def test_channel_order_and_train_only_scaling(self):
        # Taking gyro before total_acc, or fitting on validation, must fail this fixture.
        x = np.zeros((2, 128, 9), np.float32)
        x[0, :, :] = np.arange(9)
        x[1, :, :] = np.arange(9) + 2
        holdout = np.full((1, 128, 9), 100, np.float32)
        fit, val, mean, std = self.ns['prepare_channels'](x, holdout, 'total_acc_gyro')
        np.testing.assert_allclose(mean.ravel(), [7, 8, 9, 4, 5, 6])
        np.testing.assert_allclose(std.ravel(), [1]*6)
        np.testing.assert_allclose(val[0, 0], [93, 92, 91, 96, 95, 94])
        self.assertEqual(fit.shape, (2, 128, 6))
        with self.assertRaises(ValueError):
            self.ns['prepare_channels'](x, holdout, 'typo')

    def test_metrics_count_sitting_standing_and_missing_classes(self):
        result = self.ns['calculate_metrics'](np.array([3, 3, 4, 4]), np.array([3, 4, 3, 4]))
        self.assertEqual(result['sit_stand_errors'], 2)
        self.assertEqual(result['total_errors'], 2)
        self.assertAlmostEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['macro_f1'], 1/6)
        self.assertEqual(result['confusion_matrix'][3][4], 1)
        self.assertEqual(result['sitting_to_standing'], 1)
        self.assertEqual(result['standing_to_sitting'], 1)
        self.assertEqual(result['recall_STANDING'], .5)

    def test_predictions_reject_corrupt_probabilities_or_row_alignment(self):
        frame = pd.DataFrame({'sample_index': [0, 1], 'subject_id': [2, 2],
                              'y_true': [0, 1], 'y_pred': [0, 1]})
        for i, activity in enumerate(self.ns['ACTIVITIES']):
            frame['p_' + activity] = [float(i == 0), float(i == 1)]
        self.ns['validate_predictions'](frame, 'sample_index')
        for mutation in ('probability', 'duplicate', 'prediction', 'fractional', 'negative'):
            changed = frame.copy()
            if mutation == 'probability': changed.loc[0, 'p_WALKING'] = .5
            if mutation == 'duplicate': changed.loc[1, 'sample_index'] = 0
            if mutation == 'prediction': changed.loc[0, 'y_pred'] = 1
            if mutation == 'fractional': changed['subject_id'] = [2.5, 2]
            if mutation == 'negative':
                changed.loc[0, 'p_WALKING'] = 1.2
                changed.loc[0, 'p_WALKING_UPSTAIRS'] = -.2
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.ns['validate_predictions'](changed, 'sample_index')

    def test_output_cannot_escape_or_overwrite_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, frozen = root/'practical_results', root/'results'
            reserve = self.ns['reserve_output']
            for bad in (frozen, frozen/'nested', base, root/'elsewhere', base/'..'/'escape'):
                with self.subTest(path=bad), self.assertRaises(ValueError):
                    reserve(bad, base, frozen)
            target = reserve(base/'v1', base, frozen)
            (target/'evidence.txt').write_text('keep')
            with self.assertRaises(FileExistsError): reserve(target, base, frozen)
            self.assertEqual((target/'evidence.txt').read_text(), 'keep')

    def test_invalid_seeds_cannot_start_run(self):
        for seeds in ([], [True], [-1], [2**32], [42,42], ['42']):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                self.ns['validate_seeds'](seeds)


if __name__ == '__main__':
    unittest.main()
