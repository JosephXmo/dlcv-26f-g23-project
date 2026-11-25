from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from cxr_warning.cohort import write_rows
from cxr_warning.repro.cxr_logistic import fit, load_train_val


class CxrLogisticTest(unittest.TestCase):
    def test_fit_ignores_test_values_and_checks_prefix_order(self):
        with TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "cohort.csv"
            features = Path(tmp) / "features.npz"
            ids = np.array([str(i) for i in range(51)])
            rows = [dict(hadm_id=key, subject_id=key, split="train" if i < 40 else "val" if i < 50 else "test",
                         risk=i % 2 if i < 50 else "not a label") for i, key in enumerate(ids)]
            write_rows(manifest, rows[0].keys(), rows)
            rng = np.random.default_rng(24)
            cxr, ecg, labs = (rng.normal(size=(len(rows), dim)).astype(np.float32) for dim in (1024, 84, 50))
            cxr[:, 0] += np.arange(len(rows)) % 2
            cxr[-1] = np.inf
            labs[0, 0] = np.nan
            np.savez_compressed(features, hadm_id=ids, cxr=cxr, ecg=ecg, labs=labs)
            val = fit(manifest, features)["validation"]
            self.assertEqual(set(val), {"auprc", "auroc", "brier"})
            self.assertTrue(all(0 <= value <= 1 for value in val.values()))
            ids[:2] = ids[:2][::-1]
            np.savez_compressed(features, hadm_id=ids, cxr=cxr, ecg=ecg, labs=labs)
            with self.assertRaisesRegex(ValueError, "admission order"):
                load_train_val(manifest, features)
