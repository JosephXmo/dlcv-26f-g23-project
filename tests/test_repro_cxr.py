import csv
import gzip
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cxr_warning.features import LAB_COLUMNS, LEADS
from cxr_warning.repro import cxr


def write_csv(path, columns, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


class ReproCxrTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_cohort_cli_preserves_endpoint_and_coverage_rules(self):
        root = self.root
        write_csv(root / "train.csv", ("hadm_id", "subject_id"), [(i, i) for i in range(1, 8)])
        write_csv(root / "val.csv", ("hadm_id", "subject_id"), [(9, 1)])
        write_csv(root / "test.csv", ("hadm_id", "subject_id", "label_hadm_id", "label"),
                  [(1, 1, 8, 0), (8, 8, 8, 1), (10, 10, 10, 1), (10, 10, 10, 1)])
        admit, index, boundary, discharge = "2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"
        icu = sorted(cxr.ICUS)[0]
        write_csv(root / "symile_mimic_data.csv", (
            "hadm_id", "admittime", "dischtime", "cxr_StudyDateTime", "cxr_ViewPosition", "cxr_path",
            "hospital_expire_flag", "deathtime", "ecg_time", "ecg_path",
        ), [(i, admit, discharge, index, "AP", "image", int(i == 8), boundary if i == 8 else "",
             boundary if i == 1 else index, "files/record.hea") for i in range(1, 11)])
        admissions, transfers, out = [root / name for name in ("admissions.csv.gz", "transfers.csv", "cohort.csv")]
        write_csv(admissions, ("hadm_id",), [(i,) for i in range(1, 11) if i != 7])
        write_csv(transfers, ("hadm_id", "subject_id", "careunit", "intime", "outtime"), [
            (1, 1, "Medicine", admit, discharge),  # No new transfers, but continuous coverage.
            (2, 2, "Medicine", admit, "2020-01-03"),  # Gap.
            (3, 3, "Medicine", admit, "2020-01-03"), (3, 3, icu, boundary, discharge),  # Event despite gap.
            (4, 4, "Medicine", admit, discharge), (4, 4, "Medicine", admit, discharge),  # Ambiguous.
            (5, 5, "Medicine", admit, discharge), (5, 5, "Unknown", "2020-01-03", boundary),
            (6, 6, "Medicine", admit, discharge), (6, 6, "", "2020-01-03", ""),
            *[(i, i, "Medicine", admit, discharge) for i in range(7, 11)],
        ])
        with redirect_stdout(io.StringIO()) as audit:
            cxr.main(["cohort", "--symile", str(root), "--admissions", str(admissions),
                      "--transfers", str(transfers), "--out", str(out)])
        rows = list(cxr.read_rows(out, cxr.MANIFEST))
        self.assertEqual([(r["hadm_id"], r["split"], r["risk"]) for r in rows],
                         [("1", "train", "0"), ("3", "train", "1"), ("8", "test", "1")])
        self.assertEqual(rows[0]["ecg_path"], "")
        self.assertEqual(rows[1]["icu_48h"], "1")
        self.assertEqual(rows[2]["death_48h"], "1")
        self.assertEqual(rows[2]["lead_hours"], "48.0")
        for reason in ("gap_in_valid_known_unit_intervals", "unknown_unit_in_window", "trailing_unbounded_transfer"):
            self.assertIn(reason, audit.getvalue())

    def test_extract_cli_keeps_manifest_order_and_negative_embeddings(self):
        import torch

        class Backbone:
            def features(self, tensor):
                if tensor.shape[1:] != (3, 224, 224):
                    raise AssertionError("Expected a 224px center crop")
                return tensor[:, :1, :1, :1].expand(-1, 1024, 1, 1)

        root = self.root
        for split, ids, values in (("train", [1, 1], [-2, 9]), ("test", [2, 2], [9, -3])):
            directory = root / split
            directory.mkdir()
            np.save(directory / f"hadm_id_{split}.npy", ids)
            images = np.full((2, 3, 320, 320), 99, dtype=np.float32)
            images[:, :, 48:-48, 48:-48] = np.array(values)[:, None, None, None]
            np.save(directory / f"cxr_{split}.npy", images)
        np.save(root / "test/label_test.npy", [0, 1])
        np.save(root / "test/label_hadm_id_test.npy", [1, 2])
        manifest, labs, out = [root / f for f in ("manifest.csv", "labs.csv.gz", "features.npz")]
        admit, index = "2020-01-02", "2020-01-03"
        write_csv(manifest, ("hadm_id", "subject_id", "split", "risk", "admittime", "cxr_StudyDateTime", "ecg_path"),
                  [(2, 20, "test", 0, admit, index, ""), (1, 10, "train", 1, admit, index, "")])
        write_csv(labs, LAB_COLUMNS, [
            (10, 1, 4, "u", "bad", admit, admit),
            (10, 1, 4, "u", 9, admit, admit),
            (10, "", 4, " u ", 2, "2020-01-01", index),  # Earliest chart; blank hadm before admit.
            (10, 999, 4, "u", 8, "2020-01-01", admit),
            (10, 1, 4, "u", 7, "2020-01-01", "2020-01-04"),  # Store after index.
            (20, 2, 5, "u", 3, admit, admit),  # Test-only panel.
        ])
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        with patch.object(cxr, "_backbone", return_value=Backbone()):
            cxr.main(["extract", "--manifest", str(manifest), "--data-npy", str(root), "--ecg-root", str(root),
                      "--labevents", str(labs), "--out", str(out), "--batch-size", "2"])
        with np.load(out, allow_pickle=False) as cache:
            self.assertEqual(set(cache.files), {"hadm_id", "cxr", "ecg", "labs", "source", "lab_panels"})
            np.testing.assert_array_equal(cache["hadm_id"], ["2", "1"])
            np.testing.assert_array_equal(cache["cxr"], np.repeat([[-3], [-2]], 1024, axis=1))
            np.testing.assert_array_equal(cache["labs"], [[np.nan], [2]])
            np.testing.assert_array_equal(cache["lab_panels"], [["4", "u"]])
            self.assertEqual(cache["ecg"].shape, (2, 84))
            self.assertTrue(np.isnan(cache["ecg"]).all())

    @unittest.skipUnless(importlib.util.find_spec("wfdb"), "wfdb is an optional dependency")
    def test_ecg_features_match_tracked_helper(self):
        import wfdb

        signal = np.arange(48).reshape(4, 12).astype(float)
        wfdb.wrsamp("record", fs=500, units=["mV"] * 12, sig_name=list(LEADS),
                    p_signal=signal, fmt=["16"] * 12, write_dir=str(self.root))
        rows = [{"ecg_path": "record.hea"}, {"ecg_path": ""}, {"ecg_path": "record.hea"}]
        vectors = cxr._ecg_features(rows, self.root)
        expected = cxr.ecg_features([{"ecg_path": "record"}], self.root)[0]
        np.testing.assert_array_equal(vectors[[0, 2]], [expected, expected])
        self.assertTrue(np.isnan(vectors[1]).all())

    def test_ecg_rejects_escaped_paths_and_header_references(self):
        root = self.root
        for path in ("../record", "/record", "https://example.org/record"):
            with self.assertRaisesRegex(ValueError, "Invalid ECG record path"):
                cxr._ecg_pair(root, path)
        (root / "escape.hea").symlink_to(root.parent / "out-of-root.hea")
        with self.assertRaisesRegex(ValueError, "within the input root"):
            cxr._ecg_pair(root, "escape")
        (root / "record.hea").write_text("record 12 500 2\n" + "../other.dat 16\n" * 12, encoding="ascii")
        with self.assertRaisesRegex(ValueError, "paired signal file"):
            cxr._ecg_pair(root, "record")


if __name__ == "__main__":
    unittest.main()
