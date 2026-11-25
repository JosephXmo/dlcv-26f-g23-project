import csv
import gzip
import stat
import tempfile
import unittest
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cxr_warning.repro import ward_labs as labs


START = datetime(2000, 1, 2)


def at(hours):
    return (START + timedelta(hours=hours)).isoformat(sep=" ")


def make_csv(path, fields, rows):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    return path


class WardLabsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_first_ward_endpoint_and_coverage_precedence(self):
        names = ("icu", "death", "covered", "gap", "unit", "unbounded", "flag", "outside", "prior", "none")
        admissions = make_csv(self.root / "admissions.csv.gz", labs.ADMISSION_FIELDS, [
            (name, f"patient-{name}", at(0), at(60 if name in {"icu", "gap"} else 36),
             at(36) if name == "death" else "", "1" if name == "flag" else "0") for name in names
        ])
        icu = "Medical Intensive Care Unit (MICU)"
        transfers = make_csv(self.root / "transfers.csv", labs.TRANSFER_FIELDS, [
            ("icu", "Medicine", at(4), at(10)), ("icu", "Medicine", at(2), at(4)),
            ("icu", icu, at(51), at(55)), ("icu", icu, at(50), at(52)),
            ("death", "Medicine", at(2), at(3)),
            ("covered", "Medicine", at(2), at(20)), ("covered", "Surgery", at(20), at(36)),
            ("covered", "other", at(36), at(37)), ("covered", icu, at(37), at(38)),
            ("gap", "Medicine", at(2), at(49)), ("gap", "other", at(49), at(50)),
            ("gap", "Medicine", at(49), ""), ("gap", icu, at(51), at(52)),
            ("unit", "Medicine", at(2), at(36)), ("unit", "other", at(10), at(12)),
            ("unit", "Medicine", at(20), ""),
            ("unbounded", "Medicine", at(2), at(36)), ("unbounded", "other", at(36), ""),
            ("flag", "Medicine", at(2), at(36)),
            ("outside", "Medicine", at(-1), at(2)), ("outside", "Medicine", at(2), at(36)),
            ("prior", icu, at(10), at(12)), ("prior", "Medicine", at(2), at(36)),
            ("prior", icu, at(2), at(4)), ("none", "Medicine", "invalid", at(36)),
        ])
        rows, stats = labs.build_index(admissions, transfers)
        by_id = {r["hadm_id"]: r for r in rows}
        self.assertEqual({h: r["endpoint"] for h, r in by_id.items()}, {
            "icu": "positive", "death": "positive", "covered": "negative", "gap": "unknown",
            "unit": "unknown", "unbounded": "unknown", "flag": "unknown",
        })
        self.assertEqual(by_id["icu"]["index_time"].isoformat(sep=" "), at(2))
        self.assertEqual(by_id["icu"]["icu_event"].isoformat(sep=" "), at(50))
        self.assertEqual(by_id["death"]["death_event"].isoformat(sep=" "), at(36))
        for name, reason in (("gap", "gap"), ("unit", "unknown_unit"), ("unbounded", "trailing_unbounded")):
            self.assertEqual(by_id[name]["unknown_reason"], reason)
        self.assertEqual(stats["unknown_reason_death_flag_inconsistent"], 1)
        self.assertEqual(stats["excluded_index_outside_stay_or_bad_times"], 1)
        self.assertEqual(stats["excluded_icu_at_or_before_index"], 1)
        self.assertEqual(stats["excluded_no_ward_entry"], 1)
        self.assertTrue(stats["check_partition"])

        out = self.root / "cohort" / "out"
        labs.main(["cohort", "--admissions", str(admissions), "--transfers", str(transfers), "--out", str(out)])
        self.assertEqual({p.name for p in out.iterdir()}, {"expansion_index.csv", "expansion_index_log.json"})
        self.assertEqual(stat.S_IMODE((out / "expansion_index.csv").stat().st_mode), 0o600)

    def test_lab_matching_record_ranking_and_earliest_chart_store(self):
        patients = {
            "a": [(0, "A", at(0), at(-24), at(24), at(30))],
            "b": [(1, "B", at(0), at(-24), at(24), at(10))],
            "c": [(2, "C", at(0), at(-24), at(24), at(10))],
            "d": [(3, "D", at(0), at(-24), at(24), at(10))],
        }
        events = make_csv(self.root / "labs.csv.gz", labs.LAB_FIELDS, [
            ("a", "A", "z", 9, at(0), at(0), "u"),
            ("a", "", "z", 8, at(-24), at(2), "u"),
            ("a", "A", "z", 7, at(-24), at(1), "u"),
            ("a", "A", "z", 6, at(-24), at(1), "u"),
            ("a", "A", "a", 2, at(24), at(30), "u"),
            ("a", "A", "a", 3, at(0), at(0), "v"),
            ("b", "B", "a", 4, at(0), at(0), "u"),
            ("c", "C", "validation-only", 5, at(0), at(0), "u"),
            ("d", "", "reject", 1, at(0), at(0), "u"),
            ("d", "A", "reject", 1, at(-1), at(0), "u"),
            ("d", "D", "reject", 1, at(0), at(11), "u"),
            ("d", "D", "reject", 1, at(0), at(-1), "u"),
            ("d", "D", "reject", 1, at(0), "", "u"),
            ("d", "D", "reject", "inf", at(0), at(0), "u"),
            ("d", "D", "reject", "bad", at(0), at(0), "u"),
            ("a", "A", "z", 0, at(-25), at(0), "u"),
            ("a", "A", "a", 1, at(25), at(26), "u"),
        ])
        stats = Counter()
        indices, panel, matrix = labs._lab_matrix(events, patients, ["train", "train", "val", "test"], stats)
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(panel, [("z", "u"), ("a", "u"), ("a", "v")])
        np.testing.assert_array_equal(matrix, [[7, 2, 3], [np.nan, 4, np.nan], [np.nan] * 3])
        self.assertEqual(stats["pass1_time_rejected"], 4)
        self.assertEqual(stats["pass1_skipped_nonfinite"], 1)

    def test_c_selection_scores_test_only_after_final_fit(self):
        labels = np.array([0, 1] * 6)
        splits = np.array(["train"] * 4 + ["val"] * 2 + ["test"] * 6)
        predictions = []

        class LogisticStub:
            def __init__(self, *, C, **kwargs):
                self.c = C

            def fit(self, x, y):
                np.testing.assert_array_equal(y, labels[:4])
                return self

            def predict_proba(self, x):
                predictions.append((self.c, len(x)))
                p = np.tile([0.2, 0.8] if self.c == 0.3 else [0.8, 0.2], len(x) // 2)
                return np.column_stack([1 - p, p])

        with patch("sklearn.linear_model.LogisticRegression", LogisticStub):
            chosen, grid, metrics = labs._fit(labels[:, None].astype(np.float32), labels, splits)
        self.assertEqual(chosen, 0.3)
        self.assertEqual(grid["0.3"], 1.0)
        self.assertEqual(predictions, [(c, 2) for c in labs.C_GRID] + [(0.3, 4), (0.3, 2), (0.3, 6)])
        self.assertEqual(metrics["test_metrics"]["auroc"], 1.0)

    def test_train_cli_post_filter_splits_and_specimens(self):
        subjects = [f"p{i:02}" for i in range(40)]
        labels = [i % 2 for i in range(40)]
        assignment = labs._patient_split(subjects, labels)
        self.assertEqual(Counter(assignment.values()), {"train": 28, "val": 6, "test": 6})
        self.assertEqual(labs._patient_split(subjects[::-1], labels[::-1]), assignment)
        self.assertEqual(labs._patient_split(["p01", *subjects], [0, *labels]), assignment)
        self.assertEqual([assignment[p] for p in ("p00", "p01", "p02", "p03")],
                         ["train", "val", "train", "test"])
        omitted = {next(p for p in subjects if assignment[p] == name and p in subjects[::2])
                   for name in labs.SPLITS}
        rows, events = [], []
        for subject, label in zip(subjects, labels):
            rows.append((subject, subject, at(0), at(2), "positive" if label else "negative"))
            if subject not in omitted:
                for item, unit, value in (("marker", "u", label), ("marker", "v", label), ("other", "u", 1)):
                    events.append((subject, subject, item, value, at(0), at(2), unit))
        rows.extend([("unknown", "unknown", at(0), at(2), "unknown"),
                     ("repeat", "p01", at(0), at(2), "positive")])
        events.append(("p01", "repeat", "marker", 1, at(0), at(2), "u"))
        index = make_csv(self.root / "synthetic-index.csv", labs.INDEX_FIELDS[:4] + ("endpoint",), rows)
        labevents = make_csv(self.root / "labevents.csv", labs.LAB_FIELDS, events)
        dictionary = make_csv(self.root / "dict.csv.gz", ("itemid", "fluid", "category"), [
            ("marker", "Blood", "Chemistry"), ("other", "Urine", "Chemistry"),
        ])
        out = self.root / "train" / "out"
        with patch.object(labs, "_write_report", wraps=labs._write_report) as write_report:
            labs.main(["train", "--index", str(index), "--labevents", str(labevents),
                       "--out", str(out), "--d-labitems", str(dictionary)])
        (_, baseline), (_, verification) = [call.args for call in write_report.call_args_list]
        self.assertEqual({p.name for p in out.iterdir()}, {"labs_baseline_log.json", "verify_split_panel_log.json"})
        self.assertEqual(verification["denominators"], {
            "decidable": 41, "cohort_total": 38, "cohort_positive": 21, "cohort_patients": 37,
        })
        for name, after in verification["in_cohort_split_stats"].items():
            self.assertEqual(after["admissions"], baseline["splits"][name]["admissions"] - 1)
            self.assertEqual(after["positive_admissions"], baseline["splits"][name]["positives"])
            self.assertEqual(after["positive_admissions"] - after["positive_patients"],
                             int(assignment["p01"] == name))
        self.assertTrue(all(verification["assertions"].values()))
        self.assertEqual(verification["panel_specimen_check"], {
            "panel_size": 3, "fluid_counts": {"Blood": 2, "Urine": 1},
            "category_counts": {"Chemistry": 3}, "blood_fluid_items": 2, "non_blood_items": 1,
        })
        self.assertEqual(baseline["test_metrics"]["auprc"], 1.0)
        self.assertEqual(baseline["chosen_c"], labs.C_GRID[0])


if __name__ == "__main__":
    unittest.main()
